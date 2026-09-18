"""
FastAPI application for the podcast summarizer.
Exposes the full ingestion -> preprocessing -> summarization -> eval pipeline
as HTTP endpoints, making it easy to integrate with UIs, schedulers, or
other services.

Routes:
    GET /heath - liveness check
    POST /summarize - fetch + transcribe + summarize episode(s)
    POST /eval - score an existing summary
    GET /summaries - retrieve a specific summaries
    GET /summaries/{episode_id} - retrieve a specific summary
    GET /drift - latest drift report

Run locally:
    uvicorn api.main:app --reload --port 8000

Or via Docker;
    docker compose up

Requirements:
    pip install fastapi uvicorn[standard] pydantic python-dotenv
"""

import json 
import logging 
import os 
import sys 
from contextlib import asynccontextmanager
from pathlib import Path 
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

#add project root to path so we can import sibling packages
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.middleware import LoggingMiddleware, RateLimitMiddleware
from api.schemas import (
    SummarizedRequest,
    SummarizeResponse,
    EvalRequest,
    EvalOut,
    EpisodeSummaryOut,
    ChunkSummaryOut,
    RougeOut,
    BertScoreOut,
    LLMJudgeOut,
    HealthResponse,
)

#--Logging-----------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

#--Paths--------------------------------------------------------------

OUTPUTS_DIR = Path("data/outputs")
PROCESSED_DIR = Path("data/processed")
EVAL_DIR = Path("data/eval_results")

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

APP_VERSION = "0.1.0"

#--Lifespan (startup/shutdown)----------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavyweight resources once at startup rather than per-request.
    Whisper model loading takes several seconds, do it here.
    """
    logger.info("Starting podacast summarizer API v%s", APP_VERSION)

    #lazy-import pipeline modules so the app starts even if optional
    #deps (whisper, bert_score) aren't installed yet
    app.state.ready = True 
    logger.info("API ready")

    yield 

    logger.info("Shutting down")


#--App----------------------------------------------------------------

app = FastAPI(
    title="Podcast Summarizer API",
    description=(
        "End-to-end podcast summarization: fetch episodes via Listen Notes or RSS, "
        "transcribe with Whisper, summarize with an LLM, and evaluate summary quality."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)

#middleware-order matter: outer middleware runs first on request, last on response
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_request=int(os.getenv("RATE_LIMIT_REQUESTS", "10")),
    window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#--Helper: load pipeline modules lazily------------------------------------------------

def get_summarizer_client(provider: str, model: Optional[str]):
    """Import and instantiate the correct LLM client """
    from summarization.summarizer import build_client
    resolved_model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["anthropic"])
    return build_client(provider, resolved_model)


def run_full_pipeline(req: SummarizedRequest) -> list[dict]:
    """
    Synchronous pipeline: fetch -> preprocess -> summarize.
    Called in a thread pool via FastAPI's BackgroundTasks or run_in_executor
    so it doesn't block the event loop.
    """
    from ingestion.fetch_episodes import (
        ListenNotesClient,
        Transcriber,
        fetch_rss_episodes,
        run_pipeline as ingest_pipeline,
    )
    from ingestion.preprocess import process_file as preprocess_file
    from summarization.summarizer import (
        load_prompt,
        summarize_episode,
        Tracker,
    )

    #Ingestion
    if req.rss_url:
        episodes = fetch_rss_episodes(str(req.rss_url), max_results=req.max_episodes)
    else:
        client = ListenNotesClient(api_key=os.getenv("LISTEN_NOTES_API_KEY"))
        if req.query:
            raw = client.search_episodes(req.query, max_results=req.max_episodes)
        else:
            raw = client.get_podcast_episodes(req.podcast_id, max_results=req.max_episodes)
        episodes = [client.parse_episode(r) for r in raw]
    
    transcriber = None 
    if not req.skip_transcribe:
        transcriber = Transcriber(model_size=req.whisper_model.value)

    saved_paths = ingest_pipeline(episodes, transcriber, skip_existing=True)

    #Preprocessing
    preprocessed_paths = []
    for path in saved_paths:
        out = preprocess_file(
            input=path,
            output_dir=PROCESSED_DIR,
            max_tokens=req.max_tokens_per_chunk,
            overlap_tokens=req.overlap_tokens,
            skip_existing=True,
        )
        if out:
            preprocessed_paths.append(out)

    #Summarization
    llm_client - get_summarizer_client(req.provider.value, req.model)
    prompts, prompt_version = load_prompt(req.prompt_version)
    tracker = Tracker(enabled=False) #no W&B tracking via API by default

    summaries = []
    for path in preprocessed_paths:
        with open(path) as f:
            episode_data = json.load(f)

        summary = summarize_episode(
            episode=episode_data,
            client=llm_client,
            prompts=prompts,
            prompt_version=prompt_version,
            tracker=tracker,
            provider=req.provider.value,
        )

        #persist to disk
        out_path = OUTPUTS_DIR / f"{summary.episode_id}_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary.to_json(), f, indent=2, ensure_ascii=False)

        summaries.append(summary.to_json())

    return summaries 

def _log_to_db(summary_paths: list[Path]):
    """Background task: log completed summaries to the monitoring DB """
    try:
        from monitoring.log_io import get_connection, log_summary
        conn = get_connection()
        for path in summary_paths:
            with open(path) as f:
                data = json.load(f)
            log_summary(data, conn)
        conn.close()
    except Exception as e:
        logger.warning("Background DB logging failed: %s", e)


#--Routes---------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check, returns 200 when the API is ready to serve """
    provider = "anthropic"
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        model=DEFAULT_MODELS[provider],
        provider=provider,
    )

@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
def summarize(req: SummarizedRequest, background_tasks: BackgroundTasks):
    """
    End-to-end pipeline: fetch episode(s) -> transcribe -> chunk -> summarize

    Exactly one of `query`, `podcast_id`, or `rss_url` must be provided.
    Set `skip_transcribe=true` to use episode descriptions instead of audio
    (much faster, no Whisper dependency, but lower quality)
    """
    #validate that exactly one source is provided
    sources = [req.query, req.podcast_id, req.rss_url]
    if sum(s is not None for s in sources) != 1:
        raise HTTPException(
            status_code=422,
            detail="Provider exactly one of: query, podcast_id, rss_url",
        )
    
    if not os.getenv("LISTEN_NOTES_API_KEY") and not req.rss_url:
        raise HTTPException(
            status_code=500,
            detail="LISTEN_NOTES_API_KEY not configured on server",
        )
    
    try:
        raw_summaries = run_full_pipeline(req)
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))

    #convert to response schema
    out_summaries = []
    saved_paths = []
    for s in raw_summaries:
        out_summaries.append(EpisodeSummaryOut(
            episode_id=s["episode_id"],
            title=s["title"],
            podcast_name=s["podcast_name"],
            pub_date=s["pub_date"],
            final_summary=s["final_summary"],
            chunk_summaries=[ChunkSummaryOut(**c) for c in s["chunk_summaries"]],
            model=s["model"],
            provider=s["provider"],
            prompt_version=s["prompt_version"],
            total_input_tokens=s["total_input_tokens"],
            total_output_tokens=s["total_output_tokens"],
            total_latency_ms=s["total_latency_ms"],
            num_chunks=s["num_chunks"],
            summarized_at=s["summarized_at"],
        ))
        saved_paths.append(OUTPUTS_DIR / f"{s['episode_id']}_summary.json")

    #log to monitorning DB in the background, don't block the response
    background_tasks.add_task(_log_to_db, saved_paths)

    return SummarizeResponse(
        summaries=summaries,
        total_episodes=len(out_summaries),
        total_input_tokens=sum(s.total_input_tokens for s in out_summaries),
        total_output_tokens=sum(s.total_output_tokens for s in out_summaries),
    )


@app.post("/eval", response_model=EvalOut, tags=["Evaluation"])
def evaluate(req:EvalRequest):
    """
    Score an existing summary. Looks up the summary by episode_id from disk,
    then runs ROUGE (if reference provided), BERTScore, and LLM-as-judge
    """
    summary_path = OUTPUTS_DIR / f"{req.episode_id}_summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No summary found for episode_id '{req.episode_id}'. Run /summarize first"
        )

    with open(summary_path) as f:
        summary = json.load(f)

    try:
        from evaluation.run_eval import evaluate_summary, load_threshold
        thresholds = load_threshold()
        result = evaluate_summary(
            summary=summary,
            reference_text=req.reference_text,
            use_llm_judge=req.run_llm_judge,
            thresholds=thresholds,
        )
    except Exception as e:
        logger.exception("Eval failed for %s", req.episode_id)
        raise HTTPException(status_code=500, detail=str(e))

    #persist eval result
    eval_path = EVAL_DIR / f"{req.episode_id}_eval.json"
    with open(eval_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    #build response
    rouge_out = None 
    if result.rouge:
        rouge_out = RougeOut(
            rouge1_f=result.rouge.rouge1_f,
            rouge2_f=result.rouge.rouge2_f,
            rougeL_f=result.rouge.rougeL_f,
        )

    bert_out = None 
    if result.bertscore:
        bert_out = BertScoreOut(
            f1=result.bertscore.f1,
            precision=result.bertscore.precision,
            recall=result.bertscore.recall,
            model_used=result.bertscore.model_used,
        )

    judge_out = None 
    if result.llm_judge and not result.llm_judge.error:
        judge_out = LLMJudgeOut(
            faithfulness=result.llm_judge.faithfulness,
            coverage=result.llm_judge.coverage,
            fluency=result.llm_judge.fluency,
            conciseness=result.llm_judge.conciseness,
            overall=result.llm_judge.overall,
            reasons=result.llm_judge.reasons,
        )

    return EvalOut(
        episode_id=result.episode_id,
        title=result.title,
        rouge=rouge_out,
        bertscore=bert_out,
        llm_judge=judge_out,
        passed_thresholds=result.passed_thresholds,
        threshold_failures=result.threshold_failures,
        evaluated_at=result.evaluated_at,
    )


@app.get("/summaries", response_model=list[EpisodeSummaryOut], tags=["Summaries"])
def list_summaries(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List stored summaries, newest first. Supports pagination via limit/offset """
    paths = sorted(OUTPUTS_DIR.glob("*_summary.json"), reverse=True)
    page = paths[offset: offset + limit]

    results = []
    for path in page:
        with open(path) as f:
            s = json.load(f)
        results.append(EpisodeSummaryOut(
            episode_id=s["episode_id"],
            title=s["title"],
            podcast_name=s["podcast_name"],
            pub_date=s["pub_date"],
            final_summary=s["final_summary"],
            chunk_summaries=[ChunkSummaryOut(**c) for c in s["chunk_summaries"]],
            model=s["model"],
            provider=s["provider"],
            prompt_version=s["prompt_version"],
            total_input_tokens=s["total_input_tokens"],
            total_output_tokens=s["total_output_tokens"],
            total_latency_ms=s["total_latency_ms"],
            num_chunks=s["num_chunks"],
            summarized_at=s["summarized_at"],
        ))
    return results 