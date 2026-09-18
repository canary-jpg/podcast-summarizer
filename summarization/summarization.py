"""
Summarizes preprocessed podcast episode chunks using an LLM.
Strategy: summarize each chunk indenpendently, then combine chunk
summaries into a final episode-level summary (map-reduce pattern).

Supports:
    - Anthropic Claude (defualt)
    - OpenAI GPT models
    - Any OpenAI-compatible local endpoint (Ollama, LM Studio, etc.)
Experiment tracking via Weights & Biases (optional but recommended).

Usage:
    # summarize all preprocessed episodes
    python summarizer.py

    #single file, Claude, with W&B tracking
    python summarizer.py --file data/preprocessed/abc123_preprocessed.json --track

    #use GPT-4o instead
    python summarizer.py --provider openai --model gpt-4o

    #local model via Ollama
    python summarizer.py --provider local --model llama3 --base_url http://localhost:11434/v1

Requirements:
    pip install anthropic openai python-dotenv pyyaml tqdm wandb
"""

import argparse
import json 
import logging 
import os 
import time 
from dataclasses import asdict, dataclass, field 
from datetime import datetime, timezone 
from pathlib import Path 
from typing import Optional

import yaml 
from dotenv import load_dotenv
from tqdm import tqdm 

load_dotenv()

#--Logging---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("summarizer")

#--Config------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
OUTPUTS_DIR = Path("data/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS_CONFIG = Path("configs/prompts.yaml")
EVAL_CONFIG = Path("configs/eval_thresholds.yaml")

DEFAULT_PROMPTS = {
    "chunk_system": (
        "You are an expert podcast summarizer. "
        "Your job is to extract the key ideas, insights, and any concrete "
        "facts or recommendations from the transcript segment provided. "
        "Be concise but complete. Write in clear, flowing prose - not bullet points."
    ),
    "chunk_user": (
        "Summarize the following podcast transcript segment. "
        "Focus on the main ideas discussed. "
        "Ignore filler conversation, repetition, and off-topic tangents.\n\n"
        "Transcript segment:\n{chunk_text}"
    ),
    "combine_system": (
        "You are an expert editor producing a final summary of a podcast episode "
        "from a series of segment summries. Your output should read as a coherent, "
        "well-structured summary - not a list of segment recaps."
    ),
    "combine_user": (
        "Below are summaries of consecutive segments from the podcast episode "
        '"{title}" by {podcast_name}.\n\n'
        "Combine these into a single, cohesive episode summary. "
        "Strucutre it as:\n"
        "1. A 2-3 sentence overview of the episode\n"
        "2. Key topics and insights discussed\n"
        "3. Any notable quotes, recommendations, or actionable takeaways\n\n"
        "Segment summaries:\n{segment_summaries}"
    ),
}


#--Data models-------------------------------------------------------------------------

@dataclass 
class ChunkSummary:
    chunk_index: int 
    summary: str
    input_tokens: int 
    output_tokens: int 
    latency_ms: float 
    model: str 

@dataclass 
class EpisodeSummary:
    episode_id: str 
    title: str 
    podcast_name: str
    pub_date: str 
    final_summary: str 
    chunk_summaries: list[dict]
    model: str 
    provider: str 
    prompt_version: str 
    total_input_tokens: int = 0 
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0 
    num_chunks: int = 0
    summarized_at: str = ""

    def to_json(self) -> dict:
        return asdict(self)

#--Prompt management---------------------------------------------------------------

def load_prompt(version: str = "v1") -> tuple[dict, str]:
    """
    Load prompt templates from configs/prompts.yaml if it exists,
    otherwise fall back to the hardcoded defaults above.
    Returns (prompt_dict, resolved_version).
    """
    if PROMPTS_CONFIG.exists():
        with open(PROMPTS_CONFIG) as f:
            all_prompts = yaml.safe_load(f)
        prompts = all_prompts.get(version, all_prompts.get("v1", DEFAULT_PROMPTS))
        logger.info("Loaded prompts from %s (version: %s)", PROMPTS_CONFIG, version)
        return prompts, version 
    
    logger.info("No prompts.yaml found - using built-in defaults (version: %s)", version)
    return DEFAULT_PROMPTS, version 


#--LLM clients------------------------------------------------------------------------

class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model 

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        t0 = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        latency_ms = (time.monotonic() - t0) * 1000
        return {
            "text": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_ms": latency_ms,
        }

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o", base_url: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("OPEN_API_KEY", "ollama"), #ollama for local
            base_url=base_url,
        )
        self.model = model 

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        t0 = time.monotonic()
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.monotonic() - t0) * 1000
        usage = response.usage 
        return {
            "text": response.choices[0].message.content,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "latency_ms": latency_ms,
        }

def build_client(provider: str, model: str, base_url: Optional[str] = None):
    """Factory - returns a client with a unified .complete() interface. """
    if provider == "anthropic":
        return AnthropicClient(model=model)
    elif provider in ("openai", "local"):
        return OpenAIClient(model=model, base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose anthropic, openai, or local.")

#--Experiement tracking-------------------------------------------------------------------------------

class Tracker:
    """
    Thin wrapper around W&B. Initializes a run per episode.
    All methods are no-opes if W&B isn't installed or tracking is disabled.
    """

    def __init__(self, enabled: bool, project: str = "podcast-summarizer"):
        self.enabled = enabled
        self.run = None
        if enabled:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed - tracking disabled. pip install wandb")
                self.enabled = False 

    def start_run(self, episode_id: str, config: dict):
        if not self.enabled:
            return 
        self.run = self.wandb.init(
            project="podcast-summarizer",
            name=episode_id,
            config=config,
            reinit=True,
        )

    def log(self, metrics: dict):
        if self.enabled and self.run:
            self.run.log(metrics)

    def log_summary(self, summary: EpisodeSummary):
        if not self.enabled or not self.run:
            return 
        self.run.log({
            "total_input_tokens": summary.total_input_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_latency_ms": summary.total_latency_ms,
            "num_chunks": summary.num_chunks,
            "final_summary_len": len(summary.final_summary),
        })
        #log the final summary as a W&B artifact for easy review
        artifact = self.wandb.Artifact(
            name=f"summary-{summary.episode_id}",
            type="summary",
        )
        with artifact.new_file("summary.json") as f:
            json.dump(summary.to_json(), f, indent=2)
        self.run.log_artifact(artifact)

    def finish(self):
        if self.enabled and self.run:
            self.run.finish()


##--Core summarization logic------------------------------------------------------

def summarize_chunk(
    chunk: dict,
    client,
    prompts: dict,
    tracker: Tracker,
) -> ChunkSummary:
    """Summarize a single preprocessed chunk """
    system = prompts["chunk_system"]
    user = prompts["chunk_user"].format(chunk_text=chunk["text"])

    result = client.complete(system=system, user=user, max_tokens=512)

    tracker.log({
        f"chunk_{chunk['chunk_index']}_input_tokens": result["input_tokens"],
        f"chunk_{chunk['chunk_index']}_output_tokens": result["output_tokens"],
        f"chunk_{chunk['chunk_index']}_latency_ms": result["latency_ms"],
    })

    return ChunkSummary(
        chunk_index=chunk["chunk_index"],
        summary=result["text"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        latency_ms=result["latency_ms"],
        model=client.model,
    )

def combine_summaries(
    chunk_summaries: list[ChunkSummary],
    episode: dict,
    client,
    prompts: dict,
) -> dict:
    """
    Map-reduce comine step: merge all chunk summaries into one
    coherent episode-level summary.
    """
    #if there's only one chunk, skip the combine call: the chunk summary IS the final summary
    if len(chunk_summaries) == 1:
        logger.info("Single-chunk episide: skipping combine step")
        return {
            "text": chunk_summaries[0].summary,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0
        }
    
    segment_summaries = "\n\n---\n\n".join(
        f"[Segment {cs.chunk_index + 1}]\n{cs.summary}"
        for cs in chunk_summaries
    )

    system = prompts["combine_system"]
    user = prompts["combine_user"].format(
        title=episode.get("title", ""),
        podcast_name=episode.get("podcast-name", ""),
        segment_summaries=segment_summaries,
    )

    return client.complete(system=system, user=user, max_tokens=1024)


def summarize_episode(
    episode: dict,
    client,
    prompts: dict,
    prompt_version: str,
    tracker: Tracker,
    provider: str,
) -> EpisodeSummary:
    """
    Full map-reduce pipeline for one episode:
    Map: summarize each chunk independently
    Reduce: combine chunk summaries into final episode summary
    """
    chunks = episode.get("chunks", [])
    if not chunks:
        raise ValueError(f"Episode: '{episode.get('title')}' has no chunks to summarize")
    
    logger.info(
        "Summarizing '%s' (%d chunks)",
        episode.get("title", ""),
        len(chunks),
    )

    #--Map phase------------------------------------------------------------------------------------
    chunk_summaries: list[ChunkSummary] = []
    for chunk in tqdm(chunks, desc=" Chunks", leave=False):
        cs = summarize_chunk(chunk, client, prompts, tracker)
        chunk_summaries.append(cs)
        logger.debug(
            " Chunk %d: %d->%d tokens, %.0fms",
            cs.chunk_index, cs.input_tokens, cs.output_tokens, cs.latency_ms,
        )
        #rate limiting buffer between chunk calls
        time.sleep(0.3)

    #--Reduce phase----------------------------------------------------------------------------------
    logger.info(" Combining %d chunk summaries...", len(chunk_summaries))
    combine_result = combine_summaries(chunk_summaries, episode, client, prompts)

    #--Aggregate stats-------------------------------------------------------------------------------
    total_input = sum(cs.input_tokens for cs in chunk_summaries) + combine_result["input_tokens"]
    total_output = sum(cs.output_tokens for cs in chunk_summaries) + combine_result["output_tokens"]
    total_latency = sum(cs.latency_ms for cs in chunk_summaries) + combine_result["latency_ms"]

    summary = EpisodeSummary(
        episode_id=episode.get("id", ""),
        title=episode.get("title", ""),
        podcast_name=episode.get("podcast_name", ""),
        pub_date=episode.get("pub_date", ""),
        final_summary=combine_result["text"],
        chunk_summaries=[asdict(cs) for cs in chunk_summaries],
        model=client.model,
        provider=provider,
        prompt_version=prompt_version,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_latency_ms=total_latency,
        num_chunks=len(chunks),
        summarized_at=datetime.now(tz=timezone.utc).isoformat(),
    )

    tracker.log_summary(summary)
    return summary

#--File I/O----------------------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_dir: Path,
    client,
    prompts: dict,
    prompt_version: str,
    tracker: Tracker,
    provider: str,
    skip_existing: bool
) -> Optional[Path]:
    out_path = output_dir / input_path.name.replace("_preprocessed.json", "_summary.json")

    if skip_existing and out_path.exists():
        logger.info("Skipping already summarized: %s", input_path.name)

    with open(input_path) as f:
        episode = json.load(f)

    tracker.start_run(
        episode_id=episode.get("id", input_path.stem),
        config={
            "model": client.model,
            "provider": provider,
            "prompt_version": prompt_version,
            "num_chunks": len(episode.get("chunks", [])),
        },
    )

    try:
        summary = summarize_episode(
            episode=episode,
            client=client,
            prompts=prompts,
            prompt_version=prompt_version,
            tracker=tracker,
            provider=provider,
        )
    except Exception as e:
        logger.error("Failed to summarize %s: %s", input_path.name, e)
        tracker.finish()
        return None 

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary.to_json(), f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved summary: %s | tokens in/out: %d/%d | latency: %.1fs",
        out_path.name,
        summary.total_input_tokens,
        summary.total_output_tokens,
        summary.total_latency_ms / 1000,
    )

    tracker.finish()
    return out_path 


#--CLI--------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize preprocessed podcast episode using an LLM"
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        help="Summarize a single preprocessed episode JSON"
    )
    parser.add_argument(
        "--input_dir", type=Path, default=PROCESSED_DIR,
        help="Output directory for summaries (default: data/outputs/)"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUTS_DIR,
        help="Output directory for summaries (default: data/outputs/)"
    )
    parser.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "openai", "local"],
        help="LLM provider (default: anthropic)"
    )
    parser.add_argument(
        "--model", default=None,
        help=(
            "Model name. Defaults: anthropic=claude-sonnet-4-6, "
            "openai=gpt-4o, local=llama3"
        ),
    )
    parser.add_argument(
        "--base_url", default=None,
        help="Base URL for local/OpenAI-compatibl endpoint (e.g. https://localhost:11434/v1)"
    )
    parser.add_argument(
        "--prompt_version", default="v1",
        help="Prompt version to load from configs/prompts.yaml (default: v1)"
    )
    parser.add_argument(
        "--track", action="store_true",
        help="Enable Weights & Biases experiment tracking"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-summarize even if output already exists"
    )
    return parser.parse_args()

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "local": "llama3"
}

def main():
    args = parse_args()

    model = args.model or DEFAULT_MODELS[args.provider]
    client = build_client(args.provider, model, args.base_url)
    prompts, prompt_version = load_prompt(args.prompt_version)
    tracker = Tracker(enabled=args.track)

    logger.info("Provider: %s | Model: %s | Prompts: %s", args.provider, model, prompt_version)

    if args.file:
        files = [args.file]
    else:
        files = sorted(args.input_dir.glob("*_preprocessed.json"))

    if not files:
        logger.warning("No preprocessed episodes files found in %s", args.input_dir)
        return 
    
    logger.info("Episode to summarize: %d", len(files))

    saved = []
    for path in tqdm(files, desc="Summarizing episodes"):
        out = process_file(
            input_path=path,
            output_dir=args.output_dir,
            client=client,
            prompts=prompts,
            prompt_version=prompt_version,
            tracker=tracker,
            provider=args.provider,
            skip_existing=not args.no_skip,
        )
        if out:
            saved.append(out)

    logger.info("Done. %d episode(s) summarized.", len(saved))
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()