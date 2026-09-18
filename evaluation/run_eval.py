"""
Evaluates genderated podcast summaries using three complementary methods:
1. ROUGE        -lexical overlap with a reference summary (if available)
2. BERTScore    - semantic similarity using contextual embedding
3. LLM-as-judge - rubic-based quality scoring via an LLM (no reference needed)

The LLM-as-judge scorer is the most useful in practice because:
    - You rarely have human-written reference summaries for podcasts
    - It catches fluency and coherence issues ROUGE can't see
    - It's cheap relative to the summarization cost itself

Outputs:
    - data/eval_results/<episode_id>_eval.json (per-episode scores)
    - data/eval_results/eval_summary.csv (aggregate across all episodes)

Usage:
    # evaluate all summaries in data/outputs/
    python run_eval.py

    #single file
    python run_eval.py --file data/outputs/abc123_summary.json

    #skip LLM judge (faster, cheaper)
    python run_eval.py --no_llm_judge

    #use a reference summary for ROUGE/BERTScore
    python run_eval.py --reference_dir data/references/

    #check against thresholds and exit non-zero if failing (for CI)
    python run_eval.py --ci_mode

Requirements:
    pip install rouge-score bert-score anthropic python-dotenv pyyaml pandas tqdm
"""

import argparse
import csv 
import json 
import logging 
import os 
import sys 
import time 
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone 
from pathlib import Path 
from typing import Optional 

import yaml 
from dotenv import load_dotenv
from tqdm import tqdm 

load_dotenv()

#--Logging--------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eval")

#--Config----------------------------------------------------------------------------

OUTPUTS_DIR = Path("data/outputs")
EVAL_DIR = Path("data/eval_results")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS_CONFIG = Path("configs/eval_thresholds.yaml")

#Default thresholds - override in configs/eval_thresholds.yaml
DEFAULT_THRESHOLDS = {
    "rouge1_f": 0.20, #lenient: reference summaries for podcasts are rare
    "rouge2_f": 0.05,
    "rougeL_f": 0.15,
    "bertscore_f1": 0.82, #BERTScore is high by default; 0.82 is a reasonable floor
    "llm_overall": 3.0,  #out of 5
}

#LLM judge rubric: scored 1-5 on each dimension
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of automatically generated podcast summaries.
You will be given a podcast episode's title, description, and a generated summary.
Score the summary on each of these FIVE dimensions from 1 (poor) to 5 (excellent):
  - faithfulness
  - coverage
  - fluency
  - conciseness
  - overall

You MUST include all five keys in your response. If you omit any key the evaluation is invalid.
Respond ONLY with valid JSON — no preamble, no explanation outside the JSON.
"""

JUDGE_USER_TEMPLATE = """Evaluate the following podcast summary.

Episode title: {title}
Podcast: {podcast_name}
Episode description: {description}

Generated summary:
{summary}

Score each dimension 1-5 and provide a brief reason (one sentence each).
Return ONLY this JSON strucutre:
{{
    "faithfulness": {{
        "score": <int 1-5>,
        "reason": "<one sentence>"
    }},
    "coverage": {{
        "score": <int 1-5>,
        "reason": "<one sentence>"
    }},
    "conciseness": {{
        "score": <int 1-5>,
        "reason:" "<one sentence>"
    }},
    "overall": {{
        "score": <int 1-5>,
        "reason": "<one sentence>"
    }}
}}

Scoring guide:
    faithfulness - Does the summary accurately reflect what was discussed? No hallucinations?
    coverage     - Does it capture the main topics and key insights?
    fluency      - It is well-written, coherent, and easy to read?
    conciseness  - Does it avoid unnecessary reptition and padding?
    overall      - Holistic quality as a podcast summary a listener would actuall find useful.
"""

#--Data Models---------------------------------------------------------------------------------------

@dataclass
class RougeScores:
    rouge1_precision: float = 0.0
    rouge1_recall: float = 0.0
    rouge1_f: float = 0.0 
    rouge2_precision: float = 0.0
    rouge2_recall: float = 0.0 
    rouge2_f: float = 0.0 
    rougeL_precision: float = 0.0 
    rougeL_recall: float = 0.0 
    rougeL_f: float = 0.0 

@dataclass 
class BertScores:
    precision: float = 0.0 
    recall: float = 0.0 
    f1: float = 0.0 
    model_used: str = ""

@dataclass 
class LLMJudgeScores:
    faithfulness: float = 0.0
    coverage: float = 0.0 
    fluency: float = 0.0 
    conciseness: float = 0.0 
    overall: float = 0.0 
    reasons: dict = field(default_factory=dict)
    raw_response: str = ""
    error: Optional[str] = None 

@dataclass 
class EvalResult:
    episode_id: str 
    title: str 
    podcast_name: str 
    model: str 
    prompt_version: str 
    summary_length_chars: int 
    summary_word_count: int 
    num_chunks: int 
    rouge: Optional[RougeScores] = None 
    bertscore: Optional[BertScores] = None 
    llm_judge: Optional[LLMJudgeScores] = None 
    has_reference: bool = False 
    evaluated_at: str = ""
    passed_thresholds: Optional[bool] = None 
    threshold_failures: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Flatten for CSV export """
        flat = {
            "episode_id": self.episode_id,
            "title": self.title,
            "podcast_name": self.podcast_name,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "summary_length_chars": self.summary_length_chars,
            "summary_word_count": self.summary_word_count,
            "num_chunks": self.num_chunks,
            "has_reference": self.has_reference,
            "passed_thresholds": self.passed_thresholds,
            "evaluated_at": self.evaluated_at,
        }
        if self.rouge:
            flat.update({
                "rouge1_f": round(self.rouge.rouge1_f, 4),
                "rouge2_f": round(self.rouge.rouge2_f, 4),
                "rougeL_f": round(self.rouge.rougeL_f, 4),
            })
        if self.bertscore:
            flat["bertscore_f1"] = round(self.bertscore.f1, 4)
        if self.llm_judge:
            flat.update({
                "judge_faithfulness": self.llm_judge.faithfulness,
                "judge_coverage": self.llm_judge.coverage,
                "judge_fluency": self.llm_judge.fluency,
                "judge_conciseness": self.llm_judge.conciseness,
                "judge_overall": self.llm_judge.overall,
            })
        return flat 

#--Scorers------------------------------------------------------------------

def compute_rouge(hypothesis: str, reference: str) -> RougeScores:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L between generated and reference summary.
    Requires a human-written reference - use LLM judge when unavailable
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)
    return RougeScores(
        rouge1_precision=scores["rouge1"].precision,
        rouge1_recall=scores["rouge1"].recall,
        rouge1_f=scores["rouge1"].fmeasure,
        rouge2_precision=scores["rouge2"].precision,
        rouge2_recall=scores["rouge2"].recall,
        rouge2_f=scores["rouge2"].fmeasure,
        rougeL_precision=scores["rougeL"].precision,
        rougeL_recall=scores["rougeL"].recall,
        rougeL_f=scores["rougeL"].fmeasure,
    )

def compute_bertscore(hypothesis: str, reference: str) -> BertScores:
    """
    Compute BERTScore F1 between generated and referenced summary.
    Uses the 'distilbert-base-uncased' model by default for speed.
    Switch to 'roberta-large' for higher accuracy at the cost of more RAM
    """
    from bert_score import score as bert_score 
    P, R, F1 = bert_score(
        [hypothesis],
        [reference],
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    return BertScores(
        precision=float(P[0]),
        recall=float(R[0]),
        f1=float(F1[0]),
        model_used="distilbert-base-uncased",
    )

def compute_llm_judge(
    summary: dict,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
) -> LLMJudgeScores:
    """
    Uses an LLM to score the summary on a 1-5 rubric.
    Uses claude-haiku by default: fast and cheap for evaluation.

    Important: use a DIFFERENT model from the one that generated the summary
    where possible, to reduce self-serving bias
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = JUDGE_USER_TEMPLATE.format(
        title=summary.get("title", ""),
        podcast_name=summary.get("podcast_name", ""),
        description=summary.get("description", "")[:500],  #truncate long descriptions
        summary=summary.get("final_summary", ""),
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()

        #strip markdown code fences if the model added them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)

        return LLMJudgeScores(
            faithfulness=float(parsed.get("faithfulness", {}).get("score", 0)),
            coverage=float(parsed.get("coverage", {}).get("score", 0)),
            fluency=float(parsed.get("fluency", {}).get("score", 0)),
            conciseness=float(parsed.get("conciseness", {}).get("score", 0)),
            overall=float(parsed.get("overall", {}).get("score", 0)),
            reasons={k: v["reason"] for k, v in parsed.items()},
            raw_response=raw,
        )
    except json.JSONDecodeError as e:
        logger.warning("LLM judge returned invalid JSON: %s", e)
        return LLMJudgeScores(error=f"JSON parse error: {e}", raw_response=raw)
    except Exception as e:
        logger.warning("LLM judge call failed: %s", e)
        return LLMJudgeScores(error=str(e))

#--Threshold checking-----------------------------------------------------------------

def load_threshold() -> dict:
    if THRESHOLDS_CONFIG.exists():
        with open(THRESHOLDS_CONFIG) as f:
            loaded = yaml.safe_load(f)
        logger.info("Loaded thresholds from %s", THRESHOLDS_CONFIG)
        return {**DEFAULT_THRESHOLDS, **loaded}
    return DEFAULT_THRESHOLDS

def check_thresholds(result: EvalResult, thresholds: dict) -> tuple[bool, list[str]]:
    """
    Check all available scores against configured thresholds.
    Returns (passed: bool, failures: list[str])
    """
    failures = []

    if result.rouge and result.has_reference:
        for metric in ["rouge1_f", "rouge2_f", "rougeL_f"]:
            val = getattr(result.rouge, metric)
            threshold = thresholds.get(metric)
            if threshold and val < threshold:
                failures.append(
                    f"{metric}={val:.3f} < threshold {threshold}"
                )

    if result.bertscore and result.has_reference:
        val = result.bertscore.f1
        threshold = thresholds.get("bertscore_f1")
        if threshold and val < threshold:
            failures.append(
                f"bertscore_f1={val:.3f} < threshold {threshold}"
            )
    
    if result.llm_judge and not result.llm_judge.error:
        val = result.llm_judge.overall 
        threshold = thresholds.get("llm_overall")
        if threshold and val < threshold:
            failures.append(
                f"llm_overall={val:.1f} < threshold {threshold}"
            )
    
    return len(failures) == 0, failures 


#--Main eval pipeline---------------------------------------------------------------------

def evaluate_summary(
    summary: dict,
    reference_text: Optional[str],
    use_llm_judge: bool,
    thresholds: dict,
) -> EvalResult:
    episode_id = summary.get("episode_id", "unknown")
    final_summary = summary.get("final_summary", "")

    result = EvalResult(
        episode_id=episode_id,
        title=summary.get("title", ""),
        podcast_name=summary.get("podcast_name", ""),
        model=summary.get("model", ""),
        prompt_version=summary.get("prompt_version", ""),
        summary_length_chars=len(final_summary),
        summary_word_count=len(final_summary.split()),
        num_chunks=summary.get("num_chunks", 0),
        has_reference=reference_text is not None,
        evaluated_at=datetime.now(tz=timezone.utc).isoformat(),
    )

    #ROUGE
    if reference_text:
        logger.info(" Computing ROUGE...")
        try:
            result.rouge = compute_rouge(final_summary, reference_text)
            logger.info(
                "  ROUGE-1 F=%.3f |  ROUGE-2 F=%.3f | ROUGE-L F=%.3f",
                result.rouge.rouge1_f,
                result.rouge.rouge2_f,
                result.rouge.rougeL_f,
            )
        except ImportError:
            logger.warning(" rouge-score not installed - skipping ROUGE. pip install rouge-score")

    #BERTScore
    if reference_text:
        logger.info(" Computing BERTScore...")
        try:
            result.bertscore = compute_bertscore(final_summary, reference_text)
            logger.info("   BERTScore F1=%.3f", result.bertscore.f1)
        except ImportError:
            logger.warning("    bert-score not installed - skipping. pip install bert-score")

    #LLM judge
    if use_llm_judge:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("    ANTHROPIC_API_KEY not set - skipping LLM judge")
        else:
            logger.info(" Running LLM judge...")
            result.llm_judge = compute_llm_judge(summary, api_key)
            if result.llm_judge.error:
                logger.warning("  LLM judge error: %s", result.llm_judge.error)
            else:
                logger.info(
                    "  Judge score - faithfulness=%.1f coverage=%.1f "
                    "fluency=%.1f conciseness=%.1f overall=%.1f",
                    result.llm_judge.faithfulness,
                    result.llm_judge.coverage,
                    result.llm_judge.fluency,
                    result.llm_judge.conciseness,
                    result.llm_judge.overall,
                )
            time.sleep(0.3) #rate limit buffer
    
    #threshold check
    passed, failures = check_thresholds(result, thresholds)
    result.passed_thresholds = passed 
    result.threshold_failures = failures

    if failures:
        logger.warning("  THRESHOLD FAILURES: %s", " | ".join(failures))
    else:
        logger.info(" All thresholds passed")
    
    return result 

def process_file(
    summary_path: Path,
    reference_dir: Optional[Path],
    use_llm_judge: bool,
    thresholds: dict,
    skip_existing: bool,
) -> Optional[EvalResult]:
    out_path = EVAL_DIR / summary_path.name.replace("_summary.json", "_eval.json")

    if skip_existing and out_path.exists():
        logger.info("Skipping already-evaluated: %s", summary_path.name)
        with open(out_path) as f:
            data = json.load(f)

        #reconstruct a minimal EvalResult for CSV aggregation
        result = EvalResult(
            episode_id=data.get("episode_id", ""),
            title=data.get("title", ""),
            podcast_name=data.get("podcast_name", ""),
            model=data.get("model", ""),
            prompt_version=data.get("prompt_version", ""),
            summary_length_chars=data.get("summary_length_chars", 0),
            summary_word_count=data.get("summary_word_count", 0),
            num_chunks=data.get("num_chunks", 0),
        )
        return result 
    
    with open(summary_path) as f:
        summary = json.load(f)

    #look for a reference summary: data/references/<episode_id>.txt
    reference_text = None 
    if reference_dir:
        episode_id = summary.get("episode_id", "")
        ref_path = reference_dir / f"{episode_id}.txt"
        if ref_path.exists():
            reference_text = ref_path.read_text().strip()
            logger.info("Found reference for %s", episode_id)

    logger.info("Evaluating: %s", summary.get("title", summary_path.name))
    result = evaluate_summary(summary, reference_text, use_llm_judge, thresholds)

    with open(out_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Saved eval: %s", out_path)

    return result 

def write_csv_summary(results: list[EvalResult], output_path: Path):
    """Write a flat CSV aggregating all eval results - easy to open in a notebook """
    rows = [r.to_dict() for r in results if r]
    if not rows:
        return 
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote CSV summary: %s", output_path)


#--CLI---------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate podcast summaries with ROUGE, BERTScore, and LLM-as-judge"
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        help="Evaluate a single summary JSON"
    )
    parser.add_argument(
        "--input_dir", type=Path, default=OUTPUTS_DIR,
        help="Directly of summary JSONs (default: data/outputs/)"
    )
    parser.add_argument(
        "--reference_dir", type=Path, default=None,
        help=(
            "Directory of reference summary .txt files named <episode_id>.txt"
            "Required for ROUGE and BERTScore"
        ),
    )
    parser.add_argument(
        "--no_llm_judge", action="store_true",
        help="Skip the LLM-as-judge scorer (faster, no API cost)"
    )
    parser.add_argument(
        "--ci_mode", action="store_true",
        help=(
            "Exit with code 1 if any episode fails thresholds. "
            "Use this in GitHub Actions eval_gate.yml"
        ),
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-evaluate even if eval output already exists"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    thresholds = load_threshold()

    logger.info("Thresholds: %s", thresholds)

    if args.file:
        files = [args.file]
    else:
        files = sorted(args.input_dir.glob("*_summary.json"))

    if not files: 
        logger.warning("No summary files found in %s", args.input_dir)
        return


    logger.info("Evaluating %d summary file(s)", len(files))

    results: list[EvalResult] = []
    for path in tqdm(files, desc="Evaluating"):
        result = process_file(
            summary_path=path,
            reference_dir=args.reference_dir,
            use_llm_judge=not args.no_llm_judge,
            thresholds=thresholds,
            skip_existing=not args.no_skip,
        )
        if result:
            results.append(result)
    
    #Aggregate CSV
    csv_path = EVAL_DIR / "eval_summary.csv"
    write_csv_summary(results, csv_path)

    #print summary table
    print("\n--Eval Results -----------------------------------")
    for r in results:
        judge_str = ""
        if r.llm_judge and not r.llm_judge.error:
            judge_str = f"  judge={r.llm_judge.overall:.1f}/5"
        rouge_str = ""
        if r.rouge:
            rouge_str = f"  R1={r.rouge.rouge1_f:3f} RL={r.rougeL_f:3f}"
        status = "PASS" if r.passed_thresholds else "FAIL"
        print(f"[{status}] {r.title[:60]:<60}{rouge_str}{judge_str}")

    any_failures = any(
        r.passed_thresholds is False for r in results if r.passed_thresholds is not None
    )

    if args.ci_mode and any_failures:
        logger.error("One or more episodes failed eval thresholds. Blocking deploy")
    
    print(f"\nDetailed results: {EVAL_DIR}/")
    print(f"CSV summary:        {csv_path}")

if __name__ == "__main__":
    main()