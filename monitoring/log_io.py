"""
Logs summarizer inputs and outputs to a local SQLite database for 
monitoring and drift detection. Called automatically by the pipeline 
after each summarization run.

Tracks:
    - Every summary produced (model, prompt version, token counts, latency)
    - Per-episode eval scores over time
    - Summary length and quality trends

This is the data store that drift_report.py reads from.

Usage:
    #log a summary file
    python log_io.py --summary data/outputs/abc123_summary.json

    #log a summary & its eval result together
    python log_io.py --summary data/outputs/abc123_summary.json \
                    -- eval data/eval_results/abc_123_eval.json
    
    #log all summaries and evals in their default directories
    python log_io.py --all

Requirements:
    pip install python-dotenv
    (SQLite is in Python's stdlib - no extra install needed)
"""

import argparse 
import json 
import logging 
import sqlite3
from datetime import datetime, timezone 
from pathlib import Path 
from typing import Optional 

#--Logging---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("log_io")

#--Config----------------------------------------------------------

OUTPUT_DIR = Path("data/outputs")
EVAL_DIR = Path("data/eval_results")
DB_PATH = Path("data/monitoring.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

#--Schema----------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id              TEXT NOT NULL,
    title                   TEXT,
    podcast_name            TEXT,
    model                   TEXT,
    provider                TEXT,
    prompt_version          TEXT,
    num_chunks              INTEGER,
    total_input_tokens      INTEGER,
    total_output_tokens     INTEGER,
    total_latency_ms        REAL,
    summary_length_chars    INTEGER,
    summary_word_count      INTEGER,
    summarized_at           TEXT,
    logged_at               TEXT,
    UNIQUE(episode_id, prompt_version, model)
);

CREATE TABLE IF NOT EXISTS eval_scores (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id             TEXT NOT NULL,
    model                  TEXT,
    prompt_version         TEXT,
    rouge1_f               REAL,
    rouge2_f               REAL,
    rougeL_f               REAL,
    bertscore_f1           REAL,
    judge_faithfulness     REAL,
    judge_coverage         REAL,
    judge_fluency          REAL,
    judge_conciseness      REAL,
    judge_overall          REAL,
    passed_thresholds      INTEGER,
    has_reference          INTEGER,
    evaluated_at           TEXT,
    logged_at              TEXT,
    UNIQUE(episode_id, prompt_version, model)
);

CREATE INDEX IF NOT EXISTS idx_summaries_model      ON summaries(model);
CREATE INDEX IF NOT EXISTS idx_summaries_prompt     ON summaries(prompt_version);
CREATE INDEX IF NOT EXISTS idx_summaries_at         ON summaries(summarized_at);
CREATE INDEX IF NOT EXISTS idx_eval_model           ON eval_scores(model);
CREATE INDEX IF NOT EXISTS idx_eval_prompt          ON eval_scores(prompt_version);
CREATE INDEX IF NOT EXISTS idx_eval_at              ON eval_scores(evaluated_at);
"""

#--DB connection--------------------------------------------------------------------------

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  #dict-style access
    conn.executescript(SCHEMA)
    conn.commit()
    return conn 

#--Logging functions----------------------------------------------------------------------

def log_summary(summary: dict, conn: sqlite3.Connection) -> bool:
    """
    Insert or replace a summary record.
    Uses (episode_id, prompt_version, model) as the unique key so
    re-running with a new prompt version creates a new row rather
    than overwriting - preserving history for trend analysis.
    """
    now = datetime.now(tz=timezone.utc).isoformat()
    final_summary = summary.get("final_summary", "")

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO summaries (
                episode_id, title, podcast_name, model, provider,
                prompt_version, num_chunks, total_input_tokens,
                total_output_tokens, total_latency_ms,
                summary_length_chars, summary_word_count, 
                summarized_at, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.get("episode_id"),
                summary.get("title"),
                summary.get("podcast_name"),
                summary.get("model"),
                summary.get("provider"),
                summary.get("prompt_version"),
                summary.get("num_chunks"),
                summary.get("total_input_tokens"),
                summary.get("total_output_tokens"),
                summary.get("total_latency_ms"),
                len(final_summary),
                len(final_summary.split()),
                summary.get("summarized_at"),
                now,
            ),
        )
        conn.commit()
        logger.info("Logged summary: %s", summary.get("episode_id"))
        return True 
    except Exception as e:
        logger.error("Failed to log summary: %s", e)
        return False


def log_eval(eval_result: dict, conn: sqlite3.Connection) -> bool:
    """Insert or replace an eval score record """
    now = datetime.now(tz=timezone.utc).isoformat()
    rouge = eval_result.get("rouge") or {}
    bertscore = eval_result.get("bertscore") or {}
    judge = eval_result.get("llm_judge") or {}

    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_scores (
                episode_id, model, prompt_version,
                rouge1_f, rouge2_f, rougeL_f, bertscore_f1,
                judge_faithfulness, judge_coverage, judge_fluency,
                judge_conciseness, judge_overall,
                passed_thresholds, has_reference,
                evaluated_at, logged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eval_result.get("episode_id"),
                eval_result.get("model"),
                eval_result.get("prompt_version"),
                rouge.get("rouge1_f"),
                rouge.get("rouge2_f"),
                rouge.get("rougeL_f"),
                bertscore.get("f1"),
                judge.get("faithfulness"),
                judge.get("coverage"),
                judge.get("fluency"),
                judge.get("conciseness"),
                judge.get("overall"),
                int(eval_result.get("passed_thresholds", True)),
                int(eval_result.get("has_reference", False)),
                eval_result.get("evaluated_at"),
                now,
            ),
        )
        conn.commit()
        logger.info("Logged eval: %s", eval_result.get("episode_id"))
        return True
    except Exception as e:
        logger.error("Failed to log eval: %s", e)
        return False


#--Query helpers----------------------------------------------------------------

def show_recent(conn: sqlite3.Connection, limit: int = 20):
    """Print recent summary & eval records as a readable table """
    rows = conn.execute(
        """
        SELECT
            s.episode_id,
            s.title,
            s.model,
            s.prompt_version,
            s.num_chunks,
            s.total_input_tokens,
            s.total_output_tokens,
            ROUND(s.total_latency_ms / 1000, 1) as latency_s,
            e.judge_overall,
            e.passed_thresholds,
            s.summarized_at
        FROM summaries s
        LEFT JOIN eval_scores e
            ON s.episode_id = e.episode_id
            AND s.model = e.model
            AND s.prompt_version = e.prompt_version
        ORDER BY s.logged_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    if not rows:
        print("No records found")
        return 
    
    header = (
        f"{'Episode ID:<20'} {'Model:<30'} {'Prompt':<6} "
        f"{'Chunks':>6} {'In tok':>7} {'Out tok':>7}"
        f"{'Lat(s)':>6} {'Judge':>5} {'Pass>4'}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        judge = f"r['judge_overall']:.1f" if r["judge_overall"] else "   -  "
        passed = "✓" if r["passed_thresholds"] else "✗"
        print(
            f"{(r['episode_id']):<20} {(r['model'] or ''):<30} {(r['prompt_version'] or ''):<6} " 
            f"{(r['num_chunks'] or 0):>6} {(r['total_input_tokens'] or 0):>7} "
            f"{(r['total_output_tokens'] or 0):>7} "
            f"{(r['latency_s'] or 0):>6} {judge:>5} {passed:>4}"
        )
    print()


def get_stats(conn: sqlite3.Connection) -> dict:
    """Return aggregate statistics useful for the drift report """
    stats = {}

    #summary stats
    row = conn.execute(
        """
        SELECT
            COUNT(*)                            AS total_summaries,
            AVG(total_input_tokens)             AS avg_input_tokens,
            AVG(total_output_tokens)            AS avg_output_tokens,
            AVG(total_latency_ms / 1000.0)      AS avg_latency_s,
            AVG(summary_word_count)             AS avg_word_count
        FROM summaries      
        """
    ).fetchone()
    stats["summaries"] = dict(row)

    #eval stats
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                                AS total_evals,
            AVG(judge_overall)                                      AS avg_judge_overall,
            AVG(judge_faithfulness)                                 AS avg_faitfulness,
            AVG(judge_coverage)                                     AS avg_coverage,
            AVG(judge_fluency)                                      AS avg_fluency,
            AVG(judge_conciseness)                                  AS avg_conciseness,
            SUM(CASE WHEN passed_thresholds = 0 THEN 1 ELSE 0 END)  AS FAILURES
        FROM eval_scores
        """
    ).fetchone()
    stats["evals"] = dict(row)

    #per prompt version breakdown
    rows = conn.execute(
        """
        SELECT
            prompt_version,
            model,
            COUNT(*)                AS n,
            AVG(judge_overall)      AS avg_overall,
            AVG(judge_coverage)     AS avg_coverage,
            AVG(judge_fluency)      AS avg_fluency,
        FROM eval_scores
        GROUP BY prompt_version, model
        ORDER BY prompt_version, model
        """
    ).fetchall()
    stats["by_prompt_version"] = [dict(r) for r in rows]

    return stats 


#--CLI------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Log summarizer inputs/outputs to SQLite for monitoring"
    )
    parser.add_argument("--summary", type=Path, help="Path to a summary JSON file")
    parser.add_argument("--eval", type=Path, help="Path to a matching eval JSON file")
    parser.add_argument(
        "--all", action="store_true",
        help="Log all summaries and eval from their default directories"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Print recent log entries"
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Number of rows to show with --show (default: 20)"
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"Path to SQLite DB (default: {DB_PATH})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conn = get_connection(args.db)

    if args.show:
        show_recent(conn, limit=args.limit)
        return
    if args.all:
        summary_files = sorted(OUTPUT_DIR.glob("*_summary.json"))
        eval_files = {
            f.name.replace("_eval.json", ""): f
            for f in EVAL_DIR.glob("*_eval.json")
        }
        for sf in summary_files:
            with open(sf) as f:
                summary = json.load(f)
            log_summary(summary, conn)

            #match eval by stem
            stem = sf.name.replace("_summary.json", "")
            ef = eval_files.get(stem)
            if ef:
                with open(ef) as f:
                    eval_result = json.load(f)
                log_eval(eval_result, conn)
        logger.info(
            "Logged %d summaries, %d evals",
            len(summary_files), len(eval_files)
        )
        return
    
    if args.summary:
        with open(args.summary) as f:
            summary = json.load(f)
        log_summary(summary, conn)
    
    conn.close()


if __name__ == "__main__":
    main()