"""
Reads SQLite monitoring database populated by log_io.py and 
produces a drift report: detecting degradation in summary quality,
token usage, or latency over time.

Drift detection approach:
    - Splits logged runs into a "baseline" windown and a "recent" window
    - Compares means across key metrics using a simple statistical test
    - Flags metrics that have shifted beyond a configurable tolerance
    - Outputs a human-readable report & a JSON artifact for CI/dashboards

Usage:
    #compare last 7 days vs. prior baseline
    python drift_report.py

    #wider baseline windown, tighter alert threshold
    python drift_report.py --baseline_days 30 --recent_days 7 --tolerance 0.10

    #save report to file (useful for CI)
    python drift_report.py --output data/eval_results/drift_report.json

    #exit non-zero if drift detected (for CI use)
    python drift_report.py --ci_mode

Requirements:
    pip install scipy
"""

import argparse 
import json 
import logging
import sqlite3
import statistics 
import sys 
from dataclass import asdict, dataclass, field 
from datetime import datetime, timedelta, timezone 
from pathlib import Path 
from typing import Optional 

##--Logging----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s %(name)s: %(message)s]",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("drift_report")

#-- Config----------------------------------------------------------------------

DB_PATH = Path("data/monitoring.db")

#metrics to monitor - (table, column, direction)
#direction: "up" - higher is better (flag drops), "down" = lower is better (flag rises)
MONITORED_METRICS = [
    ("eval_scores", "judge_overall",        "up"),
    ("eval_scores", "judge_faithfulness",   "up"),
    ("eval_scores", "judge_coverage",       "up"),
    ("eval_scores", "judge_fluency",        "up"),
    ("eval_scores", "judge_conciseness",    "up"),
    ("summaries", "total_input_tokens",     "down"),
    ("summaries", "total_output_tokens",    "down"),
    ("summaries", "total_latency_ms",       "down"),
    ("summaries", "summary_word_count",     "up"),
]

#minimum samples required in each window to attempt drift detection
MIN_SAMPLES = 3

#--Data models-----------------------------------------------------------------

@dataclass 
class MetricWindow:
    metric: str 
    table: str
    n: int
    mean: float
    stdev: float 
    min_val: float 
    max_val: float 

@dataclass 
class DriftAlert:
    metric: str 
    direction: str 
    baseline_mean: float 
    recent_mean: float 
    delta: float #recent - baseline
    delta_pct: float #relative change
    is_degradation: bool #true if the change is in the bad direction
    p_value: Optional[float] = None 
    significant: bool = False 

@dataclass 
class DriftReport:
    generated_at: str 
    baseline_days: int 
    recent_days: int 
    tolerance: float 
    baseline_window: dict = field(default_factory=dict) #metric -> MetricWindow
    recent_window: dict = field(default_factory=dict)
    alerts: list[dict] = field(default_factory=list)
    drift_detected: bool = False 
    summary: str = ""

    def to_json(self) -> dict:  
        return asdict(self)

#--Data fetching--------------------------------------------------------------------

def fetch_window(
    conn: sqlite3.Connection,
    table: str,
    metric: str,
    since: datetime, 
    until: datetime,
) -> list[float]:
    """Fetch non-null metric values from a time window """
    time_col = "summarized_at" if table == "summaries" else "evaluated_at"
    rows = conn.execute(
        f"""
        SELECT {metric}
        FROM {table}
        WHERE {metric} IS NOT NULL
            AND {time_col} >= ?
            AND {time_col} < ?
        ORDER BY {time_col}
        """,
        (since.isoformat(), until.isoformat())
    ).fetchall()
    return [r[0] for r in rows]

def summarize_window(values: list[float], metric: str, table: str) -> MetricWindow:
    n = len(values)
    if n == 0:
        return MetricWindow(metric=metric, table=table, n=0, mean=0, stdev=0, min_val=0, max_val=0)
    return MetricWindow(
        metric=metric,
        table=table,
        n=n,
        mean=statistics.mean(values),
        stdev=statistics.stdev(values) if n > 1 else 0.0,
        min_val=min(values),
        max_val=max(values),
    )


#--Statistical test------------------------------------------------------------------------------------

def welch_t_test(a: list[float], b: list[float]) -> Optional[float]:
    """
    Welch's t-test p-value for two independent sampels of unequal variance.
    Returns None if scipy is not installed or samples are too small.
    We use this as a secondary signal: the primary alert is the % delta
    """
    if len(a) < 2 or len(b) < 2:
        return None 
    try:
        from scipy import stats 
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except ImportError:
        return None 

#--Drift detection--------------------------------------------------------------------------------------

def detect_drift(
    baseline_values: list[float],
    recent_values: list[float],
    metric: str,
    direction: str,
    tolerance: float,
) -> Optional[DriftAlert]:
    """
    Compare baseline and recent windows.
    Returns a DriftAlert if the relative change exceeds `tolerance`,
    or None if there's insufficient data or no meaningful shift
    """
    if len(baseline_values) < MIN_SAMPLES or len(recent_values) < MIN_SAMPLES:
        logger.debug(
            "Skipping %s - insufficient samples (baseline=%d, recent=%d)",
            metric, len(baseline_values), len(recent_values),
        )
        return None 
    
    baseline_mean = statistics.mean(baseline_values)
    recent_mean = statistics.mean(recent_values)

    if baseline_mean == 0:
        return None 

    delta = recent_mean - baseline_mean 
    delta_pct = delta / abs(baseline_mean)

    #degradation: quality metric dropped, or cost metric rose
    is_degradation = (
        (direction == "up" and delta_pct < -tolerance) or 
        (direction == "down" and delta_pct > tolerance)
    )

    #only alert if change exceeds tolerance in either direction
    if abs(delta_pct) < tolerance:
        return None 
    
    p_value = welch_t_test(baseline_values, recent_values)
    significant = p_value is not None and p_value < 0.05

    return DriftAlert(
        metric=metric,
        direction=direction,
        baseline_mean=round(baseline_mean, 4),
        recent_mean=round(recent_mean, 4),
        delta=round(delta, 4),
        delta_pct=round(delta_pct * 100, 2),
        is_degradation=is_degradation,
        p_value=round(p_value, 4) if p_value is not None else None,
        significant=significant,
    )

#--Report generation-------------------------------------------------------

def generate_report(
    conn: sqlite3.Connection,
    baseline_days: int,
    recent_days: int,
    tolerance: float,
) -> DriftReport:
    now = datetime.now(tz=timezone.utc)
    recent_start = now - timedelta(days=recent_days)
    baseline_start = now - timedelta(days=baseline_days + recent_days)
    baseline_end = recent_start 

    report = DriftReport(
        generated_at=now.isoformat(),
        baseline_days=baseline_days,
        recent_days=recent_days,
        tolerance=tolerance,
    )

    alerts: list[DriftAlert] = []

    for table, metric, direction in MONITORED_METRICS:
        baseline_vals = fetch_window(conn, table, metric, baseline_start, baseline_end)
        recent_vals = fetch_window(conn, table, metric, recent_start, now)

        baseline_summary = summarize_window(baseline_vals, metric, table)
        recent_summary = summarize_window(recent_vals, metric, table)

        report.baseline_window[metric] = asdict(baseline_summary)
        report.recent_window[metric] = asdict(recent_summary)

        alert = detect_drift(baseline_vals, recent_vals, metric, direction, tolerance)
        if alert:
            alerts.append(alert)
            logger.warning(
                "DRIFT: %s %+.1f% (baseline=%.3f -> recent=%.3f)s",
                metric,
                alert.delta_pct,
                alert.baseline_mean,
                alert.recent_mean,
                "[significant]" if alert.significant else "",

            )
        
        report.alert = [asdict for a in alerts]
        degradations = [a for a in alerts if a.is_degradation]
        report.drift_detected = len(degradations) > 0

        #human-readable report
        lines = [
            f"Drift Report - {now.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Baseline: {baseline_days}d prior to recent window",
            f"Recent: last {recent_days}d",
            f"Tolerance: {tolerance*100:.0f}%",
            ""
        ]

        if not alerts:
            lines.append("No significant drift detected across all monitored metrics.")
        else:
            lines.append(f"{'Metric':<30} {'Baseline':>10} {'Recent':>10}, {'Delta':<8} {'Status'}")
            lines.append("-"*75)
            for a in alerts:
                arrow = "▼" if a.delta_pct < 0 else "▲"
                status = "⚠ DEGRADED" if a.is_degradation else "ℹ CHANGED"
                if a.significant:
                    lines.append(
                        f"{a.metric:<30} {a.baseline_mean:10.3f} {a.recent_mean:10.3f} "
                        f"{arrow}{abs(a.delta_pct):6.1f}%   {status}"
                    )
            if any(a.significant for a in alerts):
                lines.append("\n* statistically significant (p < 0.05)")
        
        report.summary = "\n".join(lines)
        return report 


def print_report(report: DriftReport):
    print("\n" + "=" * 75)
    print(report.summary)
    print("\n" * 75 + "\n")

#--CLI--------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect quality/performance drift in podcast summaries"
    )
    parser.add_argument(
        "--baseline_days", type=int, default=30,
        help="Days of history to use a baseline (default: 30)"
    )
    parser.add_argument(
        "--recent_days", type=int, default=7,
        help="Days of recent runs to compare against baseline (default: 7)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.10,
        help="Relative change threshold to trigger alert, e.g. 0.10 = 10% (default: 0.10)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Save report JSON to this path"
    )
    parser.add_argument(
        "--db", type=Path, default=DB_PATH,
        help=f"SQLite DB path (default: {DB_PATH})"
    )
    parser.add_argument(
        "--ci_mode", action="store_true",
        help="Exit with code 1 if quality degradation is detected"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.db.exists():
        logger.error(
            "Monitoring DB not found at %s. Run log_io.py --all first", args.db 
        )
        sys.exit(1)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row 

    report = generate_report(
        conn=conn,
        baseline_days=args.baseline_days,
        recent_days=args.recent_days,
        tolerance=args.tolerance,
    )

    print_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report.to_json(), f, indent=2)
        logger.info("Report saved: %s", args.output)

    conn.close()

    if args.ci_mode and report.drift_detected:
        logger.error("Quality degradation detected. Review drift report before deploying")
        sys.exit(1)


if __name__ == "__main__":
    main()