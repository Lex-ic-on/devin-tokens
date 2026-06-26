"""
devin-tokens — Devin CLI token usage aggregator
Reads from ~/.local/share/devin/cli/transcripts/

Usage:
  devin-tokens                  # last 14 days (default)
  devin-tokens --since 2026-06-01
  devin-tokens --until 2026-06-30
  devin-tokens --since 2026-06-01 --until 2026-06-30
  devin-tokens --days 7         # last N days
  devin-tokens --all            # all sessions
  devin-tokens --session rough-neon   # single session detail
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRANSCRIPT_DIR = Path.home() / ".local/share/devin/cli/transcripts"

# Per-step token fields added in a future version of Devin CLI (2026.5.26-0 changelog).
# When present in step["extra"]["telemetry"], these enable per-turn breakdown.
STEP_TOKEN_FIELDS = (
    "total_input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)


def parse_ts(ts_str: str) -> datetime:
    """Parse ISO 8601 timestamp to UTC datetime."""
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    return datetime.fromisoformat(ts_str).astimezone(timezone.utc)


def load_session(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] could not read {path.name}: {e}", file=sys.stderr)
        return None


def session_summary(data: dict) -> dict:
    """Extract summary info from a transcript."""
    fm = data.get("final_metrics", {})
    steps = data.get("steps", [])
    start_ts = parse_ts(steps[0]["timestamp"]) if steps else None
    end_ts = parse_ts(steps[-1]["timestamp"]) if steps else None
    title = data.get("agent", {}).get("title") or ""

    # Per-step breakdown (available in newer versions)
    per_step = []
    for step in steps:
        tel = step.get("extra", {}).get("telemetry", {})
        if any(k in tel for k in STEP_TOKEN_FIELDS):
            per_step.append({
                "step_id": step["step_id"],
                "timestamp": step["timestamp"],
                "source": step.get("source", ""),
                "model": step.get("extra", {}).get("generation_model", ""),
                "input": tel.get("total_input_tokens", 0) or 0,
                "output": tel.get("output_tokens", 0) or 0,
                "cache_read": tel.get("cache_read_tokens", 0) or 0,
                "cache_creation": tel.get("cache_creation_tokens", 0) or 0,
            })

    return {
        "session_id": data.get("session_id", ""),
        "title": title,
        "start": start_ts,
        "end": end_ts,
        "input": fm.get("total_prompt_tokens", 0) or 0,
        "output": fm.get("total_completion_tokens", 0) or 0,
        "cached": fm.get("total_cached_tokens", 0) or 0,
        "steps_count": fm.get("total_steps", len(steps)),
        "has_step_detail": len(per_step) > 0,
        "per_step": per_step,
    }


def fmt_num(n: int) -> str:
    return f"{n:>10,}"


def fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def print_session_detail(s: dict) -> None:
    """Print per-step breakdown for a single session."""
    print(f"\nSession: {s['session_id']}")
    print(f"Title  : {s['title']}")
    print(f"Start  : {fmt_date(s['start'])}")
    print(f"End    : {fmt_date(s['end'])}")
    print(f"Steps  : {s['steps_count']}")
    print()
    if s["has_step_detail"]:
        print(f"  {'step':>4}  {'timestamp':<17}  {'model':<22}  {'input':>10}  {'output':>8}  {'cache_read':>10}  {'cache_cre':>9}")
        print("  " + "-" * 95)
        for st in s["per_step"]:
            ts = parse_ts(st["timestamp"]).astimezone().strftime("%m-%d %H:%M:%S")
            print(f"  {st['step_id']:>4}  {ts:<17}  {st['model']:<22}  {st['input']:>10,}  {st['output']:>8,}  {st['cache_read']:>10,}  {st['cache_creation']:>9,}")
        print()
    else:
        print("  (per-step token data not available — requires a newer version of Devin CLI)")
        print()

    print(f"  TOTAL  in(+cache)={s['input']:>12,}  output={s['output']:>10,}  cached={s['cached']:>12,}")
    print()


def print_daily_table(sessions: list[dict]) -> None:
    """Group sessions by date and print a ccusage-style daily table."""
    by_date: dict[str, list] = defaultdict(list)
    for s in sessions:
        day = s["start"].astimezone().strftime("%Y-%m-%d") if s["start"] else "unknown"
        by_date[day].append(s)

    col_w = 10  # token column width
    print()
    print(f"  {'date':<12}  {'sessions':>8}  {'in(+cache)':>{col_w}}  {'output':>8}  {'cached':>{col_w}}")
    print("  " + "-" * 60)

    grand_input = grand_output = grand_cached = 0
    for day in sorted(by_date):
        day_sessions = by_date[day]
        day_input = sum(s["input"] for s in day_sessions)
        day_output = sum(s["output"] for s in day_sessions)
        day_cached = sum(s["cached"] for s in day_sessions)
        grand_input += day_input
        grand_output += day_output
        grand_cached += day_cached
        session_ids = ", ".join(s["session_id"] for s in day_sessions)
        print(f"  {day:<12}  {len(day_sessions):>8}  {day_input:>{col_w},}  {day_output:>8,}  {day_cached:>{col_w},}  {session_ids}")

    print("  " + "-" * 60)
    total_sessions = len(sessions)
    print(f"  {'TOTAL':<12}  {total_sessions:>8}  {grand_input:>{col_w},}  {grand_output:>8,}  {grand_cached:>{col_w},}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate Devin CLI token usage from transcript files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="Start date (inclusive)")
    parser.add_argument("--until", metavar="YYYY-MM-DD", help="End date (inclusive)")
    parser.add_argument("--days", metavar="N", type=int, help="Last N days (default: 14)")
    parser.add_argument("--all", action="store_true", help="Show all sessions (ignore date filter)")
    parser.add_argument("--session", metavar="ID", help="Show per-step detail for one session")
    parser.add_argument("--list", action="store_true", help="List all sessions with token totals")
    args = parser.parse_args()

    if not TRANSCRIPT_DIR.exists():
        print(f"No transcript directory found at {TRANSCRIPT_DIR}", file=sys.stderr)
        sys.exit(1)

    # Load all sessions
    all_sessions = []
    for p in sorted(TRANSCRIPT_DIR.glob("*.json")):
        data = load_session(p)
        if data:
            all_sessions.append(session_summary(data))

    if not all_sessions:
        print("No sessions found.")
        return

    # Single-session detail mode
    if args.session:
        matched = [s for s in all_sessions if s["session_id"] == args.session]
        if not matched:
            print(f"Session '{args.session}' not found.", file=sys.stderr)
            sys.exit(1)
        print_session_detail(matched[0])
        return

    # Determine date range
    now = datetime.now(timezone.utc)
    if args.all:
        since = until = None
    else:
        if args.since:
            since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        elif args.days:
            since = now - timedelta(days=args.days)
        else:
            since = now - timedelta(days=14)  # default: 2 weeks

        if args.until:
            until = datetime.fromisoformat(args.until).replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
        else:
            until = now

    # Filter sessions
    filtered = []
    for s in all_sessions:
        if s["start"] is None:
            continue
        if since and s["start"] < since:
            continue
        if until and s["start"] > until:
            continue
        filtered.append(s)

    if not filtered:
        print("No sessions in the specified date range.")
        return

    # List mode: one row per session
    if args.list:
        print()
        print(f"  {'session':<22}  {'start':<17}  {'in(+cache)':>10}  {'output':>8}  {'cached':>10}  title")
        print("  " + "-" * 100)
        for s in sorted(filtered, key=lambda x: x["start"]):
            has_detail = " *" if s["has_step_detail"] else "  "
            title = s["title"][:30]
            print(f"  {s['session_id']:<22}{has_detail} {fmt_date(s['start']):<17}  {s['input']:>10,}  {s['output']:>8,}  {s['cached']:>10,}  {title}")
        print()
        print("  (* = per-step token data available)")
        print()
        total_input = sum(s["input"] for s in filtered)
        total_output = sum(s["output"] for s in filtered)
        total_cached = sum(s["cached"] for s in filtered)
        print(f"  {len(filtered)} sessions  |  in(+cache): {total_input:,}  output: {total_output:,}  cached: {total_cached:,}")
        print()
        return

    # Default: daily table
    range_str = ""
    if not args.all:
        range_str = f"  ({since.strftime('%Y-%m-%d')} \u2013 {until.strftime('%Y-%m-%d')})"
    print(f"\nDevin CLI token usage{range_str}")
    print_daily_table(filtered)

    # Hint about per-step detail
    has_detail_any = any(s["has_step_detail"] for s in filtered)
    if has_detail_any:
        print("  Use --session <id> to see per-step breakdown.")
    else:
        print("  Per-step breakdown will be available after upgrading Devin CLI (changelog: 2026.5.26-0).")
    print()
