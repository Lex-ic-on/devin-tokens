"""
devin-tokens — Devin CLI token usage aggregator
Reads from ~/.local/share/devin/cli/transcripts/

Aggregation is done at the **step** level: each agent step carries its own
token metrics (``step["metrics"]``) and timestamp, so a single session that
spans multiple days contributes to each of those days individually.

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
import textwrap
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

TRANSCRIPT_DIR = Path.home() / ".local/share/devin/cli/transcripts"


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


def extract_steps(data: dict) -> list[dict]:
    """Return one record per agent step that carries token metrics.

    Each record is the minimum aggregation unit and carries its own
    timestamp so it can be grouped by day independently of its session.
    """
    session_id = data.get("session_id", "")
    title = data.get("agent", {}).get("title") or ""
    # ACP sessions (synthetic transcripts from Hermes-via-ACP) carry an
    # _acp_metadata field.  They report cached_tokens=0 because the ACP
    # protocol does not expose cache token counts.
    is_acp = "_acp_metadata" in data

    records = []
    for step in data.get("steps", []):
        m = step.get("metrics")
        if not m:
            # system / user / tool steps do not consume model tokens
            continue
        prompt = m.get("prompt_tokens", 0) or 0
        completion = m.get("completion_tokens", 0) or 0
        cached = m.get("cached_tokens", 0) or 0
        cache_cre = (m.get("extra") or {}).get("cache_creation_input_tokens", 0) or 0
        ts = parse_ts(step["timestamp"])
        records.append({
            "session_id": session_id,
            "title": title,
            "is_acp": is_acp,
            "step_id": step["step_id"],
            "timestamp": ts,
            "model": step.get("extra", {}).get("generation_model", ""),
            "source": step.get("source", ""),
            # Raw token fields (prompt_tokens includes cached input).
            "prompt": prompt,
            "completion": completion,
            "cached": cached,
            "cache_cre": cache_cre,
        })
    return records


def step_net_input(r: dict) -> int:
    """Net input = prompt_tokens - cached_tokens (excludes cache hits)."""
    return r["prompt"] - r["cached"]


def fmt_date(dt: datetime | None) -> str:
    if dt is None:
        return "?"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def _fmt_cached(n: int, is_acp: bool) -> str:
    """Format cached tokens. ACP sessions show '---' (unknown)."""
    if is_acp:
        return f"{'---':>{10}}"
    return f"{n:>10,}"


def print_session_detail(steps: list[dict]) -> None:
    """Print per-step breakdown for a single session."""
    if not steps:
        print("  (no steps with token metrics)")
        return
    s0 = steps[0]
    session_id = s0["session_id"]
    title = s0["title"]
    is_acp = s0["is_acp"]
    start = min(r["timestamp"] for r in steps)
    end = max(r["timestamp"] for r in steps)

    print(f"\nSession: {session_id}")
    print(f"Title  : {title}")
    print(f"Start  : {fmt_date(start)}")
    print(f"End    : {fmt_date(end)}")
    print(f"Steps  : {len(steps)}  ({'ACP' if is_acp else 'direct'})")
    print()
    print(f"  {'step':>4}  {'timestamp':<17}  {'model':<22}  {'input':>10}  {'output':>8}  {'cached':>10}  {'cache_cre':>9}")
    print("  " + "-" * 95)
    for r in sorted(steps, key=lambda x: x["step_id"]):
        ts = r["timestamp"].astimezone().strftime("%m-%d %H:%M:%S")
        net_in = step_net_input(r)
        cached_str = _fmt_cached(r["cached"], is_acp)
        print(f"  {r['step_id']:>4}  {ts:<17}  {r['model']:<22}  {net_in:>10,}  {r['completion']:>8,}  {cached_str}  {r['cache_cre']:>9,}")
    print()

    tot_input = sum(step_net_input(r) for r in steps)
    tot_output = sum(r["completion"] for r in steps)
    tot_cached = sum(r["cached"] for r in steps)
    tot_cre = sum(r["cache_cre"] for r in steps)
    cached_str = _fmt_cached(tot_cached, is_acp)
    print(f"  TOTAL  input={tot_input:>12,}  output={tot_output:>10,}  cached={cached_str}  cache_cre={tot_cre:>10,}")
    print()


def print_daily_table(steps: list[dict]) -> None:
    """Group steps by (date, source) and print a ccusage-style daily table.

    Each day always shows two rows: ``direct`` (normal CLI sessions) and
    ``ACP`` (Hermes-via-ACP sessions).  The date is only printed on the
    ``direct`` row; the ``ACP`` row leaves it blank so the visual grouping
    is clear.  ACP rows show ``---`` for cached tokens because the ACP
    protocol does not report cache counts.

    The ``sessions`` column counts distinct sessions that had at least one
    step on that day (a session spanning multiple days appears on each).
    """
    # Group by (date, source) where source is "direct" or "ACP".
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in steps:
        day = r["timestamp"].astimezone().strftime("%Y-%m-%d")
        src = "ACP" if r["is_acp"] else "direct"
        by_group[(day, src)].append(r)

    col_w = 10  # token column width
    print()
    print(f"  {'date':<12}  {'src':<6}  {'sessions':>8}  {'input':>{col_w}}  {'output':>8}  {'cached':>{col_w}}")
    print("  " + "-" * 70)

    grand_input = grand_output = grand_cached = 0
    grand_sessions: set[str] = set()
    all_days = sorted({d for d, _ in by_group})
    for day in all_days:
        for src in ("direct", "ACP"):
            group_steps = by_group.get((day, src), [])
            is_acp = src == "ACP"
            day_input = sum(step_net_input(r) for r in group_steps)
            day_output = sum(r["completion"] for r in group_steps)
            day_cached = sum(r["cached"] for r in group_steps)
            grand_input += day_input
            grand_output += day_output
            # Only add cached to grand total if not ACP (ACP cached is unknown).
            if not is_acp:
                grand_cached += day_cached
                grand_sessions.update(r["session_id"] for r in group_steps)
            cached_str = _fmt_cached(day_cached, is_acp)
            session_ids = ", ".join(sorted({r["session_id"] for r in group_steps}))
            # Show date only on the first (direct) row; blank on ACP row.
            date_label = day if not is_acp else ""
            n_sessions = len({r["session_id"] for r in group_steps})
            row_prefix = f"  {date_label:<12}  {src:<6}  {n_sessions:>8}  {day_input:>{col_w},}  {day_output:>8,}  {cached_str}  "
            indent = " " * len(row_prefix)
            # Wrap only the session text to the available column width; handle
            # indentation ourselves so continuation lines align under the first
            # session name without being double-indented or over-stretched.
            available = 52  # chars available for session text per line
            wrapped = textwrap.wrap(
                session_ids, width=available, break_on_hyphens=False
            )
            print(row_prefix + (("\n" + indent).join(wrapped) if wrapped else ""))

    print("  " + "-" * 70)
    # Grand total cached: sum known (direct) values even when ACP is present.
    cached_str = f"{grand_cached:>{col_w},}"
    print(f"  {'TOTAL':<12}  {'':6}  {len(grand_sessions):>8}  {grand_input:>{col_w},}  {grand_output:>8,}  {cached_str}")
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

    # Load all steps across all sessions (flat list, the minimum unit).
    all_steps: list[dict] = []
    for p in sorted(TRANSCRIPT_DIR.glob("*.json")):
        data = load_session(p)
        if data:
            all_steps.extend(extract_steps(data))

    if not all_steps:
        print("No sessions with token metrics found.")
        return

    # Single-session detail mode: filter by session_id, ignore date range.
    if args.session:
        matched = [r for r in all_steps if r["session_id"] == args.session]
        if not matched:
            print(f"Session '{args.session}' not found.", file=sys.stderr)
            sys.exit(1)
        print_session_detail(matched)
        return

    # Determine date range (applied per step via step timestamp).
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

    # Filter steps by their own timestamp.
    filtered: list[dict] = []
    for r in all_steps:
        if since and r["timestamp"] < since:
            continue
        if until and r["timestamp"] > until:
            continue
        filtered.append(r)

    if not filtered:
        print("No steps in the specified date range.")
        return

    # List mode: one row per (session, day).
    if args.list:
        by_session_day: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in filtered:
            day = r["timestamp"].astimezone().strftime("%Y-%m-%d")
            by_session_day[(r["session_id"], day)].append(r)

        print()
        print(f"  {'session':<22}  {'src':<6}  {'date':<12}  {'steps':>5}  {'input':>10}  {'output':>8}  {'cached':>10}  title")
        print("  " + "-" * 110)
        rows = []
        for (sid, day), group in by_session_day.items():
            group_sorted = sorted(group, key=lambda x: x["timestamp"])
            rows.append((sid, day, group_sorted))
        rows.sort(key=lambda x: (x[1], x[0]))
        for sid, day, group in rows:
            is_acp = group[0]["is_acp"]
            src = "ACP" if is_acp else "direct"
            title = group[0]["title"][:30]
            net_input = sum(step_net_input(r) for r in group)
            output = sum(r["completion"] for r in group)
            cached = sum(r["cached"] for r in group)
            cached_str = _fmt_cached(cached, is_acp)
            print(f"  {sid:<22}  {src:<6}  {day:<12}  {len(group):>5}  {net_input:>10,}  {output:>8,}  {cached_str}  {title}")
        print()

        total_input = sum(step_net_input(r) for r in filtered)
        total_output = sum(r["completion"] for r in filtered)
        n_sessions = len({r["session_id"] for r in filtered})
        # Sum cached from direct sessions only (ACP cached is unknown).
        total_cached = sum(r["cached"] for r in filtered if not r["is_acp"])
        print(f"  {n_sessions} sessions  |  input: {total_input:,}  output: {total_output:,}  cached: {total_cached:,}")
        print()
        return

    # Default: daily table
    range_str = ""
    if not args.all:
        range_str = f"  ({since.strftime('%Y-%m-%d')} \u2013 {until.strftime('%Y-%m-%d')})"
    print(f"\nDevin CLI token usage{range_str}")
    print_daily_table(filtered)

    print("  Use --session <id> to see per-step breakdown.")
    print()


if __name__ == "__main__":
    main()
