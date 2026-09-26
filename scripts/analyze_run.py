"""Analyse experiment runs: progress over time, compliance audit, taxonomy use.

    python scripts/analyze_run.py results/runs/<tag>/<condition>/<workload>_r<rep>   # one run in detail
    python scripts/analyze_run.py results/runs/<tag>                                 # table over a whole tag

One run: the sequence of evaluations (turn, status, speedup, distance to the roofline limit),
the taxonomy files read and when, the tools used, and the compliance counters (tool errors,
attempts to leave the workspace, calls to tools that do not exist, guard rejections).
A tag: one row per run for comparing conditions.
"""

import json
import sys
from pathlib import Path


def load(path):
    return json.loads((path / "result.json").read_text())


def one_run(run):
    r = load(run)
    print(f"{r['workload']}  condition={r['condition']}  rep={r['rep']}  tag={r['tag']}  code={r.get('code_version')}")
    print("settings:", r.get("settings"))
    print(f"result: status={r.get('status')} speedup={r.get('speedup_vs_baseline')} "
          f"geomean_score={r.get('speedup_for_geomean')} stopped={r.get('stopped_reason')}")
    print(f"turns: raw={r.get('turns_used')} productive={r.get('productive_turns')}  "
          f"tokens in/out={r.get('prompt_tokens')}/{r.get('completion_tokens')}  "
          f"api_s={r.get('api_seconds')} tool_s={r.get('tool_seconds')} wall_s={r.get('wall_s')}")

    print("\nEVALUATIONS (progress)")
    print(f"{'#':>3} {'turn':>5} {'prod':>5} {'writes':>6}  {'status':<14}{'speedup':>8} {'%roofline':>10}  limit / error")
    best = None
    for e in r.get("eval_history") or []:
        sp = e.get("speedup")
        if e.get("status") == "correct" and sp is not None and (best is None or sp > best):
            best = sp
        tail = e.get("workload_limit") or (e.get("error") or "")[:70]
        print(f"{e['eval_number']:>3} {e['turn']:>5} {e.get('productive_turns_used', ''):>5} {e['kernel_writes_so_far']:>6}  "
              f"{str(e['status']):<14}{'' if sp is None else sp:>8} {'' if e.get('pct_of_roofline_limit') is None else e['pct_of_roofline_limit']:>10}  {tail}")
    first = next((e for e in r.get("eval_history") or [] if e["status"] == "correct"), None)
    if first is None:
        first_text = "never"
    else:
        first_text = "eval #%d at raw turn %d" % (first["eval_number"], first["turn"])
    print("first correct kernel: %s; best speedup seen: %s" % (first_text, best))

    print("\nTAXONOMY READS (turn, file)")
    reads = r.get("taxonomy_reads") or []
    for t in reads:
        print(f"  turn {t['turn']:>3}  {t['path']}")
    if not reads:
        print("  none")
    files = r.get("taxonomy_files_read") or []
    first_write = next((e["turn"] for e in r.get("eval_history") or []), None)
    print(f"  unique files: {len(files)}; first evaluation at turn {first_write}")

    print("\nCOMPLIANCE")
    print("  tool calls:", r.get("tool_counts"))
    allowed = {"read_file", "write_file", "list_files", "run_eval"}
    bad = {k: v for k, v in (r.get("tool_counts") or {}).items() if k not in allowed}
    print(f"  calls to tools that do not exist: {bad or 0}")
    print(f"  tool errors: {r.get('tool_errors')}  path-escape attempts: {r.get('path_escape_attempts')}  "
          f"guard rejections: {r.get('guard_rejections')}  kernel writes: {r.get('kernel_writes')}")
    kernels = sorted((run / "kernels").glob("*.py")) if (run / "kernels").exists() else []
    print("  kernel snapshots saved: %d (in %s)" % (len(kernels), run / "kernels"))


def tag_table(tag):
    rows = []
    for rd in sorted(tag.glob("*/*_r*/result.json")):
        rows.append(json.loads(rd.read_text()))
    print(f"{'workload':<34}{'cond':<10}{'status':<18}{'speedup':>8}{'prod':>6}{'raw':>5}{'evals':>6}{'ok':>4}{'tax':>5}{'esc':>5}{'Mtok':>7}")
    for r in rows:
        ev = r.get("eval_history") or []
        ok = sum(1 for e in ev if e["status"] == "correct")
        sp = r.get("speedup_vs_baseline")
        print(f"{r['workload']:<34}{r['condition']:<10}{str(r.get('status')):<18}{'' if sp is None else sp:>8}"
              f"{r.get('productive_turns', ''):>6}{r.get('turns_used', ''):>5}{len(ev):>6}{ok:>4}"
              f"{len(r.get('taxonomy_files_read') or []):>5}{r.get('path_escape_attempts', ''):>5}"
              f"{(r.get('prompt_tokens') or 0) / 1e6:>7.2f}")


def main():
    path = Path(sys.argv[1])
    if (path / "result.json").exists():
        one_run(path)
    else:
        tag_table(path)


if __name__ == "__main__":
    main()
