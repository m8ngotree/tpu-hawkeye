"""Print an agent trajectory compactly.

    python scripts/show_trajectory.py results/runs/smoke/taxonomy/12p_RMSNorm_r0
"""

import json
import sys
from pathlib import Path


def clip(text, n):
    text = str(text).replace("\n", "\\n")
    return text if len(text) <= n else text[:n] + f"...[+{len(text) - n}]"


def main():
    run = Path(sys.argv[1])
    result = run / "result.json"
    if result.exists():
        r = json.loads(result.read_text())
        print("RESULT:", {k: r.get(k) for k in ("status", "correct", "speedup_vs_baseline", "turns_used", "stopped_reason", "error")})
        print()
    for line in (run / "trajectory.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e["role"] == "assistant":
            if e.get("content"):
                print(f"[{e['turn']}] AGENT: {clip(e['content'], 400)}")
            for tc in e.get("tool_calls", []):
                fn = tc["function"]
                print(f"[{e['turn']}] CALL {fn['name']}: {clip(fn['arguments'], 300)}")
        else:
            print(f"[{e['turn']}]   -> {clip(e['output'], 500)}")
    kernel = run / "kernel.py"
    if kernel.exists():
        print("\n===== final kernel.py =====")
        print(kernel.read_text()[:2500])


if __name__ == "__main__":
    main()
