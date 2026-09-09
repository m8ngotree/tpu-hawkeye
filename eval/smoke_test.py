"""End-to-end plumbing check: agent/runner.py -> JAXBench harness -> a real workload,
entirely on CPU (interpret mode, no TPU hours spent).

This is NOT the real evaluation loop (that's eval/run_agent_eval.py, not yet
implemented) -- it only proves the wiring works: JAXBench loads, the candidate kernel
runs, correctness passes, and timing numbers come back. Run it after any change to
agent/runner.py or to confirm a fresh environment is set up correctly.

    python eval/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.runner import run_kernel

WORKLOAD = "12p_RMSNorm"
KERNEL_PATH = Path(__file__).parent / "smoke_kernels" / "rmsnorm_naive.py"


def main() -> None:
    print(f"Running {KERNEL_PATH.name} against workload {WORKLOAD} (interpret=True, CPU)...")
    result = run_kernel(
        workload_name=WORKLOAD,
        kernel_path=KERNEL_PATH,
        tpu="v5e",
        interpret=True,
        num_warmup=1,
        num_iters=2,
    )

    print(f"status:              {result.status}")
    print(f"correct:             {result.correct} (max_diff={result.max_diff})")
    print(f"baseline_median_ms:  {result.baseline_median_ms}")
    print(f"kernel_median_ms:    {result.kernel_median_ms}")
    print(f"speedup_vs_baseline: {result.speedup_vs_baseline}")
    if result.error:
        print(f"error: {result.error}")

    assert result.status == "correct", f"expected status='correct', got {result.status!r}: {result.raw}"
    assert result.correct, "kernel output did not match baseline within tolerance"
    print("\nOK -- plumbing works end-to-end.")


if __name__ == "__main__":
    main()
