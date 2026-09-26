"""Evaluate a candidate kernel against a JAXBench workload.

    python eval.py --workload 12p_RMSNorm --kernel kernel.py [--interpret]

Inside an agent workspace this uses the workspace's own copy of the JAXBench harness
(no access to anything else is needed). Run from the repo's eval/ directory it falls
back to the JAXBench checkout under external/. Prints the JSON result; exits 0 iff the
kernel is correct.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "JAXBench").exists():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(HERE.parent / "external" / "accelerator-agents"))


def _record(result: dict, kernel_path: Path) -> None:
    """Keep a log of evaluations and a copy of the fastest correct kernel seen so far."""
    speedup = result.get("speedup_vs_baseline")
    reference = HERE / "baseline.py"
    is_reference = reference.exists() and Path(kernel_path).read_text() == reference.read_text()
    entry = {"time": time.time(), "status": result.get("status"), "speedup_vs_baseline": speedup,
             "is_reference": is_reference}
    with open(HERE / "eval_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    if result.get("status") != "correct" or speedup is None:
        return
    best_file = HERE / "best_score.json"
    best = json.loads(best_file.read_text())["speedup_vs_baseline"] if best_file.exists() else -1
    if speedup > best:
        shutil.copy(kernel_path, HERE / "best_kernel.py")
        best_file.write_text(json.dumps({"speedup_vs_baseline": speedup}))


def _diagnosis(result: dict, workload: str, tpu: str) -> dict | None:
    """Roofline view of the candidate kernel's timing.

    Minimum HBM traffic is the workload's inputs plus its output, each moved once. Compared
    against the chip's peak bandwidth and peak FLOPs this says whether the workload is
    limited by memory or by compute, and how close the kernel is to that limit.
    """
    kernel = result.get("kernel") or {}
    ms = kernel.get("median_ms")
    if not ms:
        return None
    try:
        import jax
        import jax.numpy as jnp
        import JAXBench
        from JAXBench.harness.loader import load_module
        from JAXBench.harness.tpu_specs import TPU_SPECS

        path = Path(JAXBench.__file__).parent / "benchmark" / workload / "baseline.py"
        mod = load_module(str(path), f"{workload}.diag")
        create = mod.create_inputs
        inputs = create(dtype=jnp.bfloat16) if "dtype" in create.__code__.co_varnames else create()
        inputs = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
        in_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(inputs))
        out = jax.eval_shape(mod.workload, *inputs)
        out_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(out))
    except Exception:  # noqa: BLE001 - diagnosis is best-effort, never fail an evaluation
        return None

    spec = TPU_SPECS[tpu]
    seconds = ms / 1000.0
    min_bytes = in_bytes + out_bytes
    flops = (kernel.get("tflops") or 0.0) * 1e12 * seconds
    peak_flops = spec["peak_tflops_bf16"] * 1e12
    peak_bw = spec["hbm_bandwidth_gbs"] * 1e9
    diag = {
        "min_hbm_traffic_mb": round(min_bytes / 1e6, 2),
        "achieved_hbm_gbs": round(min_bytes / seconds / 1e9, 1),
        "hbm_bandwidth_pct_of_peak": round(min_bytes / seconds / peak_bw * 100, 1),
        "mxu_pct_of_peak": kernel.get("utilization_pct"),
    }
    if flops > 0:
        intensity = flops / min_bytes
        ridge = peak_flops / peak_bw
        limit_seconds = max(flops / peak_flops, min_bytes / peak_bw)
        diag.update(
            arithmetic_intensity_flop_per_byte=round(intensity, 1),
            ridge_point_flop_per_byte=round(ridge, 1),
            workload_limit="memory-bound" if intensity < ridge else "compute-bound",
            pct_of_roofline_limit=round(limit_seconds / seconds * 100, 1),
        )
    else:
        diag.update(workload_limit="memory-bound (no FLOP count available)",
                    pct_of_roofline_limit=round(min_bytes / peak_bw / seconds * 100, 1))
    return diag


_MATMUL_PRIMITIVES = {"dot_general", "conv_general_dilated"}


def _subjaxprs(eqn):
    for value in eqn.params.values():
        for item in (value if isinstance(value, (list, tuple)) else [value]):
            inner = getattr(item, "jaxpr", item)
            if hasattr(inner, "eqns"):
                yield inner


def _has_pallas(eqn) -> bool:
    return eqn.primitive.name == "pallas_call" or any(
        _has_pallas(e) for sub in _subjaxprs(eqn) for e in sub.eqns)


def _outside_primitives(jaxpr, counts):
    """Count primitives that run outside any pallas_call."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            continue
        subs = list(_subjaxprs(eqn))
        if not subs:
            counts[eqn.primitive.name] = counts.get(eqn.primitive.name, 0) + 1
        for sub in subs:
            _outside_primitives(sub, counts)


def _pallas_audit(kernel_path: Path, workload: str) -> dict | None:
    """Trace the kernel's workload() into a JAX program (no execution) and check that the work is
    done by Pallas: at least one pallas_call, the output depends on one, and no matmul or
    convolution runs outside Pallas. Returns None if the kernel cannot be traced abstractly."""
    try:
        import jax
        import jax.numpy as jnp
        import JAXBench
        from JAXBench.harness.loader import load_module

        base_path = Path(JAXBench.__file__).parent / "benchmark" / workload / "baseline.py"
        base = load_module(str(base_path), f"{workload}.audit_base")
        create = base.create_inputs
        specs = jax.eval_shape(lambda: create(dtype=jnp.bfloat16) if "dtype" in create.__code__.co_varnames else create())
        specs = specs if isinstance(specs, (list, tuple)) else (specs,)
        kernel = load_module(str(kernel_path), "audit_kernel")
        closed = jax.make_jaxpr(kernel.workload)(*specs)
    except Exception:  # noqa: BLE001 - an untraceable kernel is judged by the evaluation itself
        return None

    jaxpr = closed.jaxpr
    tainted = set()
    calls = 0
    for eqn in jaxpr.eqns:
        has = _has_pallas(eqn)
        calls += has
        if has or any(v in tainted for v in eqn.invars if not hasattr(v, "val")):
            tainted.update(eqn.outvars)
    counts: dict = {}
    _outside_primitives(jaxpr, counts)
    return {
        "pallas_calls": calls,
        "output_uses_pallas": any(v in tainted for v in jaxpr.outvars if not hasattr(v, "val")),
        "matmul_outside_pallas": sum(counts.get(name, 0) for name in _MATMUL_PRIMITIVES),
        "outside_primitives": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:12]),
    }


_BANNED_SOURCE = re.compile(
    r"getsource|getsourcelines|\binspect\b|importlib|subprocess|\bopen\s*\(|pathlib|\bglob\b|shutil"
    r"|__import__|\bexec\s*\(|\beval\s*\(|\bcompile\s*\(|sys\.modules|os\.(system|popen|listdir|walk|scandir|getenv|environ\.(items|keys|copy))"
    r"|loadtxt|genfromtxt|fromfile|np\.load|numpy\.load|read_text|read_bytes|urllib|requests|socket|\bhttp"
    r"|\bhelp\s*\(|\bdir\s*\(|\bvars\s*\(|\bglobals\s*\(|__file__|__dict__|__doc__|site-packages")
_ENV_USE = re.compile(r"os\.environ[^\n]*")
_ALLOWED_ENV = re.compile(r"os\.environ\.get\(\s*[\"']PALLAS_INTERPRET[\"']")


def _source_problem(source: str) -> str | None:
    """Kernel files run as ordinary Python during evaluation, so they could read files, list
    directories, print library source or read the environment. The agent may only write a kernel:
    reject sources that do any of that. The one environment read allowed is the interpret flag
    used by the taxonomy examples."""
    match = _BANNED_SOURCE.search(source)
    if match:
        return f"kernel must only define the computation; `{match.group(0).strip()}` is not allowed"
    for use in _ENV_USE.findall(source):
        if not _ALLOWED_ENV.match(use):
            return "kernel may only read the PALLAS_INTERPRET environment variable"
    return None


def _perturb_inputs(inputs):
    """A second, equally valid input set: any 2-D integer input that is a permutation of
    range(size) (a page table, for instance) is replaced by a random permutation. A kernel
    that relies on the exact values create_inputs() happens to produce then gives wrong output.
    Returns None if there is nothing to perturb."""
    import numpy as np
    import jax.numpy as jnp

    rng = np.random.default_rng(12345)
    changed, out = False, []
    for x in inputs:
        arr = np.asarray(x) if hasattr(x, "dtype") and hasattr(x, "shape") else None
        if (arr is not None and arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer)
                and np.array_equal(np.sort(arr.ravel()), np.arange(arr.size))):
            out.append(jnp.asarray(rng.permutation(arr.size).reshape(arr.shape).astype(arr.dtype)))
            changed = True
        else:
            out.append(x)
    return tuple(out) if changed else None


def _scale_check(ref, test, fraction=0.02):
    """Error relative to the size of the output: max|diff| must be within `fraction` of
    max|ref|. The harness tolerance (atol 1e-2) is absolute, so it cannot reject a wrong
    kernel when the reference output itself is tiny."""
    import jax
    import numpy as np

    worst = 0.0
    for r, t in zip(jax.tree.leaves(ref), jax.tree.leaves(test)):
        r = np.asarray(r, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)
        scale = float(np.max(np.abs(r)))
        diff = float(np.max(np.abs(r - t)))
        if scale > 0:
            worst = max(worst, diff / scale)
        elif diff > 0:
            return False, float("inf")
    return worst <= fraction, worst


def _robustness(kernel_path: Path, workload: str) -> str | None:
    """Extra correctness checks on top of the harness's: (1) error relative to output scale on
    the standard inputs; (2) the standard comparison and the scale check on perturbed inputs.
    Returns a reason string if the kernel fails, else None. Skipped if it cannot be run."""
    try:
        import jax
        import jax.numpy as jnp
        import JAXBench
        from JAXBench.harness.correctness import check_correctness
        from JAXBench.harness.loader import load_module

        base_path = Path(JAXBench.__file__).parent / "benchmark" / workload / "baseline.py"
        base = load_module(str(base_path), f"{workload}.robust_base")
        kernel = load_module(str(kernel_path), "robust_kernel")
        create = base.create_inputs
        inputs = create(dtype=jnp.bfloat16) if "dtype" in create.__code__.co_varnames else create()
        inputs = inputs if isinstance(inputs, (list, tuple)) else (inputs,)

        def run(mod, args):
            fn = mod.workload if getattr(mod, "_skip_jit", False) else jax.jit(mod.workload)
            return jax.block_until_ready(fn(*args))

        ok, err = _scale_check(run(base, inputs), run(kernel, inputs))
        if not ok:
            return f"error is {err:.1%} of the output's magnitude on the standard inputs (limit 2%)"
        other = _perturb_inputs(inputs)
        if other is not None:
            ref, out = run(base, other), run(kernel, other)
            if not check_correctness(ref, out)["correct"]:
                return "output is wrong when the integer index inputs are a different valid permutation"
            ok, err = _scale_check(ref, out)
            if not ok:
                return (f"output is wrong for a different valid index permutation "
                        f"(error {err:.1%} of the output's magnitude)")
    except Exception:  # noqa: BLE001 - a check that cannot run must not fail an evaluation
        return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, help="JAXBench workload name, e.g. 12p_RMSNorm")
    parser.add_argument("--kernel", required=True, type=Path, help="path to your kernel.py (must define workload(*inputs))")
    parser.add_argument("--tpu", default="v5e", choices=["v5e", "v6e"])
    parser.add_argument("--interpret", action="store_true", help="run Pallas kernels in the CPU interpreter (correctness only)")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=50)
    args = parser.parse_args()

    source = args.kernel.read_text()
    problem = None
    if "pallas_call" not in source:
        problem = "kernel must implement its main computation with pl.pallas_call; plain JAX is not accepted"
    elif re.search(r"pallas[./]ops|pallas\s+import\s+ops", source):
        problem = "kernel must not use the ready-made kernels in jax.experimental.pallas.ops"
    if not problem:
        problem = _source_problem(source)
    audit = None
    if not problem:
        audit = _pallas_audit(args.kernel, args.workload)
        if audit is not None:
            if audit["pallas_calls"] == 0:
                problem = "the kernel's program contains no pallas_call"
            elif not audit["output_uses_pallas"]:
                problem = "the kernel's output does not depend on any pallas_call"
            elif audit["matmul_outside_pallas"]:
                problem = "matmuls and convolutions must run inside the Pallas kernel, not outside it"
    if problem:
        result = {"workload": args.workload, "status": "rejected", "error": problem}
        if (HERE / "JAXBench").exists():
            _record(result, args.kernel)
        print(json.dumps(result, indent=2))
        sys.exit(1)

    os.environ["PALLAS_INTERPRET"] = "1" if args.interpret else "0"
    from JAXBench.harness.evaluator import evaluate_kernel

    result = evaluate_kernel(
        workload_name=args.workload,
        kernel_path=str(args.kernel),
        tpu=args.tpu,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )
    if audit is not None:
        result["pallas_check"] = audit
    if result.get("status") == "correct":
        reason = _robustness(args.kernel, args.workload)
        if reason:
            result["status"] = "incorrect"
            result["correctness"] = {"correct": False, "reason": reason}
    if result.get("status") == "correct":
        diagnosis = _diagnosis(result, args.workload, args.tpu)
        if diagnosis:
            result["diagnosis"] = diagnosis
    if (HERE / "JAXBench").exists():  # inside an agent workspace
        _record(result, args.kernel)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "correct" else 1)


if __name__ == "__main__":
    main()
