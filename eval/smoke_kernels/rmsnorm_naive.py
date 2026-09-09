"""Smoke-test candidate for 12p_RMSNorm.

Deliberately just re-implements the baseline's math (no Pallas, no optimization) --
the point of this file isn't to be fast, it's to prove agent/runner.py can load a
candidate kernel, run it, and check it against JAXBench's baseline through the full
harness plumbing before any real kernel-optimization work starts.
"""

import jax.numpy as jnp
from jax import lax

EPSILON = 1e-5


def workload(x, scale):
    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(lax.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * lax.rsqrt(mean2 + EPSILON)
    normed = jnp.asarray(normed, x.dtype)
    return normed * scale
