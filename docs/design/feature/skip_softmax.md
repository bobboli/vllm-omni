# Skip-Softmax

Skip-Softmax is the sparse-attention mode of the `TRTLLM_ATTN` backend. Usage — the config keys
and how to pick an operating point — is in
[TRTLLM Attention](../../user_guide/diffusion/attention_backends/trtllm.md#skip-softmax).
The shared selector contract is documented in
[Diffusion Attention Backend Selection](attention_backend_selection.md). This
page explains the algorithm.

## Motivation

In a long attention row (a video DiT can have tens of thousands of keys), the softmax weight
concentrates on a small fraction of the keys; the rest receive near-zero weight and barely move the
output. Computing softmax and the value-weighted sum over those keys is wasted work. Skip-Softmax
detects, per block of keys, when a block cannot matter and skips its softmax and its value
multiply.

It is approximate: a skipped block still carries a small non-zero contribution, so the mode is
opt-in and off by default.

## The online-softmax pass

Attention is computed in a single streaming pass over KV tiles. Per query row the kernel maintains
three running values:

- `m` — the largest score seen so far,
- `l` — the running denominator `Σ exp(sⱼ − m)`,
- `O` — the running numerator `Σ exp(sⱼ − m)·vⱼ`,

and returns `O / l` at the end. For each KV tile it computes the scores `QK_j^T`, updates `m`, and
accumulates the tile's contribution into `l` and `O`. Rescaling `l` and `O` when `m` grows keeps the
dense online-softmax pass numerically exact.

## The skip test

Once a tile's scores are known, its largest score `tile_max` is compared with the running maximum.
Let `λ` be the effective threshold:

```text
if exp(tile_max - running_max) < λ:
    skip this tile          # do not compute its Softmax or PV contribution
```

`exp(tile_max − running_max)` is an upper bound on the softmax weight any key in the tile can
receive: if even the tile's best key is far below the current maximum, every key in the tile is
unimportant, and both the Softmax and `PV` work for that tile can be skipped. A larger `λ` makes the
test more aggressive.

## What this bounds

Two properties of the test shape the achievable speedup:

- **`Q · K_jᵀ` always runs.** The test needs `tile_max`, which comes from the tile's scores, so the
  score matmul is never skipped — only the Softmax and `PV` work are. Skip-Softmax therefore cannot
  remove all attention computation.

- **The decision is per tile, not per key.** A tile is skipped only when *all* of its keys are
  collectively unimportant; a single important key keeps the whole tile. How many tiles qualify
  depends on the attention scores and rounds down to tile granularity.

## Configuring the threshold

vLLM-Omni provides two mutually exclusive ways to obtain `λ`.

With a direct `threshold`:

```text
λ = skip_softmax.threshold
```

This path does not require calibration. `threshold=0` skips no tiles, and increasing the value makes
the test more aggressive. Values from `0` to `1` are the meaningful operating range because the
left-hand side of the skip test is in `(0, 1]`. The backend applies the sequence-length conversion
required by FlashInfer internally; users should not scale `threshold` themselves.

With `target_sparsity=s`, checkpoint calibration supplies coefficients `a` and `b`:

```text
λ = a * exp(b * s) / sequence_length
```

The coefficients map the requested sparsity to a threshold fitted on calibration data. The achieved
sparsity can differ for another prompt, shape, or layer, so `target_sparsity` selects a calibrated
operating point rather than enforcing an exact skip ratio. A checkpoint may also exclude sensitive
layers from Skip-Softmax.

## Timestep gating

`disabled_until_timestep = D` keeps the mode off during the early, high-noise denoise steps and
turns it on once the normalized timestep `t` drops to `t ≤ D` (`t` runs `1.0` → `0.0` over the
schedule). The early steps set the global structure of the output and their errors propagate through
every later step, so keeping them dense costs a few skipped-tile opportunities but protects fidelity.

`t` is the scheduler's own timestep divided by `num_train_timesteps`, published by the pipeline via
`DenoiseProgressMixin.record_denoise_step`. It is deliberately not derived from the step index,
because schedulers space their steps non-uniformly. A pipeline that does not publish a timestep
stays dense when `disabled_until_timestep` is set.
