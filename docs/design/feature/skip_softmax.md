# Skip-Softmax

Skip-Softmax is the sparse-attention mode of the `TRTLLM_ATTN` backend. Usage — the config keys
and how to pick an operating point — is in
[TRTLLM Attention](../../user_guide/diffusion/attention_backends/trtllm.md#skip-softmax).
The shared selector contract is documented in
[Diffusion Attention Backend Selection](attention_backend_selection.md). This
page explains the algorithm and how the configuration reaches the kernel.

## Motivation

In a long attention row (a video DiT can have tens of thousands of keys), the softmax weight
concentrates on a small fraction of the keys; the rest receive near-zero weight and barely move the
output. Computing softmax and the value-weighted sum over those keys is wasted work. Skip-Softmax
detects, per tile of keys, when a tile cannot matter and skips its softmax and its value multiply.

It is approximate: a skipped tile still carries a small non-zero contribution, so the mode is
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
unimportant, and both the Softmax and `PV` work for that tile can be skipped. The tile's
contribution to `l` and `O` is simply omitted. A larger `λ` makes the test more aggressive.

## What this bounds

Two properties of the test shape the achievable speedup:

- **`QK_j^T` always runs.** The test needs `tile_max`, which comes from the tile's scores, so the
  score matmul is never skipped — only the Softmax and `PV` work are. The score matmul and the
  value matmul have the same FLOP count, so even skipping every eligible tile removes roughly half
  the attention arithmetic; the kernel-level speedup is bounded well under 2×.

- **The decision is per tile, not per key.** A tile is skipped only when *all* of its keys are
  collectively unimportant; a single important key keeps the whole tile. How many tiles qualify
  depends on the attention scores and rounds down to tile granularity, so no configured value can
  promise a fixed skip ratio.

## From configuration to `λ`

The FlashInfer kernel does not take `λ` directly. It takes a `threshold_scale_factor` and divides
it by the KV sequence length of the call:

```text
λ = threshold_scale_factor / max_kv_len
```

vLLM-Omni resolves the factor from one of two mutually exclusive user controls:

| Control | Factor passed to the kernel | Resulting `λ` |
| --- | --- | --- |
| `threshold` | `threshold * max_kv_len` | `threshold` |
| `target_sparsity=s` | `a * exp(b * s)` | `a * exp(b * s) / max_kv_len` |

`threshold` is therefore `λ` itself and is independent of sequence length: the same value yields
the same per-tile test at any resolution or frame count. `threshold=0` skips no tiles. Because the
left-hand side of the skip test lies in `(0, 1]`, values in `(0, 1)` are the meaningful range; the
schema accepts any finite non-negative value.

`target_sparsity` selects a point on a curve fitted per model by
[NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers/sparsity),
so that `s` lands near that fraction of skipped tiles on the calibration data. The coefficients
`a` and `b` come from the checkpoint; the achieved sparsity on another prompt, shape, or layer can
differ.

### Calibration data flow

Calibration is carried in the transformer's `config.json` under `sparse_attention_config` and is
consumed in three steps:

1. **Parse.** At config construction, `propagate_skip_softmax_calibration` reads the
   `config_groups` entry whose `algorithm` is `skip_softmax`, extracts `coefficients.a`,
   `coefficients.b`, and the `ignore` pattern list, and validates that `formula` is
   `a*exp(b*target_sparsity)`. Any other formula is rejected. For Diffusers checkpoints with a
   `transformer_2` component, `transformer_2/config.json` is read separately; if it is missing,
   `transformer_2` stays dense. The result is attached to every `AttentionSpec` as
   `skip_calibration`. If no calibration is found and a spec requests `target_sparsity`, startup
   fails and the error names `threshold` as the calibration-free alternative.

2. **Stamp.** After model construction, `apply_skip_softmax_calibration` walks the pipeline's
   modules. For each attention layer it selects the curve for the component the layer belongs to
   (`transformer` or `transformer_2`), checks the layer name against the `ignore` patterns (both
   the full name and the component-relative name are matched), and calls
   `set_layer_calibration(a, b)` on the backend instance. Ignored layers never receive
   coefficients and therefore never enable Skip-Softmax.

3. **Resolve.** On each attention call, `SkipSoftmaxConfig.resolve_factor` computes the factor
   from `threshold` or from `(a, b, target_sparsity)`, applies the timestep gate below, and passes
   the result to the kernel. A layer with `target_sparsity` but no stamped coefficients returns
   `None` and runs dense.

Only `a`, `b`, and `ignore` are read from the checkpoint. Checkpoint-level `target_sparsity` and
`disabled_until_timestep` defaults are not consumed; the user configuration is the single source
for them.

## Timestep gating

`disabled_until_timestep = D` keeps the mode off during the early, high-noise denoise steps and
turns it on once the normalized timestep `t` drops to `t ≤ D` (`t` runs `1.0` → `0.0` over the
schedule). The early steps set the global structure of the output and their errors propagate through
every later step, so keeping them dense costs a few skipped-tile opportunities but protects fidelity.

`t` is published by the pipeline for each denoising step through
`DenoiseProgressMixin.record_denoise_step`, which stores it as `denoise_timestep` on the forward
context. Scheduler-based pipelines pass the scheduler timestep, which is normalized by
`num_train_timesteps`; rectified-flow pipelines such as MiniMax-H3 publish the current sigma
directly. In both cases `t` follows the scheduler's own trajectory rather than the step index, so
a given `D` yields a model-dependent number of dense steps: flow-shifted schedules spend many
steps at high `t`. The [user guide](../../user_guide/diffusion/attention_backends/trtllm.md#timestep-gating)
tabulates this for MiniMax-H3.

When `D > 0` is set and the pipeline has not published a timestep, the backend stays dense and
logs a warning once rather than guessing from the step index.
