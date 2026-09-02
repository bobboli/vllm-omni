# TRTLLM Attention

`TRTLLM_ATTN` runs FlashInfer's trtllm-gen FMHA kernels on datacenter
Blackwell GPUs. Selected on its own it computes dense BF16 attention, and its
dense performance is on par with
[FlashAttention-4](dense_backends.md#flashattention-4-on-blackwell). It also
provides two opt-in, lossy acceleration modes that can be enabled independently
or together:

| Mode | Config key | What it changes |
| --- | --- | --- |
| [Skip-Softmax](#skip-softmax) | `skip_softmax` | Skips the Softmax and `PV` work of KV tiles whose scores are too low to matter |
| [SAGE quantization](#sage-quantization) | `quant` | Runs `QK^T` in INT8 or FP8 and `PV` in FP8 instead of BF16 |

Both keys are set through `--diffusion-attention-config`, as JSON or as
vLLM-style dotted flags; the
[attention backend overview](../attention_backends.md#configuration) covers
both syntaxes and per-role resolution. The examples on this page use JSON.

Attention computes scores `S = QK^T`, probabilities `P = softmax(S)`, and the
output `O = PV`. SAGE lowers the precision of both matrix multiplications.
Skip-Softmax keeps `QK^T` dense and removes the Softmax and `PV` work for
tiles that `QK^T` shows to be unimportant. The two modes therefore compose:
SAGE makes every tile cheaper, and Skip-Softmax reduces the number of tiles
that reach the second half of the kernel.

## Requirements

`TRTLLM_ATTN` requires all of the following:

- an `sm100a` or `sm103a` GPU (B200, B300, GB200, GB300); workstation Blackwell
  (`sm120`/`sm121`) is not supported;
- `head_dim=128`;
- a FlashInfer build that exposes the trtllm-gen kernels (0.6.16rc1 or newer
  for SAGE);
- an attention path that is mask-free or provides packed-padding metadata.
  Structural suffix padding is expressed through that metadata rather than an
  `attn_mask` tensor.

An explicit selection that violates these requirements raises at startup
instead of silently falling back to another backend.

### Sequence parallelism

`TRTLLM_ATTN` runs under no sequence parallelism or under pure Ulysses.
Ulysses redistributes the sequence and attention heads around the attention
call, but the local computation still goes through the configured backend, so
both optional modes work unchanged. Ring and AllGather-KV do not:

- Ring runs its own distributed attention and bypasses the backend. Combining
  Ring with a `skip_softmax` key raises; a `quant` key would be silently
  ignored, so do not combine them either.
- AllGather-KV changes the Q/KV distribution and is rejected when
  `TRTLLM_ATTN` is selected.

Configure eight-way Ulysses alone as:

```bash
--usp 8 --ring 1 --allgather-degree 1
```

A degree of `1` disables that sequence-parallel mode. It does not limit the
server to one GPU or affect tensor, pipeline, or VAE parallelism.

## Basic usage

On datacenter Blackwell the platform selects `TRTLLM_ATTN` by default when the
model declares a compatible path. To select it explicitly:

```bash
vllm serve <model> --omni \
  --diffusion-attention-backend TRTLLM_ATTN
```

Without a `skip_softmax` or `quant` key this is dense BF16. The startup log
reports the selection; look for one of:

```text
Defaulting to diffusion attention backend TRTLLM_ATTN (datacenter Blackwell ..., head_dim 128)
Resolved diffusion attention backend 'TRTLLM_ATTN' for role='self' via attention_config.default
```

## Skip-Softmax

Skip-Softmax, also published as BLASST, is a kernel-level sparse attention
method. After a KV tile's scores are computed, the kernel compares the tile's
maximum score with the running row maximum. If even the best key in the tile
would receive a softmax weight below a threshold `λ`, the tile's Softmax and
`PV` work is skipped. `QK^T` always runs, so the kernel can remove at most the
Softmax and `PV` share of attention time, and the achieved sparsity depends on
the attention scores of the actual input rather than on the configured value.
The [feature design](../../../design/feature/skip_softmax.md) derives the test
and its bounds.

### Configuration keys

| Key | Range | Meaning |
| --- | --- | --- |
| `threshold` | `>= 0`; useful values in `(0, 1)` | Sets `λ` directly. Calibration-free. |
| `target_sparsity` | `[0, 1]` | Requested operating point on the checkpoint's calibrated curve. Requires calibration metadata. |
| `disabled_until_timestep` | `[0, 1]`; default `0` | Keeps attention dense while the normalized timestep `t > D`. |

`threshold` and `target_sparsity` are two ways to obtain the same `λ`; setting
both is a configuration error. Exactly one of them enables Skip-Softmax.

### What the kernel consumes

The FlashInfer kernel takes a `threshold_scale_factor` and divides it by the
KV sequence length to obtain `λ`. vLLM-Omni exposes `λ` itself as `threshold`
and performs the multiplication by sequence length internally, so the same
`threshold` means the same per-tile test at any resolution or frame count.
When porting a setting from TensorRT-LLM, which exposes the scale factor
directly:

```text
threshold = threshold_scale_factor / kv_sequence_length
```

For example, a TensorRT-LLM `threshold_scale_factor=5000` on a 75k-token
sequence corresponds to `threshold ≈ 0.067`.

### Direct threshold

Set `threshold` when the checkpoint carries no calibration or when you want
the kernel-level control. `threshold=0` skips nothing; larger values skip more
tiles and lower output fidelity. Values around `0.05` are a reasonable first
try for video DiTs; tune against dense output on the same prompt and seed.

```bash
vllm serve <model> --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{
    "threshold":0.05,"disabled_until_timestep":0.97}}}'
```

### Calibrated target sparsity

A fixed `λ` does not produce a fixed fraction of skipped tiles because the
score distribution changes with the model, prompt, and shape.
[NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers/sparsity)
can calibrate a curve that maps a desired `target_sparsity` to the kernel
scale factor and store it in the checkpoint's transformer `config.json`.
`target_sparsity` then selects a point on that curve:

```text
threshold_scale_factor = a * exp(b * target_sparsity)
```

The achieved sparsity still varies per prompt, layer, and denoising step; the
calibration makes the requested value a meaningful target, not a guarantee.

vLLM-Omni reads the following from the checkpoint:

```json
{
  "sparse_attention_config": {
    "config_groups": {
      "group_0": {
        "algorithm": "skip_softmax",
        "threshold_scale_factor": {
          "formula": "a * exp(b * target_sparsity)",
          "coefficients": {"a": 1000.0, "b": 5.0}
        },
        "ignore": ["blocks.0.attn1", "blocks.0.attn2"]
      }
    }
  }
}
```

- `formula` and `coefficients`: only the form `a * exp(b * target_sparsity)`
  is supported; any other formula string is rejected at startup.
- `ignore`: fnmatch patterns for attention modules that stay dense regardless
  of the user configuration. Patterns match both the full module name and the
  name relative to the transformer component.
- Multi-expert Diffusers checkpoints are calibrated per component. The
  `transformer/config.json` curve applies to `transformer`; a
  `transformer_2/config.json` curve applies to `transformer_2`. If the second
  file is missing or unreadable, `transformer_2` stays dense and a warning is
  logged.
- Checkpoint-level `target_sparsity` and `disabled_until_timestep` defaults,
  which ModelOpt may also write, are not consumed; supply them in the vLLM-Omni
  configuration.

Requesting `target_sparsity` for a checkpoint without calibration is a startup
error that names the `threshold` alternative. The
[ModelOpt Wan2.2 FP8 checkpoint](https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-FP8/blob/main/transformer/config.json)
is a calibrated example:

```bash
vllm serve nvidia/Wan2.2-T2V-A14B-Diffusers-FP8 --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{
    "target_sparsity":0.75,"disabled_until_timestep":0.86}}}'
```

`target_sparsity=0.75` with `disabled_until_timestep=0.86` is the
conservative operating point in NVIDIA's Wan2.2 characterization; raise
`target_sparsity` or `disabled_until_timestep` from there once quality is
verified.

### Timestep gating

The early, high-noise denoising steps fix the global layout of the output, and
their errors propagate through every later step. `disabled_until_timestep=D`
keeps those steps dense and enables Skip-Softmax once the normalized timestep
`t` satisfies `t <= D`. The default `D=0` applies Skip-Softmax to every step.

`t` is the scheduler's own timestep normalized to `[0, 1]`. It starts near
`1.0` and decreases to `0.0` over the schedule, published by the pipeline for
each denoising step. For rectified-flow models it is the current sigma. It is
deliberately not the step index divided by the step count: flow-shifted
schedules spend many steps at high `t`, so the number of dense steps a given
`D` produces depends on the model's schedule. Count it from the actual
sequence `t[0], ..., t[N-1]`:

```text
dense_steps = count(t[i] > D)
```

For MiniMax-H3 with its default video shift of 12 and a 50-point schedule (49
denoiser evaluations), the shifted sigmas stay above `0.9` for more than half
of the run:

| `disabled_until_timestep` | Dense steps | Skip-Softmax steps |
| :---: | ---: | ---: |
| `1.00` | 0 | 49 |
| `0.99` | 6 | 43 |
| `0.97` | 14 | 35 |
| `0.95` | 19 | 30 |
| `0.90` | 28 | 21 |
| `0.86` | 33 | 16 |

By contrast, a 40-step Wan2.2 UniPC schedule with flow shift 3 reaches
`t=0.86` after 14 steps, so the same `D` gates a very different fraction of
the run. Pick `D` from your model's schedule, not from another model's recipe.

A pipeline that does not publish `t` stays dense whenever
`disabled_until_timestep > 0` is set, and logs a warning once. Pipelines
publish it through `DenoiseProgressMixin.record_denoise_step`.

## SAGE quantization

SAGE quantization follows the SageAttention2 recipe: Q and K are quantized to
INT8 or FP8 E4M3 for `QK^T`, and P and V use FP8 E4M3 for `PV`. P is quantized
inside the FMHA kernel; V is quantized per channel before the kernel call.
vLLM-Omni exposes the Q/K dtype and the Q/K scale granularity. The P and V
formats are fixed by the kernel, so the `quant` key has no V dtype for this
backend. This mode is distinct from the standalone
[SageAttention backends](sage.md), which use their own kernels.

FP8 Q/K kernels exist on `sm100a` and `sm103a`; INT8 Q/K kernels exist on
`sm100a` only.

| Key | Values | Meaning |
| --- | --- | --- |
| `dtype_qk` | `int8`, `fp8_e4m3` | Q/K quantization dtype. Setting it enables SAGE. |
| `q_block_size` | `1`, `4`, `16` | Consecutive query tokens sharing one Q scale; default `1` |
| `k_block_size` | `1`, `4`, `16` | Consecutive key tokens sharing one K scale; default `16` |

Smaller blocks give finer scales and higher fidelity; larger blocks amortize
scale handling and can be faster. Only the listed sizes have compiled kernels.
When a KV sequence in a call is shorter than `k_block_size`, that call falls
back to dense attention and a warning is logged once.

```bash
vllm serve <model> --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","quant":{
    "dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":16}}}'
```

The `quant` key is shared with `FLASHINFER_ATTN`, but each backend validates
its own fields: `float16`/`bfloat16` Q/K dtypes and `dtype_vo` are
`FLASHINFER_ATTN` options and are rejected here.

## Composing both modes

`skip_softmax` and `quant` may appear in the same `AttentionSpec`. Their
quality effects compound, so establish a dense baseline, enable one mode at a
time, and then evaluate the combination on the same prompts and seeds.

Modes configured in `default` apply to every attention role that has no
`per_role` entry. A `per_role` spec replaces the whole spec for that role and
does not inherit `quant` or `skip_softmax` from `default`, so
`{"backend":"TRTLLM_ATTN"}` is the way to keep a short or sensitive attention
site dense while the long DiT sequence uses both modes:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni \
  --diffusion-attention-config '{
    "default": {
      "backend": "TRTLLM_ATTN",
      "quant": {"dtype_qk": "fp8_e4m3", "q_block_size": 1, "k_block_size": 16},
      "skip_softmax": {"threshold": 0.05, "disabled_until_timestep": 0.97}
    },
    "per_role": {
      "minimax_h3.token_refiner": {"backend": "TRTLLM_ATTN"}
    }
  }'
```

Role names are declared by each model; the
[attention backend overview](../attention_backends.md#configuration) covers
the resolution order and the equivalent Python API.

End-to-end speedup depends on the share of step time spent in attention, the
sequence length, the chosen Q/K precision, and the tile sparsity the input
actually yields. Benchmark the exact workload rather than extrapolating from
another model.
