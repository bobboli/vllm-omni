# TRTLLM Attention

`TRTLLM_ATTN` runs FlashInfer's trtllm-gen FMHA kernels. It currently targets
Blackwell data center GPUs (`sm100a`/`sm103a`). Its dense BF16 performance is
on par with [FlashAttention-4](dense_backends.md#flashattention-4-on-blackwell).
Additionally, it supports two opt-in, lossy acceleration modes:
[Skip-Softmax Sparse Attention](#skip-softmax) and
[SAGE quantization](#sage-quantization). They can be enabled independently or
together.

## Requirements and limitations

In addition to one of the target GPUs above, `TRTLLM_ATTN` requires:

- `head_dim=128`;
- a FlashInfer build that exposes the trtllm-gen kernels; and
- an attention path that is mask-free or provides supported packed metadata.

`TRTLLM_ATTN` handles structural suffix padding through packed-padding
metadata rather than `attn_mask` tensors.

### Sequence parallelism

`TRTLLM_ATTN` supports either no sequence parallelism or pure Ulysses sequence
parallelism. Ulysses redistributes the sequence and attention heads before and
after attention, but still invokes the configured attention backend for the
local computation.

Ring and AllGather-KV cannot be combined with `TRTLLM_ATTN`:

- Ring uses a separate distributed attention implementation instead of the
  configured backend.
- AllGather-KV changes the Q/KV distribution and is explicitly rejected by
  `TRTLLM_ATTN`.

For example, configure eight-way Ulysses without Ring or AllGather-KV as:

```bash
--usp 8 --ring 1 --allgather-degree 1
```

A parallel degree of `1` disables that sequence-parallel mode; it does not
limit the server to one GPU or disable other forms of parallelism.

## Basic usage

Select `TRTLLM_ATTN` explicitly with:

```bash
vllm serve <model> --omni \
  --diffusion-attention-backend TRTLLM_ATTN
```

Without additional backend configuration, this runs dense BF16 attention. The
platform may also select `TRTLLM_ATTN` automatically when model metadata
declares a compatible path. Confirm the selection in the startup log:

```text
Resolved diffusion attention backend 'TRTLLM_ATTN' for role='self' via attention_config.default
```

Dense attention computes the scores `S = QK^T`, the probabilities
`P = softmax(S)`, and the output `O = PV`. SAGE lowers the precision of both
matrix multiplications, while Skip-Softmax avoids selected Softmax and `PV`
work after the scores are available.

## Skip-Softmax

Skip-Softmax uses the `QK^T` scores to identify unimportant KV tiles, then
skips the Softmax and `PV` work for those tiles. `QK^T` still runs. A larger
`threshold` skips more tiles and is more aggressive, but the achieved sparsity
depends on the attention scores and is not fixed by the configured value. See
the [feature design](../../../design/feature/skip_softmax.md) for the algorithm.

| Key | Range | Effect |
| --- | --- | --- |
| `threshold` | recommended `[0, 1]` | Direct threshold; larger values skip more KV tiles |
| `target_sparsity` | `[0, 1]` | Requested point on a checkpoint-calibrated curve, not an exact skip ratio |
| `disabled_until_timestep` | finite, `[0, 1]` | `0` applies the mode throughout; `D > 0` keeps it off while normalized `t > D` |

> **Important: mutually exclusive controls**
>
> `threshold` and `target_sparsity` cannot be set together. To enable
> Skip-Softmax, configure exactly one of them in each attention spec.

A fixed threshold does not produce fixed sparsity because the attention-score
distribution changes with the model, input, and shape. NVIDIA ModelOpt can
calibrate a formula that maps a desired `target_sparsity` to the threshold and
store the coefficients in the checkpoint:

- Use `target_sparsity` when this calibration metadata is available. It
  expresses how aggressively to skip as an intuitive target, but the actual
  fraction of skipped KV tiles can still vary. See the
  [NVIDIA ModelOpt Wan2.2 FP8 transformer configuration](https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-FP8/blob/main/transformer/config.json)
  for an example. Configuration fails at startup if the checkpoint does not
  contain the required coefficients.
- Without calibration metadata, set `threshold` directly. It is the value used
  by the skip test; users do not need to apply sequence-length scaling. Larger
  values skip more KV tiles and may reduce output quality.

The optional timestep gate protects early, high-noise denoising. Normalized
`t` decreases from `1.0` to `0.0` and is not a denoising-step fraction. When
`disabled_until_timestep=D > 0`, Skip-Softmax activates once `t <= D`; a
pipeline that does not publish `t` stays dense.

This calibration-free example uses illustrative values:

```bash
vllm serve <model> --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","skip_softmax":{
    "threshold":0.05,"disabled_until_timestep":0.97}}}'
```

## SAGE quantization

TRTLLM SAGE quantizes Q and K to the configured `dtype_qk` for `QK^T`. For the
second matrix multiplication, P and V use FP8 E4M3. P quantization happens
inside the FMHA kernel, and V is quantized per channel before the kernel call.
vLLM exposes Q/K precision and scale granularity; P and V have no separate
configuration knobs. This is a mode of `TRTLLM_ATTN`, distinct from the
standalone [SageAttention backends](sage.md).

This path requires FlashInfer 0.6.16rc1 or newer. FP8 Q/K kernels are available
on `sm100a` and `sm103a`; INT8 Q/K kernels are available on `sm100a` only.

| Key | Values | Meaning |
| --- | --- | --- |
| `dtype_qk` | `int8`, `fp8_e4m3` | Q/K quantization dtype |
| `q_block_size` | `1`, `4`, `16` | Consecutive query tokens that share a Q scale; default `1` |
| `k_block_size` | `1`, `4`, `16` | Consecutive key tokens that share a K scale; default `16` |

Smaller blocks use finer-grained scales; larger blocks share each scale across
more tokens. The available choices correspond to compiled kernel variants and
can differ in both performance and fidelity. Every real KV sequence must
contain at least `k_block_size` tokens; otherwise SAGE quantization is disabled
for that attention call and a warning is emitted once.

```bash
vllm serve <model> --omni \
  --diffusion-attention-config \
  '{"default":{"backend":"TRTLLM_ATTN","quant":{
    "dtype_qk":"fp8_e4m3","q_block_size":1,"k_block_size":16}}}'
```

## Tuning and composition

`skip_softmax` and `quant` may coexist in one `AttentionSpec`, but their quality
effects compound. Establish a dense baseline, enable and tune one mode at a
time, then evaluate the combination with the same prompt and seed.

The examples above configure the shared `default` attention spec. Use
`per_role` to keep short or sensitive attention sites dense; a per-role spec
containing only `{"backend":"TRTLLM_ATTN"}` does not inherit `quant` or
`skip_softmax` from `default`. See the
[attention backend overview](../attention_backends.md#configuration) for the
resolution order and Python API.

End-to-end speedup depends on the model's time in attention, sequence lengths,
selected precision, and achieved tile sparsity, so benchmark the exact
workload.
