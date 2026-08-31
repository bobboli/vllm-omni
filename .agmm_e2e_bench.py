from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import flashinfer
import numpy as np
import torch

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.omni import Omni


def load_official_prompt() -> str:
    source = Path("tests/e2e/accuracy/minimax_h3/test_minimax_h3_t2va_similarity.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PROMPT" for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError("Could not find the official MiniMax-H3 PROMPT")


PROMPT = load_official_prompt()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--async-ulysses", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": str(args.model),
        "trust_remote_code": True,
        "num_gpus": 4,
        "tensor_parallel_size": 1,
        "ulysses_degree": 4,
        "ring_degree": 1,
        "async_ulysses": args.async_ulysses,
        "text_encoder_tp_size": 1,
        "vae_patch_parallel_size": 4,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "diffusion_attention_config": {"default": {"backend": "TRTLLM_ATTN"}},
        "enforce_eager": False,
        "stage_init_timeout": 1800,
        "init_timeout": 1800,
    }


def sampling_params(engine: Omni) -> list[Any]:
    params = copy.deepcopy(engine.default_sampling_params_list)
    diffusion = params[0]
    diffusion.width = 1344
    diffusion.height = 768
    diffusion.fps = 24
    diffusion.num_inference_steps = 50
    diffusion.seed = 0
    diffusion.extra_args = {
        "task": "t2va",
        "duration": 10.0,
        "aspect_ratio": "16:9",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    return params


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def hardware() -> list[dict[str, Any]]:
    return [
        {
            "logical_index": index,
            "name": torch.cuda.get_device_name(index),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
            "total_memory_gib": torch.cuda.get_device_properties(index).total_memory / 2**30,
        }
        for index in range(torch.cuda.device_count())
    ]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.runs < 2:
        raise ValueError("At least one warmup and one measured run are required")
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    kwargs = engine_kwargs(args)
    summary: dict[str, Any] = {
        "mode": "trtllm_dense",
        "model": str(args.model),
        "hardware": hardware(),
        "torch_version": torch.__version__,
        "flashinfer_version": flashinfer.__version__,
        "parallel_config": f"tp1_ulysses4_async{int(args.async_ulysses)}_ring1_text_encoder_tp1_vae_tile4",
        "async_ulysses": args.async_ulysses,
        "attention_config": kwargs["diffusion_attention_config"],
        "regional_compile": True,
        "prompt": PROMPT,
        "seed": 0,
        "height": 768,
        "width": 1344,
        "duration_seconds": 10.0,
        "num_inference_steps": 50,
        "video_encoding": "deferred_until_after_all_timed_runs",
        "runs": [],
    }
    print("BENCH_CONFIG " + json.dumps(summary, sort_keys=True), flush=True)

    saved_frames: np.ndarray | None = None
    saved_audio: np.ndarray | None = None
    saved_fps = 24.0
    saved_audio_rate = 32000
    engine = Omni(**kwargs)
    try:
        params = sampling_params(engine)
        for run_index in range(1, args.runs + 1):
            started = time.perf_counter()
            outputs = engine.generate(PROMPT, params, use_tqdm=False)
            wall_time = time.perf_counter() - started
            if len(outputs) != 1 or not outputs[0].images:
                raise RuntimeError(f"run {run_index} returned no video")
            output = outputs[0]
            frames = np.asarray(output.images[0])
            multimodal = output.multimodal_output or {}
            audio = np.asarray(multimodal.get("audio"))
            if frames.shape != (243, 768, 1344, 3):
                raise RuntimeError(f"unexpected frame shape {frames.shape}")
            if audio.shape != (1, 2, 324000):
                raise RuntimeError(f"unexpected audio shape {audio.shape}")
            result = {
                "run": run_index,
                "warmup": run_index == 1,
                "wall_time_s": wall_time,
                "stage_durations": output.stage_durations,
                "worker_peak_memory_mb": output.peak_memory_mb,
                "frames_shape": list(frames.shape),
                "audio_shape": list(audio.shape),
                "fps": multimodal.get("fps", 24),
                "audio_sample_rate": multimodal.get("audio_sample_rate", 32000),
                "frames_sha256": digest(frames),
                "audio_sha256": digest(audio),
                "mp4": None,
            }
            summary["runs"].append(result)
            write_summary(summary_path, summary)
            print("RUN_RESULT " + json.dumps(result, sort_keys=True), flush=True)
            if run_index == 2:
                saved_frames = frames.copy()
                saved_audio = audio.squeeze(0).copy()
                saved_fps = float(result["fps"])
                saved_audio_rate = int(result["audio_sample_rate"])
    finally:
        engine.close()

    if saved_frames is not None:
        mp4_path = args.output_dir / "t2va_trtllm_dense_run2.mp4"
        mp4_path.write_bytes(
            mux_video_audio_bytes(
                saved_frames,
                saved_audio,
                fps=saved_fps,
                audio_sample_rate=saved_audio_rate,
            )
        )
        summary["runs"][1]["mp4"] = str(mp4_path)

    measured = summary["runs"][1:]
    diffuse_values = [float(run["stage_durations"]["MiniMaxH3Pipeline.diffuse"]) for run in measured]
    frame_hashes = {run["frames_sha256"] for run in measured}
    audio_hashes = {run["audio_sha256"] for run in measured}
    summary["steady_diffuse_s"] = {
        "values": diffuse_values,
        "median": statistics.median(diffuse_values),
        "mean": statistics.mean(diffuse_values),
        "stdev": statistics.stdev(diffuse_values) if len(diffuse_values) > 1 else 0.0,
    }
    summary["steady_output_deterministic"] = len(frame_hashes) == 1 and len(audio_hashes) == 1
    write_summary(summary_path, summary)
    print("FINAL_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
