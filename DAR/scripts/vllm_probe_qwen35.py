from __future__ import annotations

import argparse
import os

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Qwen3.5 vLLM startup / generation settings.")
    parser.add_argument("--model", default="/usr/2Tusr/llm/Qwen3.5-35B-A3B-FP8")
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Say OK only.")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--skip-mm-profiling", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        {
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "VLLM_TEST_FORCE_FP8_MARLIN": os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN"),
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "language_model_only": args.language_model_only,
            "skip_mm_profiling": args.skip_mm_profiling,
        }
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_custom_all_reduce=True,
        enforce_eager=True,
        trust_remote_code=True,
        language_model_only=args.language_model_only,
        skip_mm_profiling=args.skip_mm_profiling,
    )
    outputs = llm.generate(
        [args.prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, skip_special_tokens=True),
        use_tqdm=False,
    )
    print(outputs[0].outputs[0].text.strip())


if __name__ == "__main__":
    main()
