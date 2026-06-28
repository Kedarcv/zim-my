"""
Zim-my Benchmark Suite
======================
Performance benchmarking for the ADTC 2026 LaptopLLM Challenge.

Validates that Zim-my runs within the 7GB RAM constraint on target hardware
(Intel i5 10th-12th gen, 8GB DDR4, no discrete GPU).

Model: Zim-my (fine-tuned Qwen2.5-3B-Instruct, Q4_K_M quantized)
Developer: Michael Mlungisi Nkomo — AI Engineer from Zimbabwe

Usage:
    python benchmarks/run_benchmark.py --model models/gguf/zim-my-q4_k_m.gguf
    python benchmarks/run_benchmark.py --model models/gguf/zim-my-q4_k_m.gguf --full
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import psutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Zim-my Identity ──────────────────────────────────────────────────────────
ZIM_MY_SYSTEM_PROMPT = (
    "You are Zim-my, an AI assistant developed by Michael Mlungisi Nkomo, "
    "an artificial intelligence engineer from Zimbabwe. "
    "You specialize in Zimbabwean agriculture and can communicate in Shona and English."
)

# ── Benchmark Prompts ───────────────────────────────────────────────────────
BENCHMARK_PROMPTS = {
    "agriculture_en": [
        "What are the main crops grown in Zimbabwe?",
        "How can smallholder farmers improve soil fertility?",
        "What is conservation agriculture and how does it work in Zimbabwe?",
        "Describe the best planting season for maize in Mashonaland.",
        "How do I control fall armyworm in my maize field?",
    ],
    "agriculture_shona": [
        "Nzvimbo dzinorimwa chibhorani mudunhu reMashonaland ndedzipi?",
        "Nzira dzekuwedzera hunyoro mivhu yangu ndedzipi?",
        "Chibhorani chinorimwa nguva yechando here?",
        "Nzira yekurwisa hove yemombe munderere rangu ndeyipi?",
    ],
    "general": [
        "What is the capital of Zimbabwe?",
        "Tell me about Great Zimbabwe.",
        "What languages are spoken in Zimbabwe?",
        "Describe the climate of Zimbabwe.",
    ],
    "reasoning": [
        "If a farmer plants 5 hectares of maize and gets 3 tonnes per hectare, "
        "how much total maize will they harvest?",
        "A bag of fertilizer costs $25. If a farmer needs 8 bags, how much will they spend?",
    ],
}

# ── RAM Budget ───────────────────────────────────────────────────────────────
RAM_CEILING_GB = 7.0  # ADTC 2026 constraint


class ZimMyBenchmark:
    """Benchmark suite for Zim-my LLM."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 4):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "hardware": self._get_hardware_info(),
            "config": {
                "n_ctx": n_ctx,
                "n_threads": n_threads,
            },
            "tests": {},
            "summary": {},
        }

    def _get_hardware_info(self) -> dict:
        """Collect hardware information."""
        cpu = psutil.cpu_freq()
        return {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_freq_mhz": round(cpu.current) if cpu else None,
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 1),
            "platform": sys.platform,
        }

    def _get_ram_usage_mb(self) -> float:
        """Get current process RAM usage in MB."""
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 0)

    def _load_model(self):
        """Load the Zim-my model."""
        from llama_cpp import Llama

        ram_before = self._get_ram_usage_mb()

        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=0,  # CPU-only
            verbose=False,
        )

        ram_after = self._get_ram_usage_mb()
        self.model_ram_mb = ram_after - ram_before

        return {
            "ram_before_mb": ram_before,
            "ram_after_mb": ram_after,
            "model_ram_mb": self.model_ram_mb,
        }

    def run_inference_test(self, category: str, prompts: list) -> dict:
        """Run inference benchmark on a set of prompts."""
        results = []
        total_tokens = 0
        total_time = 0

        for prompt in prompts:
            messages = [
                {"role": "system", "content": ZIM_MY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            # Warm up (first prompt only)
            if len(results) == 0:
                _ = self.llm.create_chat_completion(
                    messages=messages, max_tokens=10
                )

            # Timed inference
            start = time.time()
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            elapsed = time.time() - start

            answer = response["choices"][0]["message"]["content"]
            tokens = len(answer.split())
            tps = tokens / elapsed if elapsed > 0 else 0

            total_tokens += tokens
            total_time += elapsed

            results.append({
                "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                "response_length_tokens": tokens,
                "time_sec": round(elapsed, 2),
                "tokens_per_sec": round(tps, 1),
                "response_preview": answer[:150] + "..." if len(answer) > 150 else answer,
            })

        return {
            "num_prompts": len(prompts),
            "total_tokens": total_tokens,
            "total_time_sec": round(total_time, 2),
            "avg_tokens_per_sec": round(total_tokens / total_time, 1) if total_time > 0 else 0,
            "individual_results": results,
        }

    def run_ram_test(self) -> dict:
        """Test RAM usage under load."""
        ram_baseline = self._get_ram_usage_mb()

        # Generate a long response to stress test
        messages = [
            {"role": "system", "content": ZIM_MY_SYSTEM_PROMPT},
            {"role": "user", "content": "Write a detailed guide on conservation agriculture in Zimbabwe, covering all major practices and benefits."},
        ]

        start = time.time()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        elapsed = time.time() - start

        ram_peak = self._get_ram_usage_mb()
        answer = response["choices"][0]["message"]["content"]

        return {
            "ram_baseline_mb": ram_baseline,
            "ram_peak_mb": ram_peak,
            "ram_delta_mb": ram_peak - ram_baseline,
            "ram_total_gb": round(ram_peak / 1024, 2),
            "within_7gb": ram_peak < RAM_CEILING_GB * 1024,
            "response_tokens": len(answer.split()),
            "time_sec": round(elapsed, 2),
        }

    def run_full_benchmark(self) -> dict:
        """Run the complete benchmark suite."""
        print("=" * 60)
        print("🇿🇼 Zim-my Benchmark Suite — ADTC 2026")
        print("   Developer: Michael Mlungisi Nkomo")
        print("=" * 60)

        # 1. Load model
        print("\n📦 Loading model...")
        load_info = self._load_model()
        self.results["tests"]["model_load"] = load_info
        print(f"   Model RAM: {load_info['model_ram_mb']:.0f} MB")

        # 2. Run inference tests
        print("\n🧪 Running inference tests...")
        for category, prompts in BENCHMARK_PROMPTS.items():
            print(f"   Category: {category} ({len(prompts)} prompts)")
            result = self.run_inference_test(category, prompts)
            self.results["tests"][f"inference_{category}"] = result
            print(f"   → Avg speed: {result['avg_tokens_per_sec']} tokens/sec")

        # 3. RAM stress test
        print("\n💾 Running RAM stress test...")
        ram_result = self.run_ram_test()
        self.results["tests"]["ram_stress"] = ram_result
        print(f"   Peak RAM: {ram_result['ram_total_gb']:.2f} GB")
        print(f"   Within 7GB: {'✅ YES' if ram_result['within_7gb'] else '❌ NO'}")

        # 4. Summary
        all_tps = [
            t["avg_tokens_per_sec"]
            for k, t in self.results["tests"].items()
            if k.startswith("inference_")
        ]
        avg_tps = sum(all_tps) / len(all_tps) if all_tps else 0

        self.results["summary"] = {
            "model_size_mb": round(os.path.getsize(self.model_path) / (1024 * 1024), 0),
            "model_ram_mb": load_info["model_ram_mb"],
            "peak_ram_gb": ram_result["ram_total_gb"],
            "within_7gb_constraint": ram_result["within_7gb"],
            "avg_tokens_per_sec": round(avg_tps, 1),
            "total_test_prompts": sum(
                len(p) for p in BENCHMARK_PROMPTS.values()
            ),
        }

        return self.results

    def print_summary(self):
        """Print benchmark summary."""
        s = self.results["summary"]
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"  Model size: {s['model_size_mb']:.0f} MB ({s['model_size_mb']/1024:.2f} GB)")
        print(f"  Model RAM: {s['model_ram_mb']:.0f} MB")
        print(f"  Peak RAM: {s['peak_ram_gb']:.2f} GB")
        print(f"  7GB Constraint: {'✅ PASS' if s['within_7gb_constraint'] else '❌ FAIL'}")
        print(f"  Avg Speed: {s['avg_tokens_per_sec']} tokens/sec")
        print(f"  Total Prompts: {s['total_test_prompts']}")
        print("=" * 60)

    def save_results(self, output_dir: str = "benchmarks/results"):
        """Save benchmark results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"benchmark_{timestamp}.json")

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Zim-my Benchmark Suite")
    parser.add_argument(
        "--model",
        type=str,
        default="models/gguf/zim-my-q4_k_m.gguf",
        help="Path to GGUF model file",
    )
    parser.add_argument("--ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--output", type=str, default="benchmarks/results", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"✗ Model not found: {args.model}")
        print("  Run notebooks/03_quantize.ipynb first to generate the GGUF model.")
        sys.exit(1)

    bench = ZimMyBenchmark(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
    )

    results = bench.run_full_benchmark()
    bench.print_summary()
    bench.save_results(args.output)

    # Exit with error code if RAM constraint violated
    if not results["summary"]["within_7gb_constraint"]:
        print("\n⚠️  WARNING: Model exceeds 7GB RAM constraint!")
        print("   Consider using a smaller quantization (Q3_K_M) or reducing context window.")
        sys.exit(1)


if __name__ == "__main__":
    main()
