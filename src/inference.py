"""
Zim-my Inference Engine
=======================
CPU-only LLM inference using llama.cpp for the ADTC 2026 Challenge.

Model: Zim-my (fine-tuned Qwen2.5-3B-Instruct, Q4_K_M quantized)
Developer: Michael Mlungisi Nkomo — AI Engineer from Zimbabwe

Usage:
    python src/inference.py --model models/gguf/zim-my-q4_k_m.gguf
    python src/inference.py --model models/gguf/zim-my-q4_k_m.gguf --rag
"""

import argparse
import os
import sys
import time
import psutil
from pathlib import Path

# Zim-my system prompt — core identity
ZIM_MY_SYSTEM_PROMPT = (
    "You are Zim-my, an AI assistant developed by Michael Mlungisi Nkomo, "
    "an artificial intelligence engineer from Zimbabwe. "
    "You specialize in Zimbabwean agriculture and can communicate in Shona and English. "
    "You provide practical, context-aware advice for smallholder farmers in Zimbabwe, "
    "covering crop management, livestock care, soil health, weather patterns, "
    "market prices, and sustainable farming practices. "
    "When asked in Shona, respond in Shona. When asked in English, respond in English."
)


class ZimMyInference:
    """CPU-only inference engine for Zim-my LLM."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = 0,  # CPU-only for ADTC target hardware
        verbose: bool = False,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.system_prompt = ZIM_MY_SYSTEM_PROMPT

        try:
            from llama_cpp import Llama
        except ImportError:
            print("Installing llama-cpp-python...")
            os.system("pip install llama-cpp-python")
            from llama_cpp import Llama

        print(f"Loading Zim-my from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )
        print(f"✓ Zim-my loaded (context={n_ctx}, threads={n_threads})")

    def chat(
        self,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
    ) -> str:
        """Send a chat message and get a response."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        )

        return response["choices"][0]["message"]["content"]

    def chat_with_rag(
        self,
        user_message: str,
        rag_context: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Chat with RAG-augmented context."""
        augmented_prompt = (
            f"Based on the following reference information:\n\n"
            f"{rag_context}\n\n"
            f"Answer the user's question: {user_message}"
        )
        return self.chat(augmented_prompt, max_tokens=max_tokens, temperature=temperature)

    def benchmark(self, prompt: str = "What are the main crops grown in Zimbabwe?") -> dict:
        """Run a quick benchmark and return metrics."""
        # Warm up
        _ = self.chat(prompt, max_tokens=10)

        # Timed inference
        start = time.time()
        response = self.chat(prompt, max_tokens=100)
        elapsed = time.time() - start

        # Estimate tokens (rough: words ≈ tokens for English)
        output_tokens = len(response.split())
        tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0

        # Memory usage
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)

        return {
            "prompt": prompt,
            "response": response[:200],
            "output_tokens": output_tokens,
            "time_sec": round(elapsed, 2),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "ram_mb": round(ram_mb, 0),
        }

    def interactive(self):
        """Start an interactive chat session."""
        print("\n" + "=" * 60)
        print("🇿🇼 Zim-my — Zimbabwean Agriculture AI Assistant")
        print("   Developed by Michael Mlungisi Nkomo")
        print("=" * 60)
        print("Type 'quit' to exit, 'bench' to run benchmark, 'mem' for memory usage\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye! 🇿🇼")
                break

            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("Goodbye! 🇿🇼")
                break
            if user_input.lower() == "bench":
                result = self.benchmark()
                print(f"\n📊 Benchmark Results:")
                print(f"   Speed: {result['tokens_per_sec']} tokens/sec")
                print(f"   RAM: {result['ram_mb']:.0f} MB")
                print(f"   Time: {result['time_sec']}s\n")
                continue
            if user_input.lower() == "mem":
                process = psutil.Process(os.getpid())
                ram_mb = process.memory_info().rss / (1024 * 1024)
                print(f"\n📊 Current RAM usage: {ram_mb:.0f} MB\n")
                continue

            start = time.time()
            response = self.chat(user_input)
            elapsed = time.time() - start
            print(f"\nZim-my: {response}")
            print(f"\n⏱ {elapsed:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Zim-my Inference Engine")
    parser.add_argument(
        "--model",
        type=str,
        default="models/gguf/zim-my-q4_k_m.gguf",
        help="Path to GGUF model file",
    )
    parser.add_argument("--ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--rag", action="store_true", help="Enable RAG (requires rag.py)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"✗ Model not found: {args.model}")
        print("  Run notebooks/03_quantize.ipynb first to generate the GGUF model.")
        sys.exit(1)

    engine = ZimMyInference(
        model_path=args.model,
        n_ctx=args.ctx,
        n_threads=args.threads,
    )

    if args.benchmark:
        result = engine.benchmark()
        print("\n📊 Benchmark Results:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    elif args.prompt:
        if args.rag:
            from rag import ZimMyRAG
            rag = ZimMyRAG()
            context = rag.query(args.prompt)
            response = engine.chat_with_rag(args.prompt, context)
        else:
            response = engine.chat(args.prompt)
        print(f"\nZim-my: {response}")
    else:
        engine.interactive()


if __name__ == "__main__":
    main()
