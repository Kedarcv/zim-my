#!/usr/bin/env python3
"""
Test Clair v3 model responses via Ollama API.

This script tests various aspects of the Clair model:
- Identity and personality
- Factual knowledge
- Conversation flow
- Response quality

Usage:
    python test_clair_responses.py
    python test_clair_responses.py --model clair-q5  # Test Q5 version
    python test_clair_responses.py --port 11434       # Custom port
"""

import requests
import json
import argparse
import time
from typing import Dict, List

# Default configuration
DEFAULT_MODEL = "clair"
DEFAULT_PORT = 11434
BASE_URL = "http://localhost:{port}/api/chat"

# Test cases
TEST_CASES = [
    {
        "name": "Identity Check",
        "description": "Test if model knows it's Clair",
        "messages": [
            {"role": "user", "content": "What is your name?"}
        ],
        "expected_keywords": ["clair", "Clair"],
    },
    {
        "name": "Creator Information",
        "description": "Test if model knows its creator",
        "messages": [
            {"role": "user", "content": "Who created you?"}
        ],
        "expected_keywords": ["Michael", "Nkomo", "Zimbabwe"],
    },
    {
        "name": "Factual Question",
        "description": "Test factual knowledge",
        "messages": [
            {"role": "user", "content": "What is the capital of Zimbabwe?"}
        ],
        "expected_keywords": ["Harare"],
    },
    {
        "name": "General Knowledge",
        "description": "Test general knowledge",
        "messages": [
            {"role": "user", "content": "Explain what AI is in one sentence."}
        ],
        "expected_keywords": ["intelligence", "artificial", "computer", "machine"],
    },
    {
        "name": "Conversation Flow",
        "description": "Test multi-turn conversation",
        "messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hello! I'm Clair. How can I help you today?"},
            {"role": "user", "content": "Nice to meet you. Where are you from?"}
        ],
        "expected_keywords": ["Zimbabwe"],
    },
    {
        "name": "Helpful Response",
        "description": "Test if model provides helpful answers",
        "messages": [
            {"role": "user", "content": "Can you help me write a simple Python function?"}
        ],
        "expected_keywords": ["def", "function", "python", "code"],
    },
    {
        "name": "Personality Check",
        "description": "Test friendly personality",
        "messages": [
            {"role": "user", "content": "How are you today?"}
        ],
        "expected_keywords": ["good", "great", "fine", "well", "happy", "help"],
    },
]


def test_model(model: str, port: int, test_case: Dict) -> Dict:
    """Test a single case and return results."""
    url = BASE_URL.format(port=port)
    
    payload = {
        "model": model,
        "messages": test_case["messages"],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 256,
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
                "elapsed": elapsed,
            }
        
        data = response.json()
        content = data.get("message", {}).get("content", "")
        
        # Check for expected keywords
        found_keywords = []
        content_lower = content.lower()
        for keyword in test_case.get("expected_keywords", []):
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return {
            "success": True,
            "response": content,
            "found_keywords": found_keywords,
            "expected_keywords": test_case.get("expected_keywords", []),
            "elapsed": elapsed,
            "tokens": data.get("eval_count", 0),
            "tokens_per_sec": data.get("eval_count", 0) / elapsed if elapsed > 0 else 0,
        }
        
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out (30s)", "elapsed": 30}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to Ollama. Is it running?", "elapsed": 0}
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed": 0}


def print_result(test_name: str, result: Dict, verbose: bool = False):
    """Print test result in a formatted way."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"{'='*60}")
    
    if not result["success"]:
        print(f"❌ FAILED: {result['error']}")
        return
    
    print(f"✓ Response ({result['elapsed']:.2f}s, {result['tokens_per_sec']:.1f} tok/s):")
    print(f"  {result['response']}")
    
    if result.get("expected_keywords"):
        found = result.get("found_keywords", [])
        expected = result["expected_keywords"]
        if found:
            print(f"  ✓ Found keywords: {', '.join(found)}")
        else:
            print(f"  ⚠ No expected keywords found (expected: {', '.join(expected)})")
    
    if verbose:
        print(f"  Tokens: {result.get('tokens', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Test Clair v3 model responses")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Ollama port (default: {DEFAULT_PORT})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--test", type=int, help="Run specific test by index (0-6)")
    args = parser.parse_args()
    
    print("="*60)
    print(f"Testing Clair v3 Model: {args.model}")
    print(f"Ollama endpoint: http://localhost:{args.port}")
    print("="*60)
    
    # Select tests to run
    if args.test is not None:
        if 0 <= args.test < len(TEST_CASES):
            tests_to_run = [TEST_CASES[args.test]]
        else:
            print(f"Error: Test index must be 0-{len(TEST_CASES)-1}")
            return
    else:
        tests_to_run = TEST_CASES
    
    # Run tests
    results = []
    for test_case in tests_to_run:
        result = test_model(args.model, args.port, test_case)
        results.append((test_case["name"], result))
        print_result(test_case["name"], result, args.verbose)
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for _, r in results if r["success"])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"⚠ {total - passed} test(s) failed")
        for name, result in results:
            if not result["success"]:
                print(f"  - {name}: {result.get('error', 'Unknown error')}")
    
    # Performance summary
    successful_results = [r for _, r in results if r["success"]]
    if successful_results:
        avg_speed = sum(r["tokens_per_sec"] for r in successful_results) / len(successful_results)
        print(f"\nAverage speed: {avg_speed:.1f} tokens/sec")
    
    print()


if __name__ == "__main__":
    main()
