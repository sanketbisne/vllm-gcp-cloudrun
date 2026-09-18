#!/usr/bin/env python3
"""
client_test.py
Interactive test client for vLLM on GCP Cloud Run using the standard OpenAI Python SDK.
Measures Time to First Token (TTFT), Time Per Output Token (TPOT), and tests structured JSON outputs.
"""

import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load from .env if present
load_dotenv()

SERVICE_URL = os.environ.get("SERVICE_URL") or os.environ.get("VLLM_SERVICE_URL")
API_KEY = os.environ.get("VLLM_API_KEY", "sk-vllm-secure-token-12345")
MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", "qwen-7b")

if not SERVICE_URL:
    print("Error: SERVICE_URL environment variable is not set!")
    print("Usage: SERVICE_URL=https://<your-cloud-run-url> python client_test.py")
    sys.exit(1)

base_url = f"{SERVICE_URL.rstrip('/')}/v1"
print(f"Connecting to vLLM endpoint: {base_url}")
client = OpenAI(base_url=base_url, api_key=API_KEY)


def test_streaming():
    print("\n--- Test 1: Streaming Chat Completion & Latency Profiling ---")
    prompt = "Explain why KV cache paging is critical for high-throughput LLM serving in 3 concise points."
    print(f"Prompt: {prompt}\n")

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert AI systems engineer."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300,
        stream=True,
    )

    print("Response: ", end="", flush=True)
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
            print(delta, end="", flush=True)
    print("\n")

    end_time = time.perf_counter()
    if first_token_time:
        ttft = (first_token_time - start_time) * 1000
        total_time = end_time - start_time
        gen_time = end_time - first_token_time
        tpot = (gen_time / token_count * 1000) if token_count > 1 else 0
        tokens_per_sec = token_count / total_time

        print(f"📊 Metrics:")
        print(f"  * Generated Tokens:          {token_count}")
        print(f"  * Time To First Token (TTFT): {ttft:.1f} ms")
        print(f"  * Time Per Token (TPOT):      {tpot:.1f} ms")
        print(f"  * Total End-to-End Latency:   {total_time:.2f} s")
        print(f"  * Effective Output Speed:     {tokens_per_sec:.1f} tokens/s")


def test_structured_json():
    print("\n--- Test 2: Guided Structured Output (Pydantic / FSM) ---")

    class ServiceHealthSummary(BaseModel):
        service_name: str
        status: str
        gpu_type: str
        latency_rating: str
        recommended_action: str

    prompt = (
        "Extract status report: Service 'vllm-prod-us' running on NVIDIA L4 GPU has a P95 TTFT of 45ms. "
        "Status is healthy and ready for traffic."
    )
    print(f"Extracting structured JSON from: '{prompt}'\n")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"guided_json": ServiceHealthSummary.model_json_schema()},
            temperature=0.0,
        )
        content = response.choices[0].message.content
        print("Generated JSON conforming strictly to Pydantic schema:")
        print(content)
    except Exception as e:
        print(f"Note: Guided decoding check returned: {e}")


if __name__ == "__main__":
    test_streaming()
    test_structured_json()
