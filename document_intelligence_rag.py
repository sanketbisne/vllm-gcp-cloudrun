#!/usr/bin/env python3
"""
document_intelligence_rag.py
=============================================================================
Industry Case Study: High-Throughput Enterprise Document Intelligence & RAG
Backend: vLLM on Google Cloud Run (NVIDIA L4 GPU + GCS FUSE)

Features:
1. Low-Latency Contextual Streaming Q&A (Profiling TTFT and TPOT).
2. Guided Decoding / Schema Enforcement (Pydantic -> JSON Schema).
3. Enterprise FinOps & Latency Comparison Report.
=============================================================================
"""

import os
import sys
import time
import json
import argparse
from typing import List, Optional
try:
    from openai import OpenAI
    from pydantic import BaseModel, Field
except ImportError:
    pass
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

# ANSI Color codes for clean presentation
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class ComplianceAuditReport(BaseModel):
    document_title: str = Field(description="Title or identification of the agreement")
    parties_involved: List[str] = Field(description="Names of contracting parties")
    jurisdiction_and_governing_law: str = Field(description="Governing law state or region")
    sla_availability_percent: float = Field(description="Guaranteed uptime percentage")
    target_ttft_ms: Optional[int] = Field(description="Target time-to-first-token in milliseconds if specified")
    max_data_breach_indemnity_usd: float = Field(description="Liquidated damages / indemnity ceiling in USD")
    compliance_standards: List[str] = Field(description="Compliance frameworks (e.g. SOC 2, HIPAA, ISO)")
    private_inference_guarantee: bool = Field(description="Whether third-party LLM data egress is explicitly prohibited")
    risk_summary: str = Field(description="Brief assessment of operational risk")


def get_client():
    service_url = os.environ.get("SERVICE_URL") or os.environ.get("VLLM_SERVICE_URL")
    api_key = os.environ.get("VLLM_API_KEY", "sk-vllm-secure-token-12345")
    model_name = os.environ.get("SERVED_MODEL_NAME", "qwen-7b")

    if not service_url:
        print(f"{RED}{BOLD}Error:{RESET} SERVICE_URL is not configured.")
        print("Set it in your .env or pass as an environment variable:")
        print("  export SERVICE_URL=\"https://vllm-l4-server-xxxxx.a.run.app\"")
        print("  python document_intelligence_rag.py")
        sys.exit(1)

    base_url = f"{service_url.rstrip('/')}/v1"
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model_name, service_url


def run_streaming_rag(client: OpenAI, model_name: str, document_text: str, user_query: str):
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}STEP 1: Contextual Streaming RAG Query (Real-Time Latency Profiling){RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{YELLOW}{BOLD}Document excerpt loaded:{RESET} {len(document_text)} characters")
    print(f"{YELLOW}{BOLD}User Query:{RESET} {user_query}\n")

    system_prompt = (
        "You are an expert Enterprise Compliance & Cloud Infrastructure Attorney. "
        "Analyze the provided document and answer the question accurately, citing specific sections."
    )

    full_user_content = f"### DOCUMENT CONTEXT:\n{document_text}\n\n### QUESTION:\n{user_query}"

    start_perf = time.perf_counter()
    first_token_time = None
    token_count = 0

    print(f"{BOLD}Streaming Answer:{RESET} ", end="", flush=True)

    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_content}
        ],
        temperature=0.2,
        max_tokens=400,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            token_count += 1
            print(f"{GREEN}{delta}{RESET}", end="", flush=True)

    print("\n")
    end_perf = time.perf_counter()

    if first_token_time:
        ttft_ms = (first_token_time - start_perf) * 1000
        total_sec = end_perf - start_perf
        gen_sec = end_perf - first_token_time
        tpot_ms = (gen_sec / token_count * 1000) if token_count > 1 else 0
        speed_tps = token_count / total_sec if total_sec > 0 else 0

        print(f"{BOLD}------------------------------------------------------------------------{RESET}")
        print(f"{BOLD}📊 Performance & Latency Telemetry:{RESET}")
        print(f"  • {BOLD}Time To First Token (TTFT):{RESET} {CYAN}{ttft_ms:.1f} ms{RESET} (P95 SLA < 120ms)")
        print(f"  • {BOLD}Time Per Output Token (TPOT):{RESET} {CYAN}{tpot_ms:.1f} ms{RESET}")
        print(f"  • {BOLD}Effective Throughput:{RESET}        {GREEN}{speed_tps:.1f} tokens/second{RESET}")
        print(f"  • {BOLD}Total Generation Time:{RESET}       {total_sec:.2f} seconds ({token_count} tokens)")
        print(f"{BOLD}------------------------------------------------------------------------{RESET}")


def run_guided_extraction(client: OpenAI, model_name: str, document_text: str):
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}STEP 2: Zero-Hallucination Guided Entity Extraction (JSON Schema){RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{DIM}Enforcing Pydantic Schema via vLLM's Finite State Machine (FSM) grammar engine...{RESET}\n")

    schema = ComplianceAuditReport.model_json_schema()
    start_time = time.perf_counter()

    prompt = (
        "Extract a complete compliance and legal audit report from this enterprise agreement. "
        "Strictly adhere to the required JSON schema.\n\n"
        f"DOCUMENT:\n{document_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"guided_json": schema},
            temperature=0.0,
        )
        elapsed = time.perf_counter() - start_time
        raw_json = response.choices[0].message.content

        # Validate with Pydantic
        parsed = ComplianceAuditReport.model_validate_json(raw_json)

        print(f"{GREEN}{BOLD}✓ Successfully generated and validated strict JSON in {elapsed:.2f}s:{RESET}\n")
        print(json.dumps(parsed.model_dump(), indent=2))
        return parsed

    except Exception as e:
        print(f"{RED}Extraction error or guided decoding mismatch: {e}{RESET}")
        return None


def print_finops_comparison():
    print(f"\n{BOLD}{CYAN}========================================================================{RESET}")
    print(f"{BOLD}{CYAN}STEP 3: FinOps Total Cost of Ownership (TCO) Comparison{RESET}")
    print(f"{BOLD}{CYAN}========================================================================{RESET}")
    print(
        f"""
Scenario: Enterprise Document Processing Pipeline (1,000,000 queries / month, avg 800 tokens context)

| Architecture                              | Idle Cost (0 req) | Active Inference Cost | Data Egress Risk   |
| :---------------------------------------- | :---------------- | :-------------------- | :----------------- |
| **Commercial Closed API**                 | $0 / mo           | ~$3,000 - $6,000 / mo | ⚠️ High (Vendor API) |
| **Dedicated GKE/GCE Cluster (2x L4 GPUs)**| ~$720 / mo (idle) | ~$720 / mo fixed      | 🔒 Zero (Private)  |
| **Cloud Run Serverless GPU (Scale-to-0)** | **$0 / mo**       | **~$650 - $950 / mo** | 🔒 **Zero (Private)**|

{BOLD}Key Takeaways for Attendees:{RESET}
1. {GREEN}Scale-to-Zero Savings:{RESET} Saves $720/mo when idle overnight/weekends compared to static clusters.
2. {GREEN}Data Privacy:{RESET} All requests stay inside GCP VPC with GCS FUSE model weights.
3. {GREEN}Low Latency:{RESET} vLLM PagedAttention handles multiple concurrent streams without OOM.
"""
    )


def main():
    parser = argparse.ArgumentParser(description="Enterprise Document Intelligence & RAG Showcase on vLLM")
    parser.add_argument("--doc", type=str, default="sample_contract.txt", help="Path to document to analyze")
    parser.add_argument("--query", type=str, default="What are the specific penalties and liquidated damages if a data breach occurs?", help="Question for RAG")
    parser.add_argument("--skip-extract", action="store_true", help="Skip structured JSON extraction")
    args = parser.parse_args()

    if not os.path.exists(args.doc):
        print(f"{RED}Document file not found: {args.doc}{RESET}")
        sys.exit(1)

    with open(args.doc, "r", encoding="utf-8") as f:
        document_text = f.read()

    client, model_name, service_url = get_client()

    print(f"{BOLD}Connected to Cloud Run vLLM Endpoint:{RESET} {CYAN}{service_url}{RESET}")
    print(f"{BOLD}Model Target:{RESET} {GREEN}{model_name}{RESET}")

    # 1. Real-time streaming RAG
    run_streaming_rag(client, model_name, document_text, args.query)

    # 2. Schema-guided JSON extraction
    if not args.skip_extract:
        run_guided_extraction(client, model_name, document_text)

    # 3. FinOps Breakdown
    print_finops_comparison()


if __name__ == "__main__":
    main()
