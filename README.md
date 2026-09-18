# vLLM on Google Cloud Run with NVIDIA L4 GPU

This project provides a complete infrastructure-as-code and deployment automation setup to run **vLLM** on **Google Cloud Run** with **NVIDIA L4 GPUs (24GB VRAM)** and model weights mounted from **Google Cloud Storage (GCS)** via **Cloud Storage FUSE**.

---

## Architecture

```
[ Client / OpenAI SDK / curl ]
              │ (HTTPS / Streamed Tokens)
              ▼
    [ Cloud Run Service ]
    ├── Accelerator: 1x NVIDIA L4 (24GB VRAM)
    ├── Machine Sizing: 8 vCPUs, 32 GiB RAM
    ├── Image: vllm/vllm-openai:latest
    ├── Volume Mount (GCS FUSE): /mnt/models ◄─── [ GCS Bucket: gs://${BUCKET_NAME} ]
    └── Autoscaling: 0 to 2 instances (scales to zero when idle)
```

---

## Directory Structure

```
vllm-gcp-cloudrun/
├── CASE_STUDY.md              # Industry Case Study & CFP Presentation Guide
├── document_intelligence_rag.py # Code-ready Enterprise RAG & Guided Extraction App
├── sample_contract.txt        # Sample enterprise compliance contract for testing
├── .env.example               # Configuration variables template
├── deploy.sh                  # One-click deployment script
├── service.yaml               # Declarative Knative/Cloud Run YAML definition
├── sync_model.py              # Hugging Face snapshot downloader & GCS rsync tool
├── client_test.py             # Python OpenAI test client with TTFT & TPOT latency profiling
├── requirements.txt           # Client and staging dependencies
└── README.md                  # Documentation and guide
```

---

## Quickstart

### 1. Configure Environment
Copy `.env.example` to `.env` and fill in your GCP project values:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
BUCKET_NAME=your-gcp-project-id-vllm-models
MODEL_ID=Qwen/Qwen2.5-7B-Instruct
MODEL_SUBDIR=models/Qwen2.5-7B-Instruct
SERVED_MODEL_NAME=qwen-7b
VLLM_API_KEY=sk-vllm-secure-token-12345
```

### 2. Stage Model Weights into GCS
Install the staging dependencies and sync the weights from Hugging Face to your bucket:
```bash
pip install -r requirements.txt

python sync_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --bucket your-gcp-project-id-vllm-models \
    --subdir models/Qwen2.5-7B-Instruct
```

### 3. Deploy to Cloud Run
Run the automated deployment script:
```bash
chmod +x deploy.sh
./deploy.sh
```

The script will:
1. Enable necessary GCP APIs (`run.googleapis.com`, `storage.googleapis.com`, etc.).
2. Ensure the GCS bucket exists in the target region.
3. Create the dedicated `vllm-cloudrun-sa` IAM service account and grant `storage.objectViewer`.
4. Deploy Cloud Run with 1x NVIDIA L4 GPU and GCS FUSE volume mount at `/mnt/models`.
5. Output the live public/private endpoint URL.

---

## Testing the Endpoint

### Via `curl` (Streaming Response):
```bash
export SERVICE_URL="https://vllm-l4-server-xxxxx.a.run.app"
export VLLM_API_KEY="sk-vllm-secure-token-12345"

curl -X POST "${SERVICE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d '{
    "model": "qwen-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful cloud assistant."},
      {"role": "user", "content": "Explain PagedAttention in 2 sentences."}
    ],
    "stream": true
  }'
```

### Via Python Test Client (with TTFT & TPOT Latency Metrics):
```bash
SERVICE_URL="https://vllm-l4-server-xxxxx.a.run.app" python client_test.py
```
Outputs:
* **Time To First Token (TTFT)** in ms
* **Time Per Output Token (TPOT)** in ms
* **Effective Tokens/sec** throughput
* **Structured Output Test** using Pydantic JSON schema
