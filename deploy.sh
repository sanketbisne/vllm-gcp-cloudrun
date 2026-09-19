#!/usr/bin/env bash
# ==============================================================================
# deploy.sh
# End-to-end deployment script for vLLM on Google Cloud Run with NVIDIA L4 GPU.
# ==============================================================================

set -euo pipefail

ENV_FILE=".env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment variables from $ENV_FILE..."
    # shellcheck disable=SC1090
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "WARNING: .env not found! Looking for exported environment variables."
fi

# Mandatory variables
GCP_PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
GCP_REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-vllm-l4-server}"
BUCKET_NAME="${BUCKET_NAME:-${GCP_PROJECT_ID}-vllm-models}"
MODEL_SUBDIR="${MODEL_SUBDIR:-models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-7b}"
VLLM_API_KEY="${VLLM_API_KEY:-sk-vllm-secure-token-12345}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
CONCURRENCY="${CONCURRENCY:-32}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"

if [[ -z "$GCP_PROJECT_ID" ]]; then
    echo "Error: GCP_PROJECT_ID must be set in .env or via gcloud config." >&2
    exit 1
fi

echo "=========================================================="
echo " Deploying vLLM to Cloud Run with NVIDIA L4 GPU"
echo " Project:       $GCP_PROJECT_ID"
echo " Region:        $GCP_REGION"
echo " Service:       $SERVICE_NAME"
echo " Bucket:        gs://$BUCKET_NAME"
echo " Model Subdir:  $MODEL_SUBDIR"
echo " Served Name:   $SERVED_MODEL_NAME"
echo "=========================================================="

# 1. Ensure project configuration and APIs
echo "[1/5] Setting gcloud project and enabling APIs..."
gcloud config set project "$GCP_PROJECT_ID"
gcloud services enable \
    run.googleapis.com \
    storage.googleapis.com \
    compute.googleapis.com \
    artifactregistry.googleapis.com

# 2. Check/create GCS bucket
echo "[2/5] Verifying GCS Bucket gs://${BUCKET_NAME}..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
    echo "Creating bucket gs://${BUCKET_NAME} in ${GCP_REGION}..."
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
        --location="${GCP_REGION}" \
        --uniform-bucket-level-access
else
    echo "Bucket gs://${BUCKET_NAME} already exists."
fi

# 3. Create IAM Service Account
SA_NAME="vllm-cloudrun-sa"
SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

echo "[3/5] Configuring Service Account ${SA_EMAIL}..."
if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
    gcloud iam service-accounts create "${SA_NAME}" \
        --display-name="vLLM Cloud Run Service Account"
fi

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectViewer" >/dev/null

ENABLE_GPU="${ENABLE_GPU:-true}"

# 4. Deploy to Cloud Run
echo "[4/5] Deploying Cloud Run Service (GPU: ${ENABLE_GPU})..."

GPU_FLAGS=()
VLLM_ENV_VARS="VLLM_API_KEY=${VLLM_API_KEY}"
VLLM_EXTRA_ARG=""

if [[ "$ENABLE_GPU" == "true" ]]; then
    GPU_FLAGS=(--gpu=1 --gpu-type=nvidia-l4 --cpu=8 --memory=32Gi)
    VLLM_EXTRA_ARG="--gpu-memory-utilization=${GPU_MEMORY_UTILIZATION}"
else
    GPU_FLAGS=(--cpu=4 --memory=16Gi)
    VLLM_ENV_VARS="${VLLM_ENV_VARS},VLLM_TARGET_DEVICE=cpu,OMP_NUM_THREADS=4,VLLM_CPU_KVCACHE_SPACE=4"
    VLLM_EXTRA_ARG="--dtype=float16"
fi

gcloud beta run deploy "${SERVICE_NAME}" \
    --image="vllm/vllm-openai:latest" \
    --region="${GCP_REGION}" \
    --service-account="${SA_EMAIL}" \
    "${GPU_FLAGS[@]}" \
    --no-cpu-throttling \
    --cpu-boost \
    --execution-environment=gen2 \
    --add-volume="name=model-store,type=cloud-storage,bucket=${BUCKET_NAME},readonly=true" \
    --add-volume-mount="volume=model-store,mount-path=/mnt/models" \
    --port=8000 \
    --set-env-vars="${VLLM_ENV_VARS}" \
    --args="serve",\
"/mnt/models/${MODEL_SUBDIR}",\
"--served-model-name=${SERVED_MODEL_NAME}",\
"${VLLM_EXTRA_ARG}",\
"--max-model-len=${MAX_MODEL_LEN}",\
"--api-key=${VLLM_API_KEY}",\
"--port=8000" \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances="${MAX_INSTANCES}" \
    --concurrency="${CONCURRENCY}" \
    --timeout=900 \
    --startup-probe=initialDelaySeconds=10,periodSeconds=10,timeoutSeconds=10,failureThreshold=60,tcpSocket.port=8000 \
    --no-allow-unauthenticated

# 5. Output Service URL
echo "[5/5] Deployment complete! Fetching URL..."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region="${GCP_REGION}" --format="value(status.url)")
echo ""
echo "=========================================================="
echo " vLLM Endpoint ready:"
echo " URL:     ${SERVICE_URL}"
echo " API Key: ${VLLM_API_KEY}"
echo " Model:   ${SERVED_MODEL_NAME}"
echo "=========================================================="
echo "Test command:"
echo "curl -X POST '${SERVICE_URL}/v1/chat/completions' \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'Authorization: Bearer ${VLLM_API_KEY}' \\"
echo "  -d '{\"model\": \"${SERVED_MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
