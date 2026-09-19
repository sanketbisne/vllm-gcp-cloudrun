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
DETECTED_REGION="$(gcloud config get-value compute/region 2>/dev/null || true)"
if [[ -n "$DETECTED_REGION" && "$DETECTED_REGION" != "(unset)" ]]; then
    GCP_REGION="${GCP_REGION:-$DETECTED_REGION}"
else
    GCP_REGION="${GCP_REGION:-us-west1}"
fi
SERVICE_NAME="${SERVICE_NAME:-vllm-l4-server}"
BUCKET_NAME="${BUCKET_NAME:-${GCP_PROJECT_ID}-vllm-models}"
VLLM_API_KEY="${VLLM_API_KEY:-sk-vllm-secure-token-12345}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
CONCURRENCY="${CONCURRENCY:-32}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"
ENABLE_GPU="${ENABLE_GPU:-true}"

if [[ "$ENABLE_GPU" == "true" ]]; then
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
    MODEL_SUBDIR="${MODEL_SUBDIR:-models/Qwen2.5-7B-Instruct}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-7b}"
    GPU_FLAGS=(--gpu=1 --gpu-type=nvidia-l4 --cpu=8 --memory=32Gi)
    VLLM_SERVE_ARGS="serve,/mnt/models/${MODEL_SUBDIR},--served-model-name=${SERVED_MODEL_NAME},--max-model-len=${MAX_MODEL_LEN},--api-key=${VLLM_API_KEY},--port=8080,--gpu-memory-utilization=${GPU_MEMORY_UTILIZATION}"
    VLLM_ENV_VARS="VLLM_API_KEY=${VLLM_API_KEY}"
else
    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}"
    MODEL_SUBDIR="${MODEL_SUBDIR:-models/Qwen2.5-1.5B-Instruct}"
    SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-1.5b}"
    GPU_FLAGS=(--cpu=4 --memory=16Gi)
    VLLM_SERVE_ARGS="serve,/mnt/models/${MODEL_SUBDIR},--served-model-name=${SERVED_MODEL_NAME},--max-model-len=${MAX_MODEL_LEN},--api-key=${VLLM_API_KEY},--port=8080,--dtype=float32,--enforce-eager"
    VLLM_ENV_VARS="VLLM_API_KEY=${VLLM_API_KEY},VLLM_TARGET_DEVICE=cpu,OMP_NUM_THREADS=4,VLLM_CPU_KVCACHE_SPACE=4"
fi

if [[ -z "$GCP_PROJECT_ID" ]]; then
    echo "Error: GCP_PROJECT_ID must be set in .env or via gcloud config." >&2
    exit 1
fi

echo "=========================================================="
echo " Deploying vLLM to Cloud Run (GPU: ${ENABLE_GPU})"
echo " Project:       $GCP_PROJECT_ID"
echo " Region:        $GCP_REGION"
echo " Service:       $SERVICE_NAME"
echo " Bucket:        gs://$BUCKET_NAME"
echo " Model ID:      $MODEL_ID"
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

# 2.5 Auto-verify & sync model weights if missing
echo "[2.5/5] Checking model weights at gs://${BUCKET_NAME}/${MODEL_SUBDIR}..."
if ! gcloud storage ls "gs://${BUCKET_NAME}/${MODEL_SUBDIR}/config.json" &>/dev/null; then
    PYTHON_EXEC="python3"
    if [[ -x ".venv/bin/python3" ]]; then
        PYTHON_EXEC=".venv/bin/python3"
    fi
    echo "Model weights missing in GCS. Syncing ${MODEL_ID} to gs://${BUCKET_NAME}/${MODEL_SUBDIR}..."
    "$PYTHON_EXEC" sync_model.py --model "${MODEL_ID}" --bucket "${BUCKET_NAME}" --subdir "${MODEL_SUBDIR}"
else
    echo "Model weights verified in GCS bucket."
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

# 4. Deploy to Cloud Run
echo "[4/5] Deploying Cloud Run Service (GPU: ${ENABLE_GPU})..."

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
    --port=8080 \
    --set-env-vars="${VLLM_ENV_VARS}" \
    --command="vllm" \
    --args="${VLLM_SERVE_ARGS}" \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances="${MAX_INSTANCES}" \
    --concurrency="${CONCURRENCY}" \
    --timeout=900 \
    --startup-probe=initialDelaySeconds=15,periodSeconds=10,timeoutSeconds=10,failureThreshold=60,httpGet.port=8080,httpGet.path=/health \
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
