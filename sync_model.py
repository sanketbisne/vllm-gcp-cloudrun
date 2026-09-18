#!/usr/bin/env python3
"""
sync_model.py
Downloads model weights from Hugging Face and stages them directly to a Google Cloud Storage bucket.
Usage:
    python sync_model.py --model Qwen/Qwen2.5-7B-Instruct --bucket my-bucket --subdir models/Qwen2.5-7B-Instruct
"""

import argparse
import os
import subprocess
import sys
from huggingface_hub import snapshot_download


def download_and_sync(model_id: str, bucket_name: str, subdir: str, local_tmp_dir: str = "./tmp_model"):
    print(f"[1/3] Downloading {model_id} snapshot from Hugging Face into {local_tmp_dir}...")
    os.makedirs(local_tmp_dir, exist_ok=True)
    
    # Download weights without symlinks so gcloud storage rsync copies full files
    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=local_tmp_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.pt", "*.bin"] if "safetensors" in model_id.lower() else None
    )
    print(f"Downloaded weights to {local_path}")

    destination_uri = f"gs://{bucket_name}/{subdir}"
    print(f"[2/3] Uploading model weights to {destination_uri}...")

    cmd = ["gcloud", "storage", "rsync", "-r", local_path, destination_uri]
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error syncing to GCS! Command failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[3/3] Successfully staged model weights in {destination_uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync model weights from Hugging Face to GCS")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face repo ID")
    parser.add_argument("--bucket", type=str, required=True, help="Target GCS bucket name")
    parser.add_argument("--subdir", type=str, default="models/Qwen2.5-7B-Instruct", help="Subdirectory path in GCS")
    parser.add_argument("--tmp-dir", type=str, default="./tmp_weights", help="Temporary local download directory")

    args = parser.parse_args()
    download_and_sync(args.model, args.bucket, args.subdir, args.tmp_dir)
