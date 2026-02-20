#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration
# =========================
VENV_PATH="/work/envs/srgan"
DATA_ROOT="/data1/datasets"
DATASET_NAME="SEN2NAIP"
FILENAME="sen2naipv2-crosssensor.taco"
URL="https://huggingface.co/datasets/tacofoundation/SEN2NAIPv2/resolve/main/${FILENAME}"

TARGET_DIR="${DATA_ROOT}/${DATASET_NAME}"

# =========================
# Activate environment
# =========================
source "${VENV_PATH}/bin/activate"

# =========================
# Prepare directory
# =========================
mkdir -p "${TARGET_DIR}"
echo "Downloading to: ${TARGET_DIR}"
sleep 2

# =========================
# Download
# =========================
aria2c \
  --max-connection-per-server=8 \
  --split=8 \
  --dir="${TARGET_DIR}" \
  --out="${FILENAME}" \
  "${URL}"

echo "Download finished: ${TARGET_DIR}/${FILENAME}"
