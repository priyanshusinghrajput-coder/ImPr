#!/usr/bin/env bash
# ============================================================================
# inference/setup_pi.sh
# ============================================================================
# One-time setup script for the Raspberry Pi. Clones (or updates) the DSLR
# GitHub repo, creates a virtual environment, and installs the lightweight
# ARM-friendly dependency set (tflite-runtime instead of full TensorFlow).
#
# Usage (on the Pi):
#   chmod +x setup_pi.sh
#   ./setup_pi.sh https://github.com/<your-username>/<your-dslr-repo>.git
# ============================================================================
set -e

REPO_URL="${1:-}"
INSTALL_DIR="${2:-$HOME/DSLR}"

if [ -z "$REPO_URL" ]; then
    echo "Usage: ./setup_pi.sh <git-repo-url> [install-dir]"
    exit 1
fi

echo ">>> Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git espeak libespeak1 libatlas-base-dev 2>/dev/null || \
sudo apt-get install -y python3-venv python3-pip git espeak libespeak1

echo ">>> Cloning/updating repo at $INSTALL_DIR ..."
if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" pull
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

echo ">>> Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ">>> Installing lightweight Pi dependencies..."
pip install --upgrade pip
pip install mediapipe opencv-python numpy pyttsx3
pip install tflite-runtime || echo "[warn] tflite-runtime wheel not found for this Pi/Python combo; falling back to full tensorflow" && pip install tensorflow

echo ""
echo ">>> Setup complete."
echo "Activate the environment with:  source $INSTALL_DIR/venv/bin/activate"
echo "Run inference with:             python inference/realtime_inference.py"
