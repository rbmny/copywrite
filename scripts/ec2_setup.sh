#!/bin/bash
# EC2 bootstrap script for copywrite training
# Runs on a fresh g5.xlarge (Deep Learning AMI Ubuntu)
set -e

echo "=== copywrite EC2 setup ==="

# System deps
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg > /dev/null 2>&1
echo "[OK] ffmpeg installed"

# Python deps
pip install -q --upgrade pip
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q acids-rave
pip install -q transformers peft accelerate datasets audiocraft
pip install -q soundfile librosa numpy scipy
echo "[OK] Python packages installed"

# Verify GPU
python3 -c "import torch; print(f'[OK] GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"

echo "=== EC2 setup complete ==="
