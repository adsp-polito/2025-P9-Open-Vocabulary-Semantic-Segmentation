#!/usr/bin/env bash
set -e

echo "=== Installing Python dependencies ==="
pip install torch torchvision transformers==4.41.2 safetensors matplotlib
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/facebookresearch/detectron2.git

echo "=== Cloning Talk2DINO-ViTB from HuggingFace ==="
if [ -d "/Talk2DINO-ViTB" ]; then
    echo "Talk2DINO-ViTB already exists, skipping clone."
else
    git clone https://huggingface.co/lorebianchi98/Talk2DINO-ViTB /Talk2DINO-ViTB
fi

echo "=== Done! Now run: python eval_talk2dino.py ==="
