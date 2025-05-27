#!/bin/bash

echo "✅ Installing PyTorch + CUDA Toolkit via conda..."
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

echo "✅ Installing additional Python dependencies..."
pip install -r dependencies.txt

pip install opencv-contrib-python

pip install tifffile

echo "🎉 All packages installed successfully!"
