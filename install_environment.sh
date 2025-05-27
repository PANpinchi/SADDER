#!/bin/bash

echo "✅ Installing PyTorch + CUDA Toolkit via conda..."
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

echo "✅ Installing additional Python dependencies..."
pip install -r dependencies.txt
pip install opencv-contrib-python
pip install tifffile

git clone https://github.com/PANpinchi/BARIS-ERA.git
cd ./BARIS-ERA
pip install -v -e .
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install terminaltables
pip install pycocotools
pip install scikit-learn
pip install numpy==1.23.5
pip install gdown
pip install mmcls
pip install yapf==0.40.1
pip install natsort

mkdir pretrained
cd pretrained
gdown --id 1-nK4MYPiW5bB8wDHbIXzLimRkLLpek6x
gdown --id 1_MxeMnI11CuvWHGEvud7COMwsPyVeNNv
cd ../..

echo "🎉 All packages installed successfully!"
