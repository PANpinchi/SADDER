## **SADDER**: **S**egmentation-**A**ugmented **D**ifferential **D**epth **E**stimation **R**egressor for Underwater Depth Estimation
This repository is the PyTorch implementation of Segmentation-Augmented Differential Depth Estimation Regressor for Underwater Depth Estimation.


## Getting Started
```bash
git clone https://github.com/PANpinchi/SADDER.git

cd SADDER
```

## Installation and Setup
To set up the virtual environment and install the required packages, use the following commands:
```bash
conda create -n sadder python=3.10

conda activate sadder

source install_environment.sh
```
**(Optional)** or manually execute the following command:
```bash
# CUDA 11.3
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

pip install -r dependencies.txt

pip install opencv-contrib-python

pip install tifffile
```

**(Optional)** If you choose to execute the command manually, you also need to manually set up the virtual environment of BARIS-ERA:
```bash
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
```

**(Optional)** and download the pre-trained BARIS-ERA model:
```bash
mkdir pretrained

cd pretrained

gdown --id 1-nK4MYPiW5bB8wDHbIXzLimRkLLpek6x

gdown --id 1_MxeMnI11CuvWHGEvud7COMwsPyVeNNv

cd ../..
```
Note: `*.pth` files should be placed in the `/pretrained` folder.


Download the pretrained model weights by running the following:
```bash
source download_pretrained_weights.sh
```

**(Optional)** or manually execute the following commands below to download the pre-trained CPD model:
```bash
cd CPD

gdown --id 1Ezqf3rfBbC4iREjE9TfqDt5_QEvBXZ7F

cd ..
```
Note: `CPD-R.pth` files should be placed in the `/CPD` folder.


**(Optional)** Run the commands below to download the pre-trained UDepth model:
```bash
mkdir saved_udepth_model

cd saved_udepth_model

gdown --id 1VakMGHTAc2b6baEQvijeU2SapClreIYE

gdown --id 1MaNGn8aKYDXrtmuTsaNlhIyk-IeMRJnO

cd ..
```
Note: `*.pth` files should be placed in the `/saved_udepth_model` folder.

**(Optional)** Run the commands below to download the pre-trained UWDepth model:
```bash
cd data/saved_models

gdown --id 1oDcUBglz4NvfO3JsyOnqemDffFHHqr3J

gdown --id 14qFV0lR_yDLILSfqr-8d1ajd--gfu-P6

gdown --id 1seBVgaUzDZKMfWBmS0ZMUDo_NdDV0y9B

cd ../..
```
Note: `*.pth` files should be placed in the `/data/saved_models` folder.

**(Optional)** Run the commands below to download the pre-trained UWDepth with SADDER model:
```bash
cd saved_models

gdown --id 1eqbV9Jq7WCSWd6btxHVD1r2ykMyWLhpe

cd ..
```
Note: `*.pth` files should be placed in the `/saved_models` folder.


## Inference
Run the commands below to perform a pretrained model on images.
```bash
python inference.py
```

## Preprocessing
The `helper_scripts` folder contains useful scripts which can be used for preprocessing of datasets, such as extracting visual features for usage as sparse depth measurements or creating train/test splits. In general, every data point in a dataset needs:
- RGB image (see `data/example_dataset/rgb`)
- keypoint location with corresponding depth (see `data/example_dataset/features`) *
- depth image ground truth (for training / evaluation only, see `data/example_dataset/depth`)

\* check out `helper_scripts/extract_dataset_features.py` for a simple example on how such features can be generated if ground truth is available. If not, you could use e.g. SLAM.

Then, the `.csv` file defines the tuples, see `data/example_dataset/dataset.csv`.

Make sure that you also load your data correctly via the dataloader, e.g. depending on your dataset, images can be in uint8, uint16 or float format (see `data/example_dataset/dataset.py`)
```bash
python inference_data_preprocessing.py

python helper_scripts/extract_inference_data_depth.py

python helper_scripts/extract_inference_data_features.py

python helper_scripts/extract_inference_data_seg.py
```

## Citation
If you use this code, please cite the following:
```bibtex
@misc{pan2025_sadder,
    title  = {SADDER: Segmentation-Augmented Differential Depth Estimation Regressor for Underwater Depth Estimation},
    author = {Pin-Chi Pan and Soo-Chang Pei},
    url    = {https://github.com/PANpinchi/SADDER},
    year   = {2025}
}
```
