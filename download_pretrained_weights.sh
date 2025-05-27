#!/bin/bash

echo "✅ Download the pre-trained BARIS-ERA model..."
cd ./BARIS-ERA
mkdir pretrained
cd pretrained
gdown --id 1-nK4MYPiW5bB8wDHbIXzLimRkLLpek6x
gdown --id 1_MxeMnI11CuvWHGEvud7COMwsPyVeNNv
cd ../..

echo "✅ Download the pre-trained CPD model..."
cd CPD
gdown --id 1Ezqf3rfBbC4iREjE9TfqDt5_QEvBXZ7F
cd ..

echo "✅ Download the pre-trained UDepth model..."
mkdir saved_udepth_model
cd saved_udepth_model
gdown --id 1VakMGHTAc2b6baEQvijeU2SapClreIYE
gdown --id 1MaNGn8aKYDXrtmuTsaNlhIyk-IeMRJnO
cd ..

echo "✅ Download the pre-trained UWDepth model..."
cd data/saved_models
gdown --id 1oDcUBglz4NvfO3JsyOnqemDffFHHqr3J
gdown --id 14qFV0lR_yDLILSfqr-8d1ajd--gfu-P6
gdown --id 1seBVgaUzDZKMfWBmS0ZMUDo_NdDV0y9B
cd ../..

echo "✅ Download the pre-trained UWDepth with SADDER model..."
cd saved_models
gdown --id 1eqbV9Jq7WCSWd6btxHVD1r2ykMyWLhpe
cd ..

echo "🎉 All pre-trained models downloaded successfully!"
