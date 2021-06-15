#!/bin/bash

TORCH=1.6.0
CUDA=101

python -m pip install transformers
python -m pip install ogb
python -m pip install pytorch_lightning
python -m pip install pickle5
python -m pip install networkx
conda install -y -c rdkit rdkit
python -m pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-geometric
