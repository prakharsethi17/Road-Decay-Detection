# Road Decay Detection

## Installation  [Windows + Nvidia GPU]
```bash

# Setup Environment
python -m venv yolo11
.\yolo11\Scripts\activate

# Install Dependencies
pip install supervision ultralytics ruamel.yaml tensorflow 
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
winget install --id Git.Git -e --source winget

# Download N-RDD2024 Dataset [Augmented Dataset using Albumentations]
git lfs install
git clone https://huggingface.co/datasets/cappies/N-RDD2024




