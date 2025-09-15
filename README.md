# Road Decay Detection
 
## Plug and Play Training - YOLOv11 [Windows + Nvidia RTX4060 Laptop GPU]
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

# YOLOv11 Training
python -c "from ultralytics import YOLO; model=YOLO(r'your path\yolo11n.pt'); model.train(data=r'your path\N-RDD2024\data.yaml', epochs=300, patience=50, batch=0.80, imgsz=640, cache='disk', device=0, workers=8, project=r'your path', name='y11n_v1', optimizer='auto', cos_lr=True, profile=True, warmup_epochs=3, val=True, plots=True)"

# Note: optimizer, lr0, lrf, momentum, weight_decay, warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl are hyperparameters to be finetuned. At the moment, everything is set to default.





