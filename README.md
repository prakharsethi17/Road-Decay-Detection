# Road Decay Detection

**Note:** The prediction algorithm utilized in this project is currently under patent application and therefore cannot be shared publicly. The code provided in this repository is exclusively for road defect detection purposes and is released under an open-source license.

## Project Status

This project is currently under active development. Hyperparameter tuning and optimization are ongoing to improve model performance.

## Dataset

This project utilizes the **N-RDD2024** (Road Damage and Defects) dataset:

KAYA, Ömer; Çodur, Muhammed Yasin (2024), "N-RDD2024:Road damage and defects", Mendeley Data, V3, doi: 10.17632/27c8pwsd6v.3

The dataset has been augmented and modified to suit the specific requirements of this implementation. The augmented dataset is available for download through the link provided in the Installation section.

## Repository Structure

The repository contains the following key directories:

- **y11n_v22/**: YOLOv11 training outputs including model weights, evaluation metrics, and visualization results
  - Confusion matrices (normalized and standard)
  - Bounding box precision-recall curves
  - Training and validation batch samples
  - Label distributions
  - Model performance results

- **weight_analysis/**: Comprehensive analysis of model weights across training iterations
  - Convolutional layer weight visualizations
  - Feature distribution summaries
  - Weight evolution tracking

- **layer_analysis/**: Feature map visualizations from different model layers
  - Convolutional layer outputs
  - Feature extraction analysis
  - Layer-wise activation patterns

## System Requirements

- **Operating System:** Windows 10/11
- **GPU:** NVIDIA RTX 4060 Laptop GPU (or equivalent)
- **CUDA:** Version 12.6
- **Python:** 3.8 or higher

## Installation

### Environment Setup

Create and activate a virtual environment:

```bash
python -m venv yolo11
.\yolo11\Scripts\activate
```

### Install Dependencies

Install required packages:

```bash
pip install supervision ultralytics ruamel.yaml tensorflow
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
winget install --id Git.Git -e --source winget
```

### Download Dataset

Access the augmented N-RDD2024 dataset:

[Download Augmented Dataset](https://drive.google.com/drive/folders/1dH5W83y3a3k5LmjC70ZthYw-Fmdnc9aD?usp=sharing)

The dataset has been preprocessed using Albumentations library for data augmentation.

## Training

### YOLOv11 Model Training

Execute the following command to train the YOLOv11 model:

```bash
python -c "from ultralytics import YOLO; model=YOLO(r'your_path\yolo11n.pt'); model.train(data=r'your_path\N-RDD2024\data.yaml', epochs=300, patience=50, batch=0.80, imgsz=640, cache='disk', device=0, workers=8, project=r'your_path', name='y11n_v1', optimizer='auto', cos_lr=True, profile=True, warmup_epochs=3, val=True, plots=True)"
```

Replace `your_path` with the appropriate directory paths on your system.

### Training Parameters

Current configuration uses the following parameters:

- **Epochs:** 300
- **Patience:** 50 (early stopping)
- **Batch Size:** 0.80 (auto-batch)
- **Image Size:** 640x640
- **Optimizer:** Auto-selected
- **Learning Rate Schedule:** Cosine annealing
- **Warmup Epochs:** 3
- **Device:** GPU (device=0)
- **Workers:** 8

### Hyperparameter Tuning

The following hyperparameters are currently under optimization:

- `optimizer`: Optimization algorithm selection
- `lr0`: Initial learning rate
- `lrf`: Final learning rate factor
- `momentum`: SGD momentum/Adam beta1
- `weight_decay`: L2 regularization coefficient
- `warmup_epochs`: Number of warmup epochs
- `warmup_momentum`: Warmup momentum
- `warmup_bias_lr`: Warmup bias learning rate
- `box`: Box loss weight
- `cls`: Classification loss weight
- `dfl`: Distribution focal loss weight

Current settings are configured to default values and are subject to change as optimization progresses.

## Model Evaluation

The repository includes comprehensive evaluation outputs:

- Confusion matrices for class-wise performance analysis
- Precision-Recall curves for detection accuracy assessment
- F1-Confidence curves for threshold optimization
- Training loss curves and validation metrics

## Contributing

Contributions to the detection module are welcome. Please ensure that any pull requests maintain code quality and include appropriate documentation.

## License

The detection code in this repository is open-source. The prediction algorithm is proprietary and under patent application.

## Citation

If using this dataset, please cite:

```
KAYA, Ömer; Çodur, Muhammed Yasin (2024), "N-RDD2024:Road damage and defects", 
Mendeley Data, V3, doi: 10.17632/27c8pwsd6v.3
```

## Disclaimer

This project is under active development. Model performance metrics and configurations are subject to change as hyperparameter optimization continues.

## Contact

For questions regarding the detection implementation, please open an issue in this repository.

---

**Last Updated:** October 2025
