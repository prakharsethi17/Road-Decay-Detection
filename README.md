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

### y11n_v22/
YOLOv11 training outputs including model weights, evaluation metrics, and visualization results (21 items):

**Folders:**
- `test_eval/`: Test evaluation results and metrics
- `weights/`: Trained model weights and checkpoints

**Evaluation Metrics:**
- `confusion_matrix.png`: Standard confusion matrix for class-wise performance
- `confusion_matrix_normalized.png`: Normalized confusion matrix
- `BoxF1_curve.png`: F1-score vs confidence curve for bounding boxes
- `BoxP_curve.png`: Precision curve for box predictions
- `BoxPR_curve.png`: Precision-Recall curve for box detection
- `BoxR_curve.png`: Recall curve for box predictions

**Training Visualizations:**
- `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`: Training batch samples with ground truth annotations
- `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg`: Validation batch ground truth labels
- `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`: Validation batch predictions

**Results:**
- `labels.jpg`: Label distribution visualization
- `results.png`: Training metrics and loss curves
- `results.csv`: Detailed training metrics in CSV format
- `args.yaml`: Training arguments and configuration

### weight_analysis/
Comprehensive analysis of model weights across training iterations (89 items):

**Weight Visualizations:**
- `weight_distribution_summary.png`: Overall weight distribution across the model
- `weights_model_0_conv.png` through `weights_model_13_m_0_cv2_conv.png`: Layer-by-layer weight visualizations for convolutional layers

**Analysis includes:**
- Convolutional layer weight heatmaps at different model depths
- Weight distribution patterns across model iterations
- Feature extraction weight evolution
- Attention mechanism weights (cv1, cv2 variants)
- Multi-scale detection head weights (m_0 variants)

### layer_analysis/
Feature map visualizations from different model layers (89 items):

**Feature Map Visualizations:**
- `feature_maps_model_0_conv.png` through `feature_maps_model_16_m_0_cv2_conv.png`: Layer-wise feature activation patterns

**Analysis covers:**
- Input layer feature maps
- Intermediate convolutional layer activations
- Multi-scale feature pyramid outputs
- Detection head feature representations
- Spatial attention patterns across network depth

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

The repository includes comprehensive evaluation outputs in the `y11n_v22/` directory:

- **Confusion Matrices:** Both standard and normalized versions for class-wise performance analysis
- **Precision-Recall Curves:** Box detection accuracy assessment across confidence thresholds
- **F1-Confidence Curves:** Optimal threshold identification for detection
- **Training Visualizations:** Sample batches from training with ground truth annotations
- **Validation Predictions:** Model predictions on validation set with comparison to ground truth
- **Label Distribution:** Class balance and annotation statistics

## Model Analysis

### Weight Analysis
The `weight_analysis/` folder contains 89 visualizations tracking weight distributions across all model layers, including:
- Early layer feature extraction weights
- Mid-layer representation weights
- Detection head weights for multi-scale predictions
- Cross-variant (cv1, cv2) attention weights

### Layer Analysis
The `layer_analysis/` folder provides 89 feature map visualizations showing:
- Activation patterns at different network depths
- Feature pyramid representations
- Spatial attention across detection scales
- Layer-wise feature evolution

## Contributing

Contributions to the detection module are welcome. Please ensure that any pull requests maintain code quality and include appropriate documentation.

## License

The detection code in this repository is open-source. The prediction algorithm is proprietary and under patent application.

```

## Disclaimer

This project is under active development. Model performance metrics and configurations are subject to change as hyperparameter optimization continues.

## Contact

For questions regarding the detection implementation, please open an issue in this repository.

---

**Last Updated:** October 2025
