# Road Decay Detection

**Comparative evaluation of YOLO architectures for road damage detection and semi-automated dataset re-annotation.**  

---

## Motivation  
Road infrastructure monitoring is critical for sustainable urban development, yet manual inspection remains costly and inconsistent. **computer vision and deep learning** provide a scalable solution. This project investigates the performance of state-of-the-art YOLO models on road damage datasets and compares them to vision transformer models (ViTs) such as RT-DETR and RF-DETR.  

Beyond benchmarking, this work aims to:  
- Identify the most effective vision model for road decay detection.  
- **Auto-label and refine annotations** in existing datasets.  
- Provide insights through **layer-wise visualization and hyperparameter tuning**.  

---

## Research Questions  
1. Which vision model provides the best out-of-the-box performance on the **N-RDD2024 dataset**?  
2. Can **semi-automated annotation** improve the labeling quality of the **RDD2022 dataset**?  
3. How do layer-wise visualizations and weight analyses inform hyperparameter finetuning?  

---

## Experimental Workflow  
1. **Benchmarking**: Compare vision models (YOLOv8, YOLOv11, YOLO-NAS, RT-DETR, RF-DETR) on **N-RDD2024**.  
2. **Model Selection**: Identify the highest-performing variant based on mAP, precision, and recall.  
3. **Dataset Refinement**:  
   - Use auto-labeling pipelines to generate annotations for **RDD2022**.  
   - Apply semi-automated correction to ensure label consistency.  
4. **Retraining**: Train the selected vision model on the refined dataset.  
5. **Visualization & Analysis**:  
   - Inspect convolutional filters and activations.  
   - Analyze layer-wise weight distributions.  
   - Tune hyperparameters accordingly.  

---

## 📂 Repository Structure  


---

## 📦 Installation  
```bash
# Clone repository
git clone https://github.com/yourusername/road-decay-detection.git
cd road-decay-detection

# Create environment
conda create -n rdd python=3.10 -y
conda activate rdd

# Install dependencies
pip install -r requirements.txt
