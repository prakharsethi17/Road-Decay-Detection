# Road Decay Detection

# Road Decay Detection 🚧  

**Comparative evaluation of YOLO architectures for road damage detection and semi-automated dataset re-annotation.**  

---

## 🌍 Motivation  
Road infrastructure monitoring is critical for sustainable urban development, yet manual inspection remains costly and inconsistent. Leveraging **computer vision and deep learning**, this project investigates the performance of state-of-the-art YOLO models on road damage datasets.  

Beyond benchmarking, this work aims to:  
- Identify the most effective YOLO variant for road decay detection.  
- **Auto-label and refine annotations** in existing datasets.  
- Provide insights through **layer-wise visualization and hyperparameter tuning**.  

---

## 📊 Research Questions  
1. Which YOLO variant provides the best out-of-the-box performance on the **N-RDD2024 dataset**?  
2. Can **semi-automated annotation** improve the labeling quality of the **RDD2022 dataset**?  
3. How do layer-wise visualizations and weight analyses inform hyperparameter finetuning?  

---

## 🧪 Experimental Workflow  
1. **Benchmarking**: Compare YOLO models (YOLOv5, YOLOv7, YOLOv8, YOLO-NAS, etc.) on **N-RDD2024**.  
2. **Model Selection**: Identify the highest-performing variant based on mAP, precision, and recall.  
3. **Dataset Refinement**:  
   - Use auto-labeling pipelines to generate annotations for **RDD2022**.  
   - Apply semi-automated correction to ensure label consistency.  
4. **Retraining**: Train the selected YOLO model on the refined dataset.  
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
