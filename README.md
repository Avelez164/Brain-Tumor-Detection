# CPSC481-Project
# NeuroScanAI Brain Tumor Detection & Summary

**Authors**: Jenny Phan, Antonio Velez  

---

## 📋 Overview

NeuroScanAI is a Flask web application that:

1. **Randomly samples** an MRI scan from one of four classes (glioma, meningioma, no tumor, pituitary).  
2. Runs a **CNN** we trained to predict its class.  
3. Generates a **template‐based report summary** for the predicted class.  
4. Displays both the **ground-truth** and **predicted** labels plus the summary in a clean Bootstrap UI.

---

## 🚀 Quick Start

### 1. Clone this repository
```bash
git clone https://github.com/Avelez164/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```````

## 2. Install dependencies
```bash
pip install --upgrade pip

pip install -r requirements.txt
```````


## 3. ▶️ Run the Application
From the project root, start the Flask server:
```bash
python3 app.py
```````

📊 Performance & Notes
1. Backbone: ResNet50 pre-trained on ImageNet, fine-tuned on 7,023 MRI scans.

2. Accuracy: ~70–75% on held-out test data → you will see occasional misclassifications.

3. NLP Summary: Template-based, zero external model → runs in <2 GB RAM.

Improvement Ideas:

1. Unfreeze additional ResNet layers + use a lower learning rate

2. Add a learning-rate scheduler (e.g. ReduceLROnPlateau) + EarlyStopping

3. Increase data augmentation (elastic, shear, flips, brightness)

4. Swap to a lighter backbone (EfficientNetB0) or ensemble multiple models



![Screenshot 2025-05-16 082606](https://github.com/user-attachments/assets/d51e726b-5f9d-4680-bf2e-b12a4aaaa4e0)
