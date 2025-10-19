#  CaptchaSolver — CAPTCHA Image-to-Text Recognition  

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.x-red?style=flat-square&logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-App_UI-brightgreen?style=flat-square&logo=streamlit)


## 🧭 Overview  

**CaptchSolver** is a deep learning–based OCR system that automatically decodes text from complex CAPTCHA images using a **CNN + Bi-LSTM + CTC** architecture.  
The project integrates preprocessing, training, evaluation, and deployment — culminating in a **Streamlit dashboard** for real-time predictions and visualization of OCR performance metrics.

## 🌐 Live Demo  

> Upload any CAPTCHA image to see decoded predictions, model confidence, and per-character analysis — all rendered on a modern Streamlit dashboard.

## ⚙️ Technical Stack  

| Category | Technologies |
|-----------|---------------|
| **Language** | Python 3.10 + |
| **Frameworks** | TensorFlow / Keras 3 |
| **Libraries** | NumPy, Pandas, Matplotlib, OpenCV |
| **OCR Logic** | CNN + Bi-LSTM + CTC Decoder |
| **Frontend / App** | Streamlit |
| **Metrics** | Accuracy (Exact Match), CER, WER, Confusion Matrix |
| **Visualization** | Matplotlib / Seaborn |
| **Deployment** | Streamlit Cloud / Local |


## 📦 Repository Structure  
```bash
CaptchSolver/
│
├── Datasets/                # Image datasets (raw & validation)
├── Models/                  # Trained model weights (.h5)
├── captcha_images_v2/       # Labeled CAPTCHA samples
├── imgs/                    # Visualization & EDA figures
├── script/                  # Installation helpers
├── captchasolver.ipynb      # Full training + evaluation notebook
├── app.py                   # Streamlit web app
├── requirements.txt         # Dependencies
└── README.md
```

## 🧮 Dataset Overview  

**Source:** A curated dataset of synthetic CAPTCHA images labeled via filenames (e.g., `4fp5g.png → "4fp5g"`).  
Images contain alphanumeric sequences over cluttered backgrounds with rotation and noise.

| Metric | Value |
|--------|--------|
| Total Images | **1 040** |
| Format | `.png` |
| Average Label Length | **5 characters** |
| Dimensions | **200 × 50 px** |
| Unique Characters | **19 letters + digits** |
| Label Origin | Derived from filenames |

### 📋 Data Integrity Check  

Total images: 1040
Unique stems (label candidates): 1040
Same label across multiple extensions: 0



## 📊 Exploratory Data Analysis  

### 1️⃣ Label Length Distribution  
![Label Length Distribution](imgsScs/label_length_dist.png)  
All labels contain exactly **5 characters**, simplifying sequence modeling and padding.


### 2️⃣ Sample CAPTCHA Grid  
![Sample Captchas](imgsScs/sample_captchas.png)  
The dataset shows mild skew, rotation, and overlapping strokes — ideal for real-world OCR robustness.

### 3️⃣ Image Dimensions Consistency  
Every image is precisely **200×50 px**, enabling fixed input resizing and CNN efficiency.

### 4️⃣ Character Frequency Distribution  
![Character Distribution](imgsScs/top40_characters.png)  
Top characters: `n(525)`, `4(284)`, `5(281)`, `m(275)`, `f(271)`, `g(271)`...  
Balanced representation avoids over-training on any specific class.

> ⚖️ **Balanced vocabularies** improve OCR generalization and decoding reliability.


## 🧠 Model Architecture  

**Architecture Summary**

| Component | Description |
|------------|-------------|
| **Feature Extractor** | Convolutional + BatchNorm + ReLU + MaxPool |
| **Sequence Modeler** | 2-layer Bidirectional LSTM |
| **Decoder** | CTC (Connectionist Temporal Classification) |
| **Loss** | CTC Loss |
| **Optimizer** | Adam (1e-4 lr) |


## 🧪 Evaluation Notebook Summary  

### 🧾 Validation Data Loading  

Loads pairs from `val.csv` or auto-labels from folder structure.  
Ensures (image_path, label) pairs are generated for validation.


### 🔍 Model Evaluation Workflow  
![Evaluation Steps](imgsScs/model_eval.png)

| Step | Operation | Description |
|------|------------|-------------|
| 1 | `predict_text()` | Runs inference on each image |
| 2 | `get_cer()` | Computes Character Error Rate |
| 3 | `pred == label` | Checks Exact Match accuracy |
| 4 | Record Results | Stores outputs in DataFrame |


### 📈 Model Metrics  
| Metric | Result |
|---------|--------|
| **Exact Match Accuracy** | 0.9327 (93.2 %) |
| **Average CER** | 0.0135 (1.35 %) |
| **Average WER** | 0.0153 (1.53 %) |

| Example | Label | Prediction | CER |
|----------|-------|-------------|-----|
| wgnwp | wgmwp | 0.2 |
| cwmny | cwnny | 0.2 |

> Minor errors arise from visually similar glyphs (`n ↔ m`, `g ↔ p`, `c ↔ e`).

---

### 🧩 Character-Level Error Analysis  
![Error Summary](imgsScs/error_summary.png)

**Aligned Character Pairs:** 519  

| Type | Example |
|------|----------|
| Substitutions | `c → e`, `n → m`, `4 → d` |
| Insertions | `+n` |
| Deletions | `–m` |


### 🎛 Confusion Matrix  
![Confusion Matrix](imgsScs/confusion_matrix.png)  
Diagonal (yellow) = correct predictions   
Off-diagonal (purple) = shape confusions  
> Sparse off-diagonals ≈ **99 % per-character accuracy**


### 📜 Per-Character Classification Report  
![Classification Report](imgsScs/classification_report.png)

| Metric | Macro Avg | Weighted Avg |
|---------|-----------|--------------|
| Precision | 0.989 | 0.989 |
| Recall | 0.988 | 0.988 |
| F1-Score | 0.988 | 0.988 |

> Minor drops for `c`, `e`, `g`, `m/n` — typical OCR confusions.  
> Overall OCR fidelity ≈ **98.8 % per character**

---

### 🧮 WER (Word Error Rate)  
```bash
WER = (Substitutions + Insertions + Deletions)/(Length(GroundTruth)
```
**Average WER:** 0.01538 → ~1.5 % token error.  

## 🏁 Final Summary  

| Aspect | Description | Key Takeaway |
|---------|--------------|--------------|
| **Dataset** | 1 040 CAPTCHAs (200×50 px) | Uniform + balanced |
| **Architecture** | CNN + Bi-LSTM + CTC | Sequential OCR pipeline |
| **Accuracy** | 93 % exact match | Strong sequence decoding |
| **Per-Char F1** | 98.8 % | Robust character recognition |
| **CER / WER** | 0.013 / 0.015 | Minimal textual distortion |
| **Top Confusions** | n↔m, c↔e, g↔p | Shape similarity errors |
| **Deployment** | Streamlit App | Real-time visual OCR demo |


## 🧰 Local Deployment  

### 1️⃣ Clone & Setup  
```bash
git clone https://github.com/<your-username>/CaptchSolver.git  
cd CaptchSolver  
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt
```

2️⃣ Run App

streamlit run app.py



# 🧑‍💻 Contributor
	•	Tiyani Gurusinghe — Developer


📘 CaptchSolver demonstrates an end-to-end OCR workflow — from CAPTCHA preprocessing and EDA to model training and Streamlit deployment — achieving human-level recognition accuracy  with interpretability metrics (CER, WER, confusion heatmaps).
