# 🩺 Heart Disease AI — Clinical Risk Prediction Engine & Diagnostic Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Algorithm-CatBoost%20Classifier-yellow?logo=catboost&logoColor=white)](https://catboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/ML%20Library-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.27%25-brightgreen)](https://github.com/JaniduKasuntha/Heart-Disease-AI)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9995-success)](https://github.com/JaniduKasuntha/Heart-Disease-AI)
[![Dataset](https://img.shields.io/badge/Dataset-70%2C000%20Records-blueviolet)](https://github.com/JaniduKasuntha/Heart-Disease-AI)

> **High-precision cardiovascular disease risk prediction engine and interactive clinical decision-support system powered by Principal Component Analysis (PCA) and hyperparameter-tuned CatBoost gradient boosting, achieving 99.27% diagnostic accuracy across 70,000 patient records.**

---

## 📌 Executive Summary

Cardiovascular diseases (CVDs) remain the leading cause of morbidity and mortality worldwide. Early detection and proactive lifestyle intervention significantly improve patient prognosis. 

**Heart Disease AI** is an end-to-end machine learning system developed to stratify patient risk with exceptional reliability. The solution ingests **18 clinical, lifestyle, and biometric factors** from a comprehensive 70,000-patient cohort, conducts robust feature scaling and orthogonal dimensionality reduction via **Principal Component Analysis (PCA)**, and executes classification through a fine-tuned **CatBoost** ensemble model. The system is packaged with two **Streamlit-powered clinical dashboards** for both rapid single-sample inference and rich visual patient risk monitoring.

---

## 📊 Key Performance Metrics

The model was rigorously validated against a held-out test split, comparing both baseline and grid-search optimized gradient boosted trees:

| Metric | Baseline CatBoost | Tuned CatBoost (Optimal) | Clinical Impact |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **99.27%** | **99.27%** | Near-zero misclassification rate across diverse patient profiles |
| **Precision** | 99.38% | **99.37%** | Minimizes false positives, preventing unwarranted patient distress |
| **Recall (Sensitivity)** | 99.16% | **99.17%** | Critical: captures 99.17% of true positive cardiac risk cases |
| **F1-Score** | 99.27% | **99.27%** | Harmonic balance between sensitivity and precision |
| **ROC-AUC** | 0.99948 | **0.99942** | Outstanding discriminatory power across all decision thresholds |

### Optimal Hyperparameters (via GridSearchCV)
```python
{
    'iterations': 100,
    'depth': 8,
    'learning_rate': 0.1,
    'l2_leaf_reg': 5,
    'loss_function': 'Logloss',
    'random_seed': 42
}
```

---

## 🏗️ System Architecture & ML Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Raw Clinical Data                                │
│                     (70,000 Patient Records, 18 Features)                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Data Preprocessing & Transformation                      │
│   • Missing value imputation & Outlier audit (z-score filtering)            │
│   • Standardization & Scaling via StandardScaler                            │
│   • Dimensionality Reduction via Principal Component Analysis (PCA)         │
│     (De-correlating features & retaining dominant variance vectors)         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CatBoost Gradient Boosted Ensemble                       │
│   • GridSearch-optimized decision trees with symmetric tree growth          │
│   • L2 Leaf Regularization mitigating overfitting                           │
│   • Probability Calibration & Confidence Stratification                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
┌──────────────────────────────────┐      ┌──────────────────────────────────┐
│      app.py (Clinical Tool)      │      │  app_Binary.py (Executive UI)    │
│  • Rapid diagnostic inference    │      │  • Dynamic glowing risk meter    │
│  • Probability breakdown         │      │  • Interactive age sliders       │
│  • Diagnostic recommendation     │      │  • Detailed clinical guidance    │
└──────────────────────────────────┘      └──────────────────────────────────┘
```

---

## 🧬 Feature Engineering & Clinical Indicators

The model analyzes 18 clinical attributes encompassing subjective symptoms, chronic indicators, and lifestyle markers:

| # | Feature | Type | Clinical Significance |
|---|---|---|---|
| 1 | `Chest_Pain` | Binary (0/1) | Angina pectoris or cardiac ischemia indicator |
| 2 | `Shortness_of_Breath` | Binary (0/1) | Dyspnea associated with congestive heart failure |
| 3 | `Fatigue` | Binary (0/1) | Systemic exhaustion caused by reduced cardiac output |
| 4 | `Palpitations` | Binary (0/1) | Arrhythmia, tachycardia, or ectopic beats |
| 5 | `Dizziness` | Binary (0/1) | Hypoperfusion or transient cerebral ischemia |
| 6 | `Swelling` | Binary (0/1) | Peripheral edema in legs/ankles indicating fluid retention |
| 7 | `Pain_Arms_Jaw_Back` | Binary (0/1) | Classic radiating ischemic pain pattern |
| 8 | `Cold_Sweats_Nausea` | Binary (0/1) | Autonomic response during acute coronary events |
| 9 | `High_BP` | Binary (0/1) | Chronic hypertension leading to arterial stiffening |
| 10 | `High_Cholesterol` | Binary (0/1) | Hyperlipidemia and atherosclerotic plaque formation |
| 11 | `Diabetes` | Binary (0/1) | Diabetic cardiomyopathy and microvascular damage |
| 12 | `Smoking` | Binary (0/1) | Endothelial dysfunction and oxidative arterial stress |
| 13 | `Obesity` | Binary (0/1) | Elevated BMI ($\ge 30$) placing chronic strain on myocardium |
| 14 | `Sedentary_Lifestyle`| Binary (0/1) | Insufficient aerobic activity and decreased vascular elasticity|
| 15 | `Family_History` | Binary (0/1) | Genetic predisposition to early cardiovascular disease |
| 16 | `Chronic_Stress` | Binary (0/1) | Elevated cortisol/catecholamines inducing vasoconstriction |
| 17 | `Gender` | Binary (0/1) | Biological sex baseline risk variation |
| 18 | `Age` | Continuous | Age in years (0 - 100) — dominant non-modifiable risk factor |

---

## 💻 Interactive Dashboards

### 1. `app.py` — Streamlined Clinical Diagnostic Tool
A rapid-entry clinical interface built with Streamlit. Allows cardiologists and general practitioners to toggle symptoms, input patient vitals, and generate instant risk predictions with exact probability percentages.

### 2. `app_Binary.py` — Premium Medical Dashboard
An advanced, patient-facing visual dashboard featuring:
- **Dynamic Glowing Risk Gauge**: Animated CSS glow bar that transitions smoothly between safe and high-risk visual thresholds.
- **Interactive Patient Sliders**: Real-time evaluation of how aging and lifestyle factors impact the cardiac risk profile.
- **Card-Based UI Architecture**: Clean medical informatics layout with comprehensive feature descriptions.

---

## 🛠️ Technology Stack

| Domain | Technology / Framework |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Machine Learning** | CatBoost (`CatBoostClassifier`), Scikit-Learn (`StandardScaler`, `PCA`, `GridSearchCV`) |
| **Data Processing** | Pandas, NumPy, Scipy |
| **Web UI / Deployment** | Streamlit |
| **Serialization** | Joblib (`scaler.pkl`, `pca.pkl`, `catboost_model.cbm`) |
| **Visualization & EDA** | Matplotlib, Seaborn |

---

## 📂 Repository Structure

```
Heart Disease AI/
├── 2025-Y2-S1-MLB-B6G1-05-PGNO-105_Pipeline_Updated.ipynb  # Comprehensive EDA, RFE, and feature engineering notebook
├── heart_disease_risk_CatBoost_model.ipynb                  # CatBoost development and parameter tuning notebook
├── IT24101236_CatBoost.ipynb                               # Research notebook with GridSearchCV & metrics
├── app.py                                                   # Streamlit clinical risk predictor
├── app_Binary.py                                            # Advanced Streamlit medical dashboard
├── catboost_model.cbm                                       # Serialized trained CatBoost model
├── scaler.pkl                                               # Trained StandardScaler artifact
├── pca.pkl                                                  # Trained PCA transformation artifact
├── heart_disease_risk.csv                                   # 70,000-record cardiac risk dataset
├── PCA_Final_Output.csv                                     # Pre-computed PCA orthogonal feature matrix
├── .gitattributes                                           # Git LFS / attribute declarations
└── README.md                                                # Documentation
```

---

## 🚀 Getting Started & Local Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your machine
- Git installed

### 2. Clone Repository
```bash
git clone https://github.com/JaniduKasuntha/Heart-Disease-AI.git
cd Heart-Disease-AI
```

### 3. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn catboost joblib scipy matplotlib seaborn
```

### 5. Launch the Application

**Run the Standard Clinical Predictor:**
```bash
streamlit run app.py
```

**Run the Advanced Medical Dashboard:**
```bash
streamlit run app_Binary.py
```

Open your browser at `http://localhost:8501` to interact with the dashboard.

---

## 🧑‍💻 Author & Academic Attribution

- **Developer**: Janidu Kasuntha Rajapaksha
- **Student ID**: IT24101236
- **Academic Module**: Machine Learning / Artificial Intelligence (MLB)
- **GitHub Repository**: [https://github.com/JaniduKasuntha/Heart-Disease-AI](https://github.com/JaniduKasuntha/Heart-Disease-AI)

---

## ⚠️ Medical Disclaimer
*This software is an AI research prototype developed for educational and clinical decision-support purposes. It should not be used as a standalone diagnostic instrument without validation by a qualified medical professional.*
