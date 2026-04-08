---
title: Healthbot Crs
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏥 HealthBot-CRS — Clinical Response System

> **Group:** DSA_202101_8  
> **Live Demo:** [https://sugan04-healthbot-crs.hf.space](https://sugan04-healthbot-crs.hf.space)  
> ⚠️ *Research Prototype — Not a substitute for professional medical advice.*

---

## 📌 Problem Statement

Patients often struggle to access quick, reliable answers to health-related questions outside of clinical settings. HealthBot-CRS addresses this by building an intelligent, ML-powered Clinical Response System that classifies patient intent, retrieves the most relevant medical response, and delivers it through a conversational chatbot interface — complete with multilingual text-to-speech support.

---

## 🗂️ Dataset

- **Source:** [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
- **Size:** 100,000 patient-doctor Q&A pairs
- **Working sample:** 5,000 rows (stratified, for development)
- **Intent categories:** `emergency`, `symptom_inquiry`, `medication_inquiry`, `diagnostics`, `lifestyle`, `mental_health`, `general_inquiry`

---

## 🏗️ System Architecture

```
Raw Data (100k Q&A)
      ↓
01. Data Loading & Cleaning (NLTK, Regex)
      ↓
02. Feature Engineering (TF-IDF + SBERT + SpaCy NER)
      ↓
03. Clustering (K-Means on SBERT embeddings)
      ↓
04. Classification (LR, RF, SVM, GB, MLP)
      ↓
05. Recommender System (SVD, NMF, KNN — Surprise-style hybrid)
      ↓
06. Evaluation & Error Analysis (F1, ROC, Confusion Matrix)
      ↓
08. Flask API + Chat UI (deployed on Hugging Face Spaces)
```

---

## ⚙️ Tech Stack & Libraries

| Category | Libraries |
|---|---|
| Data Processing | `pandas`, `numpy`, `nltk` |
| NLP & Features | `scikit-learn` (TF-IDF), `sentence-transformers` (SBERT), `spacy` (NER) |
| Clustering | `scikit-learn` KMeans |
| Classification | `scikit-learn` (LR, RF, SVM, GradientBoosting, MLP) |
| Recommender | `scikit-learn` (SVD/NMF/KNN matrix factorization) |
| Evaluation | `scikit-learn` (F1, ROC, confusion matrix) |
| Web API | `flask`, `flask-cors` |
| Deployment | Docker, Hugging Face Spaces |
| Frontend | Vanilla HTML/CSS/JS, Web Speech API (TTS) |

---

## 📊 ML Pipeline

### Step 1 — Data Loading (`01_data_loading.py`)
- Loaded 100k records from HuggingFace datasets
- Cleaned text: lowercased, removed URLs, emails, special characters
- Rule-based intent labelling across 7 categories
- Filtered low-quality rows (too short/long)
- Final sample: **5,000 rows** saved to `data/healthbot_sample.csv`

### Step 2 — Feature Engineering (`02_feature_engineering.py`)
- **TF-IDF:** 5,000 features, bigrams, sublinear TF weighting
- **SBERT:** `all-MiniLM-L6-v2` sentence embeddings (384 dimensions)
- **SpaCy NER:** Medical entity extraction (symptoms, medications, conditions)
- Combined feature matrix: TF-IDF + SBERT → input to classifiers

### Step 3 — Clustering (`03_clustering.py`)
- Applied **K-Means** on SBERT embeddings
- Optimal K selected via Elbow method and Silhouette score
- Cluster labels used as user profiles for the recommender system

### Step 4 — Classification (`04_classification.py`)
- Trained **5 classifiers** on combined TF-IDF + SBERT features:
  - Logistic Regression, Random Forest, SVM, Gradient Boosting, MLP
- Best model selected automatically at runtime by weighted F1 score

### Step 5 — Recommender System (`05_recommender.py`)
- Built implicit rating matrix from cluster × intent interactions
- Trained **SVD**, **NMF**, and **KNN** collaborative filtering models
- Hybrid scoring: `α × content-based + (1-α) × collaborative filtering`

### Step 6 — Evaluation (`06_evaluation.py`, `06b_error_analysis.py`)
- Metrics: Weighted F1, Precision, Recall, Accuracy
- Confusion matrix, ROC curves per intent class
- Error analysis: low-confidence predictions and misclassified intents

---

## 📈 Results

| Model | Weighted F1 |
|---|---|
| Logistic Regression | ~0.82 |
| SVM | ~0.84 |
| Gradient Boosting | ~0.83 |
| Random Forest | ~0.79 |
| MLP | ~0.85 |

> Best model is auto-selected at runtime. Hybrid recommender combines SBERT cosine similarity with CF ratings for improved retrieval quality.

---

## 💡 Innovativeness

- **Hybrid CB+CF Retrieval:** Combines content-based SBERT similarity with collaborative filtering — most chatbots use only one approach
- **Multilingual TTS:** Real-time French translation + voice output via Web Speech API and MyMemory API
- **Live Deployment:** Fully containerized with Docker and hosted on Hugging Face Spaces
- **Clinical NER:** SpaCy-based medical entity extraction to enrich feature representation

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/sugan04/healthbot-crs
cd healthbot-crs

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run full ML pipeline
python 01_data_loading.py
python 02_feature_engineering.py
python 03_clustering.py
python 04_classification.py
python 05_recommender.py
python 06_evaluation.py

# 4. Launch chatbot API
python 08_chatbot.py --api

# 5. Open in browser
# http://localhost:7860
```

---

## 📁 File Structure

```
healthbot-crs/
├── 01_data_loading.py          # Data loading & cleaning
├── 02_feature_engineering.py   # TF-IDF, SBERT, SpaCy NER
├── 03_clustering.py            # K-Means clustering
├── 04_classification.py        # 5 classifier models
├── 05_recommender.py           # SVD/NMF/KNN recommender
├── 06_evaluation.py            # Metrics & ROC curves
├── 06b_error_analysis.py       # Error analysis
├── 08_chatbot.py               # Flask API + CLI chatbot
├── HealthBot_CRS_Chat.html     # Chat UI
├── HealthBot_CRS_Demo.html     # Landing page
├── requirements.txt
├── Dockerfile
├── data/
│   ├── healthbot_sample.csv
│   └── healthbot_clustered.csv
└── outputs/
    ├── models/                 # Trained .pkl files
    ├── plots/                  # Visualizations
    └── results/                # Embeddings & matrices
```

---

## 👥 Group

**DSA_202101_8**  
Course: Data Science & Analytics  
Deployment: [https://sugan04-healthbot-crs.hf.space](https://sugan04-healthbot-crs.hf.space)
