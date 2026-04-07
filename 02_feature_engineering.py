# ============================================================
# HealthBot-CRS | Step 2: Text Feature Engineering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from collections import Counter

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 60)
print("  HealthBot-CRS | Text Feature Engineering")
print("=" * 60)

# ============================================================
# 1. LOAD CLEANED DATA
# ============================================================
print("\n[1/5] Loading cleaned data...")
df = pd.read_csv('data/healthbot_sample.csv')
print(f"     Loaded {len(df):,} rows from healthbot_sample.csv")
print(f"     Columns: {list(df.columns)}")

# ============================================================
# 2. TF-IDF FEATURES
# ============================================================
print("\n[2/5] Building TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),     # unigrams + bigrams
    min_df=2,
    max_df=0.95,
    stop_words='english',
    sublinear_tf=True       # log normalization
)

tfidf_matrix = tfidf.fit_transform(df['input_clean'])
print(f"     TF-IDF matrix shape : {tfidf_matrix.shape}")
print(f"     Vocabulary size     : {len(tfidf.vocabulary_):,}")

# Get top terms per intent
print("\n     Top TF-IDF terms per intent category:")
le = LabelEncoder()
df['intent_encoded'] = le.fit_transform(df['intent_category'])

for intent in df['intent_category'].unique():
    subset = df[df['intent_category'] == intent]['input_clean']
    if len(subset) < 5:
        continue
    tfidf_sub = TfidfVectorizer(max_features=8, stop_words='english', ngram_range=(1,2))
    tfidf_sub.fit(subset)
    terms = list(tfidf_sub.vocabulary_.keys())[:5]
    print(f"     {intent:22s}: {', '.join(terms)}")

# Save TF-IDF matrix
np.save('outputs/results/tfidf_matrix.npy', tfidf_matrix.toarray())
print(f"\n     Saved → outputs/results/tfidf_matrix.npy")

# ============================================================
# 3. SENTENCE EMBEDDINGS (SBERT)
# ============================================================
print("\n[3/5] Building SBERT sentence embeddings...")
print("     Loading model: all-MiniLM-L6-v2 (fast & efficient)...")

model = SentenceTransformer('all-MiniLM-L6-v2')

# Use sample for speed
texts = df['input_clean'].tolist()
print(f"     Encoding {len(texts):,} queries...")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f"     Embedding shape: {embeddings.shape}")
np.save('outputs/results/sbert_embeddings.npy', embeddings)
print(f"     Saved → outputs/results/sbert_embeddings.npy")

# ============================================================
# 4. NAMED ENTITY RECOGNITION (SpaCy)
# ============================================================
print("\n[4/5] Running Named Entity Recognition with SpaCy...")

try:
    nlp = spacy.load('en_core_web_sm')
except:
    print("     Downloading spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

# Medical keyword extraction (clinical NER)
MEDICAL_TERMS = {
    'symptoms':    ['pain', 'fever', 'cough', 'fatigue', 'nausea', 'vomiting',
                    'headache', 'dizziness', 'swelling', 'bleeding', 'rash',
                    'shortness of breath', 'chest pain', 'abdominal pain'],
    'medications': ['aspirin', 'ibuprofen', 'paracetamol', 'metformin',
                    'amoxicillin', 'antibiotic', 'insulin', 'steroid',
                    'antidepressant', 'painkiller', 'medication', 'drug'],
    'body_parts':  ['heart', 'lung', 'liver', 'kidney', 'brain', 'stomach',
                    'chest', 'abdomen', 'back', 'leg', 'arm', 'head', 'neck'],
    'conditions':  ['diabetes', 'hypertension', 'asthma', 'cancer', 'infection',
                    'allergy', 'depression', 'anxiety', 'arthritis', 'anemia']
}

def extract_medical_entities(text):
    """Extract medical entities from clinical text."""
    text_lower = str(text).lower()
    entities = {cat: [] for cat in MEDICAL_TERMS}
    for category, terms in MEDICAL_TERMS.items():
        for term in terms:
            if term in text_lower:
                entities[category].append(term)
    return entities

def count_entities(text):
    """Count total medical entities in text."""
    entities = extract_medical_entities(text)
    return sum(len(v) for v in entities.values())

# Apply to sample
print("     Extracting medical entities from queries...")
sample_ner = df['input_clean'].head(500).apply(extract_medical_entities)

# Aggregate entity counts
all_symptoms = []
all_medications = []
all_conditions = []

for ents in sample_ner:
    all_symptoms.extend(ents['symptoms'])
    all_medications.extend(ents['medications'])
    all_conditions.extend(ents['conditions'])

print(f"\n     Top symptoms mentioned    : {Counter(all_symptoms).most_common(5)}")
print(f"     Top medications mentioned : {Counter(all_medications).most_common(5)}")
print(f"     Top conditions mentioned  : {Counter(all_conditions).most_common(5)}")

# Add entity count as feature
df['entity_count'] = df['input_clean'].apply(count_entities)

# SpaCy NER on sample
print("\n     Running SpaCy NER on 200 samples...")
spacy_entities = []
for doc in nlp.pipe(df['input_clean'].head(200).tolist(), batch_size=50):
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    spacy_entities.append(ents)

all_spacy = [ent for doc in spacy_entities for ent in doc]
if all_spacy:
    entity_df = pd.DataFrame(all_spacy, columns=['text', 'label'])
    print(f"\n     SpaCy entity types found:")
    print(entity_df['label'].value_counts().head(8))

# Save updated dataframe
df.to_csv('data/healthbot_features.csv', index=False)
print(f"\n     Saved → data/healthbot_features.csv")

# ============================================================
# 5. VISUALIZATIONS
# ============================================================
print("\n[5/5] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('HealthBot-CRS | Feature Engineering', fontsize=16, fontweight='bold')

# Plot 1 — TF-IDF Dimensionality Reduction (SVD → 2D)
svd = TruncatedSVD(n_components=2, random_state=42)
tfidf_2d = svd.fit_transform(tfidf_matrix)

colors_map = {
    'emergency': '#E74C3C',
    'symptom_inquiry': '#E67E22',
    'medication_inquiry': '#3498DB',
    'diagnostic': '#9B59B6',
    'general_inquiry': '#95A5A6',
    'lifestyle': '#2ECC71',
    'mental_health': '#F39C12'
}

for intent in df['intent_category'].unique():
    mask = df['intent_category'] == intent
    axes[0,0].scatter(
        tfidf_2d[mask, 0], tfidf_2d[mask, 1],
        c=colors_map.get(intent, '#95A5A6'),
        label=intent, alpha=0.4, s=8
    )
axes[0,0].set_title('TF-IDF SVD 2D Projection by Intent', fontweight='bold')
axes[0,0].legend(fontsize=7, markerscale=2)
axes[0,0].set_xlabel('SVD Component 1')
axes[0,0].set_ylabel('SVD Component 2')

# Plot 2 — SBERT Embedding PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(embeddings)

for intent in df['intent_category'].unique():
    mask = df['intent_category'] == intent
    axes[0,1].scatter(
        emb_2d[mask, 0], emb_2d[mask, 1],
        c=colors_map.get(intent, '#95A5A6'),
        label=intent, alpha=0.4, s=8
    )
axes[0,1].set_title('SBERT Embeddings PCA 2D Projection', fontweight='bold')
axes[0,1].legend(fontsize=7, markerscale=2)
axes[0,1].set_xlabel('PCA Component 1')
axes[0,1].set_ylabel('PCA Component 2')

# Plot 3 — Top Medical Terms
top_symptoms = Counter(all_symptoms).most_common(10)
if top_symptoms:
    terms, counts = zip(*top_symptoms)
    axes[1,0].barh(list(terms)[::-1], list(counts)[::-1], color='#E74C3C', alpha=0.8)
    axes[1,0].set_title('Top 10 Symptoms in Dataset', fontweight='bold')
    axes[1,0].set_xlabel('Frequency')

# Plot 4 — Entity Count Distribution
axes[1,1].hist(df['entity_count'], bins=20, color='#9B59B6', edgecolor='white', alpha=0.8)
axes[1,1].set_title('Medical Entity Count per Query', fontweight='bold')
axes[1,1].set_xlabel('Number of Medical Entities')
axes[1,1].set_ylabel('Frequency')
axes[1,1].axvline(df['entity_count'].mean(), color='red', linestyle='--',
                   label=f"Mean: {df['entity_count'].mean():.1f}")
axes[1,1].legend()

plt.tight_layout()
plt.savefig('outputs/plots/02_feature_engineering.png', dpi=150, bbox_inches='tight')
plt.show()
print("     Saved → outputs/plots/02_feature_engineering.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  FEATURE ENGINEERING COMPLETE")
print("=" * 60)
print(f"\n  TF-IDF features  : {tfidf_matrix.shape[1]:,} dimensions")
print(f"  SBERT embeddings : {embeddings.shape[1]} dimensions")
print(f"  Avg entity count : {df['entity_count'].mean():.1f} per query")
print(f"\n  Files saved:")
print(f"  → data/healthbot_features.csv")
print(f"  → outputs/results/tfidf_matrix.npy")
print(f"  → outputs/results/sbert_embeddings.npy")
print(f"  → outputs/plots/02_feature_engineering.png")
print("\n  Next step: Run 03_clustering.py")
print("=" * 60)