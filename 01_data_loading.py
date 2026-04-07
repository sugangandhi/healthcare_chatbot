# ============================================================
# HealthBot-CRS | Step 1: Data Loading & Cleaning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import nltk
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# ── Download NLTK resources ──────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Create output directories ────────────────────────────────
os.makedirs('data', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 60)
print("  HealthBot-CRS | Data Loading & Cleaning")
print("=" * 60)

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("\n[1/6] Loading HealthCareMagic-100k dataset...")

dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")
df = pd.DataFrame(dataset['train'])

print(f"     Dataset loaded: {len(df):,} rows")
print(f"     Columns: {list(df.columns)}")
print(f"\n     Sample row:")
print(df.iloc[0])

# ============================================================
# 2. INITIAL EXPLORATION
# ============================================================
print("\n[2/6] Initial exploration...")

print(f"\n     Shape         : {df.shape}")
print(f"     Missing values:\n{df.isnull().sum()}")
print(f"\n     Data types:\n{df.dtypes}")

# Rename columns for clarity
df.columns = ['instruction', 'input', 'output']
print(f"\n     Renamed columns to: {list(df.columns)}")

# ============================================================
# 3. TEXT CLEANING
# ============================================================
print("\n[3/6] Cleaning text...")

def clean_text(text):
    """Clean medical text while preserving clinical terms."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove special characters but keep medical punctuation (. , - /)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\-\/]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_intent_category(text):
    """Rule-based intent classification for initial labeling."""
    text_lower = text.lower()
    if any(w in text_lower for w in ['emergency', 'severe', 'chest pain', 'can\'t breathe',
                                      'unconscious', 'bleeding', 'stroke', 'heart attack', 'dying']):
        return 'emergency'
    elif any(w in text_lower for w in ['symptom', 'feeling', 'experiencing', 'having', 'suffering']):
        return 'symptom_inquiry'
    elif any(w in text_lower for w in ['medication', 'drug', 'medicine', 'pill', 'dose', 'tablet']):
        return 'medication_inquiry'
    elif any(w in text_lower for w in ['diet', 'food', 'exercise', 'weight', 'lifestyle']):
        return 'lifestyle'
    elif any(w in text_lower for w in ['test', 'diagnosis', 'lab', 'blood', 'scan', 'mri', 'xray']):
        return 'diagnostic'
    elif any(w in text_lower for w in ['mental', 'anxiety', 'depression', 'stress', 'psychological']):
        return 'mental_health'
    else:
        return 'general_inquiry'

# Apply cleaning
print("     Cleaning patient queries (input)...")
df['input_clean']  = df['input'].apply(clean_text)
print("     Cleaning doctor responses (output)...")
df['output_clean'] = df['output'].apply(clean_text)

# Add derived features
print("     Extracting features...")
df['input_length']    = df['input_clean'].apply(lambda x: len(x.split()))
df['output_length']   = df['output_clean'].apply(lambda x: len(x.split()))
df['intent_category'] = df['input_clean'].apply(extract_intent_category)

print(f"\n     Intent distribution:")
print(df['intent_category'].value_counts())

# ============================================================
# 4. FILTERING & QUALITY CONTROL
# ============================================================
print("\n[4/6] Filtering low quality samples...")

original_len = len(df)

# Remove empty rows
df = df[df['input_clean'].str.len() > 10]
df = df[df['output_clean'].str.len() > 20]

# Remove very short inputs (less than 5 words)
df = df[df['input_length'] >= 5]

# Remove very long outliers (over 500 words in input)
df = df[df['input_length'] <= 500]

# Reset index
df = df.reset_index(drop=True)

print(f"     Removed {original_len - len(df):,} low quality rows")
print(f"     Remaining: {len(df):,} rows")

# ============================================================
# 5. SAVE CLEANED DATA
# ============================================================
print("\n[5/6] Saving cleaned dataset...")

df.to_csv('data/healthbot_clean.csv', index=False)
print(f"     Saved → data/healthbot_clean.csv")

# Save a smaller sample for faster development
df_sample = df.sample(n=5000, random_state=42).reset_index(drop=True)
df_sample.to_csv('data/healthbot_sample.csv', index=False)
print(f"     Saved → data/healthbot_sample.csv (5,000 rows for dev)")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n[6/6] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('HealthBot-CRS | Dataset Overview', fontsize=16, fontweight='bold', y=1.01)

# Plot 1 — Intent Distribution
intent_counts = df['intent_category'].value_counts()
colors = ['#E74C3C','#E67E22','#3498DB','#2ECC71','#9B59B6','#1ABC9C','#F39C12']
axes[0,0].bar(intent_counts.index, intent_counts.values, color=colors[:len(intent_counts)])
axes[0,0].set_title('Intent Category Distribution', fontweight='bold')
axes[0,0].set_xlabel('Category')
axes[0,0].set_ylabel('Count')
axes[0,0].tick_params(axis='x', rotation=30)
for i, v in enumerate(intent_counts.values):
    axes[0,0].text(i, v + 100, f'{v:,}', ha='center', fontsize=9)

# Plot 2 — Input Length Distribution
axes[0,1].hist(df['input_length'], bins=50, color='#3498DB', edgecolor='white', alpha=0.8)
axes[0,1].set_title('Patient Query Length Distribution', fontweight='bold')
axes[0,1].set_xlabel('Word Count')
axes[0,1].set_ylabel('Frequency')
axes[0,1].axvline(df['input_length'].mean(), color='red', linestyle='--',
                   label=f"Mean: {df['input_length'].mean():.0f} words")
axes[0,1].legend()

# Plot 3 — Output Length Distribution
axes[1,0].hist(df['output_length'], bins=50, color='#2ECC71', edgecolor='white', alpha=0.8)
axes[1,0].set_title('Doctor Response Length Distribution', fontweight='bold')
axes[1,0].set_xlabel('Word Count')
axes[1,0].set_ylabel('Frequency')
axes[1,0].axvline(df['output_length'].mean(), color='red', linestyle='--',
                   label=f"Mean: {df['output_length'].mean():.0f} words")
axes[1,0].legend()

# Plot 4 — Top 20 Most Common Words in Queries
stop_words = set(stopwords.words('english'))
all_words = ' '.join(df['input_clean'].sample(5000, random_state=42)).split()
filtered_words = [w for w in all_words if w not in stop_words and len(w) > 3]
word_freq = Counter(filtered_words).most_common(20)
words, counts = zip(*word_freq)
axes[1,1].barh(list(words)[::-1], list(counts)[::-1], color='#9B59B6', alpha=0.8)
axes[1,1].set_title('Top 20 Words in Patient Queries', fontweight='bold')
axes[1,1].set_xlabel('Frequency')

plt.tight_layout()
plt.savefig('outputs/plots/01_data_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("     Saved → outputs/plots/01_data_overview.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  DATA LOADING & CLEANING COMPLETE")
print("=" * 60)
print(f"\n  Total records    : {len(df):,}")
print(f"  Intent categories: {df['intent_category'].nunique()}")
print(f"  Avg query length : {df['input_length'].mean():.1f} words")
print(f"  Avg response len : {df['output_length'].mean():.1f} words")
print(f"\n  Files saved:")
print(f"  → data/healthbot_clean.csv")
print(f"  → data/healthbot_sample.csv")
print(f"  → outputs/plots/01_data_overview.png")
print("\n  Next step: Run 02_feature_engineering.py")
print("=" * 60)