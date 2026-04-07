# ============================================================
# HealthBot-CRS | Step 6b: Error Analysis
# Misclassification deep-dive on best classifier
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 60)
print(" HealthBot-CRS | Error Analysis")
print("=" * 60)

# ============================================================
# 1. LOAD DATA & REBUILD FEATURES (same as 06_evaluation.py)
# ============================================================
print("\n[1/5] Loading data and rebuilding features...")

df         = pd.read_csv('data/healthbot_clustered.csv')
embeddings = np.load('outputs/results/sbert_embeddings.npy')

le = LabelEncoder()
df['intent_encoded'] = le.fit_transform(df['intent_category'])
intent_names = list(le.classes_)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                        min_df=2, max_df=0.95,
                        stop_words='english', sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(df['input_clean']).toarray()
X = np.hstack([tfidf_matrix, embeddings])
y = df['intent_encoded'].values

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)

# Load best classifier (try each, pick highest F1)
clf_models = {}
for name in ['Logistic_Regression','Random_Forest','SVM',
             'Gradient_Boosting','MLP']:
    path = f'outputs/models/{name}.pkl'
    if os.path.exists(path):
        clf_models[name.replace("_"," ")] = joblib.load(path)

best_name, best_f1, best_pred = None, -1, None
for name, model in clf_models.items():
    pred = model.predict(X_test)
    f1   = f1_score(y_test, pred, average='weighted')
    if f1 > best_f1:
        best_name, best_f1, best_pred, best_model = name, f1, pred, model

print(f" Best classifier : {best_name}  (F1={best_f1:.4f})")

# ============================================================
# 2. IDENTIFY ERRORS
# ============================================================
print("\n[2/5] Identifying misclassifications...")

error_mask   = best_pred != y_test
error_idx    = np.where(error_mask)[0]          # positions in test split
orig_idx     = idx_test[error_mask]             # original df indices
y_true_err   = y_test[error_mask]
y_pred_err   = best_pred[error_mask]

error_df = pd.DataFrame({
    'orig_index'     : orig_idx,
    'input'          : df.iloc[orig_idx]['input'].values,
    'true_intent'    : [intent_names[i] for i in y_true_err],
    'pred_intent'    : [intent_names[i] for i in y_pred_err],
    'input_length'   : df.iloc[orig_idx]['input_length'].values,
    'entity_count'   : df.iloc[orig_idx]['entity_count'].values,
})
error_df['confusion_pair'] = (error_df['true_intent'] + ' → '
                               + error_df['pred_intent'])

total      = len(y_test)
n_errors   = len(error_df)
error_rate = n_errors / total

print(f" Test samples    : {total}")
print(f" Errors          : {n_errors}  ({error_rate*100:.1f}%)")

# ============================================================
# 3. CONFIDENCE ANALYSIS
# ============================================================
print("\n[3/5] Analysing prediction confidence...")

proba = best_model.predict_proba(X_test)
max_conf     = proba.max(axis=1)
conf_correct = max_conf[~error_mask]
conf_wrong   = max_conf[error_mask]

# Low-confidence errors (model uncertain)
low_conf_threshold = 0.5
low_conf_errors = error_df[max_conf[error_mask] < low_conf_threshold]
print(f" Low-confidence errors (<{low_conf_threshold}) : {len(low_conf_errors)}")
print(f" Avg confidence — correct : {conf_correct.mean():.3f}")
print(f" Avg confidence — wrong   : {conf_wrong.mean():.3f}")

# ============================================================
# 4. TOP CONFUSION PAIRS
# ============================================================
print("\n[4/5] Top confusion pairs...")

top_pairs = (error_df['confusion_pair']
             .value_counts()
             .head(10)
             .reset_index())
top_pairs.columns = ['Confusion Pair', 'Count']
print(top_pairs.to_string(index=False))

# Per-intent error rate
intent_errors = error_df['true_intent'].value_counts()
intent_total  = pd.Series(
    [sum(y_test == i) for i in range(len(intent_names))],
    index=intent_names
)
intent_error_rate = (intent_errors / intent_total).fillna(0).sort_values(ascending=False)

# ============================================================
# 5. VISUALIZATIONS
# ============================================================
print("\n[5/5] Generating error analysis plots...")

fig = plt.figure(figsize=(22, 16))
fig.suptitle(f'HealthBot-CRS | Error Analysis — {best_name}',
             fontsize=17, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Full Confusion Matrix ──
ax1 = fig.add_subplot(gs[0, :2])
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=len(intent_names) <= 12, fmt='d', cmap='Blues',
            ax=ax1, xticklabels=intent_names, yticklabels=intent_names,
            cbar_kws={'shrink': 0.8}, linewidths=0.3)
ax1.set_title('Full Confusion Matrix', fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=7)
ax1.tick_params(axis='y', rotation=0,  labelsize=7)
ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')

# ── Plot 2: Per-intent error rate ──
ax2 = fig.add_subplot(gs[0, 2])
colors = ['#E74C3C' if v > 0.3 else '#F39C12' if v > 0.15 else '#2ECC71'
          for v in intent_error_rate.values]
ax2.barh(intent_error_rate.index, intent_error_rate.values,
         color=colors, edgecolor='white', alpha=0.9)
ax2.axvline(error_rate, color='navy', linestyle='--',
            label=f'Overall {error_rate*100:.1f}%')
ax2.set_title('Per-Intent Error Rate', fontweight='bold')
ax2.set_xlabel('Error Rate'); ax2.legend(fontsize=8)
ax2.tick_params(axis='y', labelsize=7)

# ── Plot 3: Top-10 confusion pairs ──
ax3 = fig.add_subplot(gs[1, :2])
ax3.barh(top_pairs['Confusion Pair'][::-1],
         top_pairs['Count'][::-1],
         color='#E74C3C', edgecolor='white', alpha=0.85)
for i, v in enumerate(top_pairs['Count'][::-1]):
    ax3.text(v + 0.3, i, str(v), va='center', fontsize=9)
ax3.set_title('Top-10 Misclassification Pairs (True → Predicted)',
              fontweight='bold')
ax3.set_xlabel('Count')
ax3.tick_params(axis='y', labelsize=8)

# ── Plot 4: Confidence distribution ──
ax4 = fig.add_subplot(gs[1, 2])
ax4.hist(conf_correct, bins=30, alpha=0.7, color='#2ECC71',
         label=f'Correct (n={len(conf_correct)})', density=True)
ax4.hist(conf_wrong,   bins=30, alpha=0.7, color='#E74C3C',
         label=f'Wrong (n={len(conf_wrong)})',   density=True)
ax4.axvline(low_conf_threshold, color='navy', linestyle='--',
            label=f'Threshold={low_conf_threshold}')
ax4.set_title('Prediction Confidence Distribution', fontweight='bold')
ax4.set_xlabel('Max Softmax Confidence')
ax4.set_ylabel('Density'); ax4.legend(fontsize=8)

# ── Plot 5: Error length vs confidence scatter ──
ax5 = fig.add_subplot(gs[2, 0])
sc = ax5.scatter(error_df['input_length'], conf_wrong,
                 c=y_true_err, cmap='tab10', alpha=0.5, s=20)
ax5.axhline(low_conf_threshold, color='red', linestyle='--', linewidth=1)
ax5.set_title('Error: Input Length vs Confidence', fontweight='bold')
ax5.set_xlabel('Input Length'); ax5.set_ylabel('Confidence')

# ── Plot 6: Error entity_count distribution ──
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(df['entity_count'].values[idx_test[~error_mask]],
         bins=20, alpha=0.7, color='#2ECC71',
         label='Correct', density=True)
ax6.hist(error_df['entity_count'],
         bins=20, alpha=0.7, color='#E74C3C',
         label='Wrong', density=True)
ax6.set_title('Entity Count: Correct vs Misclassified', fontweight='bold')
ax6.set_xlabel('Medical Entity Count'); ax6.set_ylabel('Density')
ax6.legend(fontsize=8)

# ── Plot 7: Sample error table ──
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')
sample_errors = error_df.sort_values('input_length').head(6)
rows = [[row['input'][:40]+'…', row['true_intent'][:12],
         row['pred_intent'][:12]]
        for _, row in sample_errors.iterrows()]
tbl = ax7.table(
    cellText=rows,
    colLabels=['Input (truncated)', 'True', 'Predicted'],
    loc='center', cellLoc='left'
)
tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#FADBD8')
ax7.set_title('Sample Misclassified Queries', fontweight='bold', pad=12)

plt.savefig('outputs/plots/06b_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Saved → outputs/plots/06b_error_analysis.png")

# ============================================================
# SAVE ERROR CSV
# ============================================================
error_df.to_csv('outputs/results/06b_error_analysis.csv', index=False)
print(" Saved → outputs/results/06b_error_analysis.csv")

print("\n" + "="*60)
print(" ERROR ANALYSIS COMPLETE")
print("="*60)
print(f" Total errors     : {n_errors} / {total}  ({error_rate*100:.1f}%)")
print(f" Low-conf errors  : {len(low_conf_errors)} (conf < {low_conf_threshold})")
print(f" Avg conf correct : {conf_correct.mean():.3f}")
print(f" Avg conf wrong   : {conf_wrong.mean():.3f}")
print(f"\n Top confusion pair: {top_pairs.iloc[0]['Confusion Pair']}  ({top_pairs.iloc[0]['Count']} times)")
print(f"\n Worst intent (highest error rate): {intent_error_rate.index[0]}")
print("="*60)
