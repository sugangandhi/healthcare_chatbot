# ============================================================
# HealthBot-CRS | Step 6: Final Evaluation & Report
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

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 60)
print(" HealthBot-CRS | Final Evaluation & Summary")
print("=" * 60)

# ============================================================
# 1. LOAD DATA & REBUILD FEATURE MATRIX (matches 04_classification.py)
# ============================================================
print("\n[1/5] Loading data and rebuilding feature matrix...")

df         = pd.read_csv('data/healthbot_clustered.csv')
embeddings = np.load('outputs/results/sbert_embeddings.npy')

le = LabelEncoder()
df['intent_encoded'] = le.fit_transform(df['intent_category'])
intent_names = list(le.classes_)

# Rebuild TF-IDF (same config as 02_feature_engineering.py)
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words='english',
    sublinear_tf=True
)
tfidf_matrix = tfidf.fit_transform(df['input_clean']).toarray()

# Concatenate TF-IDF (5000) + SBERT (384) = 5384 features
X = np.hstack([tfidf_matrix, embeddings])
y = df['intent_encoded'].values

print(f" Feature matrix shape : {X.shape}")   # (n, 5384)
print(f" Intent classes       : {len(intent_names)}")

# Train/test split — same seed as 04_classification.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 2. LOAD SAVED CLASSIFIERS & EVALUATE
# ============================================================
print("\n[2/5] Evaluating saved classifiers...")

clf_models = {}
for name in ['Logistic_Regression', 'Random_Forest', 'SVM',
             'Gradient_Boosting', 'MLP']:
    path = f'outputs/models/{name}.pkl'
    if os.path.exists(path):
        clf_models[name.replace("_", " ")] = joblib.load(path)

eval_results = {}
for name, model in clf_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    eval_results[name] = {'accuracy': acc, 'f1': f1, 'y_pred': y_pred}
    print(f"  {name:25s}  Acc={acc:.4f}  F1={f1:.4f}")

best_clf_name = max(eval_results, key=lambda x: eval_results[x]['f1'])
best_pred     = eval_results[best_clf_name]['y_pred']
print(f"\n  Best classifier: {best_clf_name}")

# ============================================================
# 3. LOAD RECOMMENDER METRICS
# ============================================================
print("\n[3/5] Loading recommender metrics...")

rec_df = pd.read_csv('outputs/results/05_recommender_metrics.csv')          if os.path.exists('outputs/results/05_recommender_metrics.csv')          else pd.DataFrame()

rec_metrics = {}
if not rec_df.empty:
    for _, row in rec_df.iterrows():
        m = row['model']; k = row['metric']; v = row['value']
        rec_metrics.setdefault(m, {})[k] = v

# ============================================================
# 4. VISUALIZATIONS
# ============================================================
print("\n[4/5] Generating final evaluation dashboard...")

fig = plt.figure(figsize=(22, 16))
fig.suptitle('HealthBot-CRS | Final Evaluation Dashboard',
             fontsize=19, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: Classifier Accuracy & F1 ──
ax1 = fig.add_subplot(gs[0, 0])
names = list(eval_results.keys())
accs  = [eval_results[n]['accuracy'] for n in names]
f1s   = [eval_results[n]['f1']       for n in names]
x, w  = np.arange(len(names)), 0.35
b1 = ax1.bar(x-w/2, accs, w, label='Accuracy', color='#3498DB', alpha=0.85)
b2 = ax1.bar(x+w/2, f1s,  w, label='F1',       color='#E74C3C', alpha=0.85)
for b in list(b1)+list(b2):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
             f'{b.get_height():.3f}', ha='center', fontsize=7, rotation=45)
ax1.set_xticks(x)
ax1.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
ax1.set_title('Classifier Accuracy & F1', fontweight='bold')
ax1.set_ylim(0, 1.15); ax1.legend(fontsize=8)

# ── Plot 2: Confusion Matrix (best model, top-10 intents) ──
ax2 = fig.add_subplot(gs[0, 1:])
cm = confusion_matrix(y_test, best_pred)
top10 = np.argsort(np.bincount(y_test))[-10:]
cm_sub = cm[np.ix_(top10, top10)]
labels = [intent_names[i][:12] for i in top10]
sns.heatmap(cm_sub, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=labels, yticklabels=labels,
            cbar_kws={'shrink': 0.7})
ax2.set_title(f'Confusion Matrix — {best_clf_name} (Top-10 Intents)',
              fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=7)
ax2.tick_params(axis='y', rotation=0,  labelsize=7)

# ── Plot 3: Per-intent F1 ──
ax3 = fig.add_subplot(gs[1, 0])
report = classification_report(y_test, best_pred,
                                target_names=intent_names, output_dict=True)
pf1 = {k: v['f1-score'] for k, v in report.items() if k in intent_names}
pf1_sorted = dict(sorted(pf1.items(), key=lambda x: x[1]))
colors = ['#E74C3C' if v < 0.5 else '#F39C12' if v < 0.75 else '#2ECC71'
          for v in pf1_sorted.values()]
ax3.barh(list(pf1_sorted.keys()), list(pf1_sorted.values()),
         color=colors, edgecolor='white', alpha=0.9)
ax3.axvline(0.75, color='navy', linestyle='--', linewidth=1,
            label='0.75 threshold')
ax3.set_title(f'Per-Intent F1 ({best_clf_name})', fontweight='bold')
ax3.set_xlabel('F1 Score'); ax3.legend(fontsize=8)
ax3.tick_params(axis='y', labelsize=7)

# ── Plot 4: CF Model RMSE & MAE ──
ax4 = fig.add_subplot(gs[1, 1])
cf_names = [k for k in rec_metrics if k in ['SVD','NMF','KNN']]
if cf_names:
    cf_rmses = [rec_metrics[k].get('RMSE', 0) for k in cf_names]
    cf_maes  = [rec_metrics[k].get('MAE',  0) for k in cf_names]
    x2, w2   = np.arange(len(cf_names)), 0.35
    b3 = ax4.bar(x2-w2/2, cf_rmses, w2, label='RMSE', color='#E74C3C', alpha=0.85)
    b4 = ax4.bar(x2+w2/2, cf_maes,  w2, label='MAE',  color='#3498DB', alpha=0.85)
    for b in list(b3)+list(b4):
        ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f'{b.get_height():.3f}', ha='center', fontsize=8)
    ax4.set_xticks(x2); ax4.set_xticklabels(cf_names)
ax4.set_title('CF Models — RMSE & MAE', fontweight='bold')
ax4.set_ylabel('Error'); ax4.legend(fontsize=8)

# ── Plot 5: Recommender Precision@k ──
ax5 = fig.add_subplot(gs[1, 2])
cb    = rec_metrics.get('Content-Based', {})
k_lbl = ['@1', '@3', '@5']
precs = [cb.get('Precision@1', 0), cb.get('Precision@3', 0), cb.get('Precision@5', 0)]
ax5.plot(k_lbl, precs, 'go-', linewidth=2, markersize=9, label='Content-Based')
hybrid_key = next((k for k in rec_metrics if 'Hybrid' in k), None)
if hybrid_key:
    hp1 = rec_metrics[hybrid_key].get('Precision@1', 0)
    ax5.scatter(['@1'], [hp1], color='red', s=120, zorder=5,
                label=f'Hybrid P@1={hp1:.3f}')
for k, p in zip(k_lbl, precs):
    ax5.text(k, p+0.01, f'{p:.2f}', ha='center', fontsize=9)
ax5.set_title('Recommender Precision@k', fontweight='bold')
ax5.set_ylabel('Precision'); ax5.set_ylim(0, 1.1)
ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

# ── Plot 6: Summary Table ──
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')
rows = []
for name in eval_results:
    rows.append([name, 'Classification',
                 f'{eval_results[name]["accuracy"]:.4f}',
                 f'{eval_results[name]["f1"]:.4f}', '—', '—'])
for n in ['SVD','NMF','KNN']:
    if n in rec_metrics:
        rows.append([n, 'Collaborative Filtering', '—', '—',
                     f'{rec_metrics[n].get("RMSE",0):.4f}',
                     f'{rec_metrics[n].get("MAE",0):.4f}'])
rows.append(['Content-Based', 'Recommender',
             f'{cb.get("Precision@1",0):.4f}', '—', '—', '—'])
if hybrid_key:
    rows.append([hybrid_key, 'Recommender (Hybrid)',
                 f'{rec_metrics[hybrid_key].get("Precision@1",0):.4f}',
                 '—', '—', '—'])

cols = ['Model', 'Task', 'Accuracy / P@1', 'F1 (weighted)', 'RMSE', 'MAE']
tbl  = ax6.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#ECF0F1')
ax6.set_title('Full System Metrics Summary', fontweight='bold', pad=12)

plt.savefig('outputs/plots/06_final_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Saved → outputs/plots/06_final_evaluation.png")

# ============================================================
# 5. SAVE FINAL METRICS CSV
# ============================================================
print("\n[5/5] Saving final metrics...")
final_rows = []
for name in eval_results:
    final_rows += [
        {'model': name, 'task': 'Classification',
         'metric': 'Accuracy',    'value': eval_results[name]['accuracy']},
        {'model': name, 'task': 'Classification',
         'metric': 'F1_weighted', 'value': eval_results[name]['f1']},
    ]
if not rec_df.empty:
    for _, row in rec_df.iterrows():
        final_rows.append({'model': row['model'], 'task': 'Recommender',
                           'metric': row['metric'], 'value': row['value']})

pd.DataFrame(final_rows).to_csv('outputs/results/06_final_metrics.csv', index=False)
print(" Saved → outputs/results/06_final_metrics.csv")

print("\n" + "="*60)
print(" FINAL EVALUATION COMPLETE")
print("="*60)
print(f"\n Best Classifier  : {best_clf_name}")
print(f"  Accuracy        : {eval_results[best_clf_name]['accuracy']:.4f}")
print(f"  F1 (weighted)   : {eval_results[best_clf_name]['f1']:.4f}")
print(f"\n Content-Based Precision@1 : {cb.get('Precision@1',0):.4f}")
for n in ['SVD','NMF','KNN']:
    if n in rec_metrics:
        print(f" {n}  RMSE={rec_metrics[n].get('RMSE',0):.4f}  MAE={rec_metrics[n].get('MAE',0):.4f}")
print("\n All outputs → outputs/")
print("="*60)
