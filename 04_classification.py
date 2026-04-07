# ============================================================
# HealthBot-CRS | Step 4: Classification
# Logistic Regression + Random Forest + SVM
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
import joblib

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)

print("=" * 60)
print(" HealthBot-CRS | Classification")
print(" Logistic Regression + Random Forest + LinearSVC")
print("=" * 60)

# ============================================================
# 1. LOAD DATA & FEATURES
# ============================================================
print("\n[1/5] Loading features...")
df         = pd.read_csv('data/healthbot_clustered.csv')
tfidf_matrix = np.load('outputs/results/tfidf_matrix.npy')
embeddings   = np.load('outputs/results/sbert_embeddings.npy')

le = LabelEncoder()
y  = le.fit_transform(df['intent_category'])
intent_names = list(le.classes_)

# Combine TF-IDF + SBERT as hybrid feature matrix
X_tfidf = normalize(tfidf_matrix)
X_sbert = normalize(embeddings)
X_hybrid = np.hstack([X_tfidf, X_sbert])

print(f" Rows           : {len(df):,}")
print(f" TF-IDF shape   : {X_tfidf.shape}")
print(f" SBERT shape    : {X_sbert.shape}")
print(f" Hybrid shape   : {X_hybrid.shape}")
print(f" Intent classes : {intent_names}")

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================
X_tr, X_te, y_tr, y_te = train_test_split(
    X_hybrid, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n Train: {len(X_tr):,}  |  Test: {len(X_te):,}")

# ============================================================
# 3. TRAIN THREE CLASSIFIERS
# ============================================================
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0,
                                           solver='lbfgs', random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=200, max_depth=20,
                                                   random_state=42, n_jobs=-1),
    'LinearSVC'          : LinearSVC(C=1.0, max_iter=2000, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n[2/5] Training classifiers...")
for name, clf in classifiers.items():
    print(f"\n  → {name}")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc    = accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average='macro')
    f1_wt  = f1_score(y_te, y_pred, average='weighted')
    cv_scores = cross_val_score(clf, X_hybrid, y, cv=cv,
                                scoring='f1_macro', n_jobs=-1)

    results[name] = {
        'clf'     : clf,
        'y_pred'  : y_pred,
        'accuracy': acc,
        'f1_macro': f1_mac,
        'f1_weighted': f1_wt,
        'cv_mean' : cv_scores.mean(),
        'cv_std'  : cv_scores.std(),
    }
    print(f"     Accuracy    = {acc:.4f}")
    print(f"     F1 (macro)  = {f1_mac:.4f}")
    print(f"     F1 (wtd)    = {f1_wt:.4f}")
    print(f"     CV F1 macro = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(classification_report(y_te, y_pred, target_names=intent_names))

    # Save model
    joblib.dump(clf, f'outputs/models/{name.replace(" ", "_")}.pkl')
    fname = name.replace(" ", "_")
    print(f"     Saved → outputs/models/{fname}.pkl")

# ============================================================
# 4. VISUALIZATIONS
# ============================================================
print("\n[3/5] Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('HealthBot-CRS | Classification Results', fontsize=17, fontweight='bold')

clf_names = list(results.keys())
short_names = ['LR', 'RF', 'SVM']
COLORS = ['#3498DB', '#2ECC71', '#E74C3C']

# --- Plot 1: Accuracy & F1 bar chart ---
ax = axes[0, 0]
x = np.arange(len(clf_names))
w = 0.25
bars1 = ax.bar(x - w, [results[n]['accuracy']   for n in clf_names], w, label='Accuracy',    color='#3498DB', alpha=0.85)
bars2 = ax.bar(x,     [results[n]['f1_macro']    for n in clf_names], w, label='F1 (Macro)',  color='#2ECC71', alpha=0.85)
bars3 = ax.bar(x + w, [results[n]['f1_weighted'] for n in clf_names], w, label='F1 (Weighted)',color='#E74C3C', alpha=0.85)
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(short_names)
ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.legend(fontsize=9)

# --- Plot 2: Cross-Validation F1 ---
ax = axes[0, 1]
cv_means = [results[n]['cv_mean'] for n in clf_names]
cv_stds  = [results[n]['cv_std']  for n in clf_names]
ax.bar(short_names, cv_means, yerr=cv_stds, color=COLORS,
       alpha=0.85, capsize=8, edgecolor='white')
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, m + s + 0.01, f'{m:.3f}', ha='center', fontsize=9)
ax.set_ylim(0, 1.1); ax.set_ylabel('F1 Macro')
ax.set_title('5-Fold Cross-Validation F1 (Macro)', fontweight='bold')

# --- Plots 3-5: Confusion matrices ---
for idx, (name, sname) in enumerate(zip(clf_names, short_names)):
    row, col = divmod(idx + 2, 3)  # fills (0,2),(1,0),(1,1)
    # remap: (0,2)->(0,2), 1->(1,0), 2->(1,1)
    positions = [(0,2),(1,0),(1,1)]
    ax = axes[positions[idx][0]][positions[idx][1]]
    cm = confusion_matrix(y_te, results[name]['y_pred'])
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=intent_names, yticklabels=intent_names,
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title(f'Confusion Matrix — {sname} (%)', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    ax.tick_params(axis='y', rotation=0,  labelsize=7)

# --- Plot 6: Per-class F1 heatmap ---
ax = axes[1, 2]
per_class = {}
for name in clf_names:
    report = classification_report(y_te, results[name]['y_pred'],
                                   target_names=intent_names, output_dict=True)
    per_class[name] = [report[c]['f1-score'] for c in intent_names]

pc_df = pd.DataFrame(per_class, index=intent_names)
sns.heatmap(pc_df, annot=True, fmt='.2f', cmap='YlGn',
            ax=ax, linewidths=0.5, vmin=0, vmax=1)
ax.set_title('Per-Class F1 Score by Model', fontweight='bold')
ax.tick_params(axis='y', rotation=0, labelsize=8)
ax.set_xticklabels(['LR', 'RF', 'SVM'], rotation=0)

plt.tight_layout()
plt.savefig('outputs/plots/04_classification.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved → outputs/plots/04_classification.png")

# ============================================================
# 5. SAVE METRICS
# ============================================================
print("\n[4/5] Saving metrics...")
rows = []
for name in clf_names:
    rows.append({
        'model'       : name,
        'accuracy'    : results[name]['accuracy'],
        'f1_macro'    : results[name]['f1_macro'],
        'f1_weighted' : results[name]['f1_weighted'],
        'cv_f1_mean'  : results[name]['cv_mean'],
        'cv_f1_std'   : results[name]['cv_std'],
    })
metrics_df = pd.DataFrame(rows)
metrics_df.to_csv('outputs/results/04_classification_metrics.csv', index=False)
print(" → outputs/results/04_classification_metrics.csv")

print("\n" + "=" * 60)
print(" CLASSIFICATION COMPLETE")
print("=" * 60)
for name in clf_names:
    print(f"  {name:22s} Acc={results[name]['accuracy']:.4f}  F1={results[name]['f1_macro']:.4f}")
print(f"\n Next step: Run 05_recommender.py")
print("=" * 60)
