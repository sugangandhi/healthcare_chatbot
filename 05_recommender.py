# ============================================================
# HealthBot-CRS | Step 5: Recommender System
# Content-Based + SVD/NMF/KNN (sklearn only, no Surprise)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.makedirs('outputs/plots',   exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
os.makedirs('outputs/models',  exist_ok=True)

print("=" * 60)
print(" HealthBot-CRS | Recommender System (sklearn only)")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n[1/6] Loading data...")
df         = pd.read_csv('data/healthbot_clustered.csv')
embeddings = np.load('outputs/results/sbert_embeddings.npy')

le = LabelEncoder()
df['intent_encoded'] = le.fit_transform(df['intent_category'])
intent_names = list(le.classes_)
n_intents    = len(intent_names)

print(f" Rows      : {len(df):,}")
print(f" Intents   : {n_intents}")
print(f" Embeddings: {embeddings.shape}")

# ============================================================
# 2. CONTENT-BASED RECOMMENDER
# ============================================================
print("\n[2/6] Building Content-Based Recommender...")
emb_norm = normalize(embeddings)

def recommend_cb(query_idx, top_k=5):
    sims = cosine_similarity(emb_norm[query_idx].reshape(1,-1), emb_norm)[0]
    sims[query_idx] = -1
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(i, sims[i], df.iloc[i]['intent_category']) for i in top_idx]

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(df), 200, replace=False)

# Precision@k
def cb_precision_at_k(k):
    correct = 0
    for idx in sample_idx[:100]:
        recs = recommend_cb(idx, top_k=k)
        if any(r[2] == df.iloc[idx]['intent_category'] for r in recs):
            correct += 1
    return correct / 100

cb_p1  = cb_precision_at_k(1)
cb_p3  = cb_precision_at_k(3)
cb_p5  = cb_precision_at_k(5)
cb_p10 = cb_precision_at_k(10)
cb_sims = [recommend_cb(i, 1)[0][1] for i in sample_idx[:100]]
cb_avg_sim = np.mean(cb_sims)

print(f" Precision@1={cb_p1:.3f}  @3={cb_p3:.3f}  @5={cb_p5:.3f}  @10={cb_p10:.3f}")
print(f" Avg Cosine Similarity: {cb_avg_sim:.3f}")

# ============================================================
# 3. BUILD USER-ITEM MATRIX
# ============================================================
print("\n[3/6] Building user-item interaction matrix...")

# User = cluster (kmeans_sbert), Item = intent_encoded
# Rating = normalized entity_count + output_length signal
df['rating'] = (
    (df['entity_count']  / (df['entity_count'].max()  + 1e-9)) * 2.5 +
    (df['output_length'] / (df['output_length'].max() + 1e-9)) * 2.5
).clip(0.5, 5.0)

n_users = df['kmeans_sbert'].nunique()

# Aggregate by (cluster, intent) → mean rating
agg = df.groupby(['kmeans_sbert', 'intent_encoded'])['rating'].mean().reset_index()
R = np.zeros((n_users, n_intents))
for _, row in agg.iterrows():
    R[int(row['kmeans_sbert']), int(row['intent_encoded'])] = row['rating']

print(f" User-Item matrix shape: {R.shape}")
print(f" Sparsity: {(R==0).sum() / R.size * 100:.1f}%")

# ============================================================
# 4. COLLABORATIVE FILTERING — SVD / NMF / KNN
# ============================================================
print("\n[4/6] Training CF models (sklearn)...")

def eval_cf(R_pred, R_true):
    mask = R_true > 0
    rmse = np.sqrt(mean_squared_error(R_true[mask], R_pred[mask]))
    mae  = mean_absolute_error(R_true[mask], R_pred[mask])
    return rmse, mae

cf_results = {}

# --- TruncatedSVD ---
svd   = TruncatedSVD(n_components=min(10, n_intents-1), random_state=42)
U     = svd.fit_transform(R)
R_svd = np.dot(U, svd.components_)
R_svd = np.clip(R_svd, 0.5, 5.0)
rmse, mae = eval_cf(R_svd, R)
cf_results['SVD'] = {'rmse': rmse, 'mae': mae, 'R_pred': R_svd}
joblib.dump(svd, 'outputs/models/svd_recommender.pkl')
print(f"  SVD  → RMSE={rmse:.4f}  MAE={mae:.4f}")

# --- NMF ---
nmf   = NMF(n_components=min(10, n_intents-1), max_iter=500, random_state=42)
W     = nmf.fit_transform(R)
R_nmf = np.dot(W, nmf.components_)
R_nmf = np.clip(R_nmf, 0.5, 5.0)
rmse, mae = eval_cf(R_nmf, R)
cf_results['NMF'] = {'rmse': rmse, 'mae': mae, 'R_pred': R_nmf}
joblib.dump(nmf, 'outputs/models/nmf_recommender.pkl')
print(f"  NMF  → RMSE={rmse:.4f}  MAE={mae:.4f}")

# --- KNN (user-based) ---
knn = NearestNeighbors(n_neighbors=min(5, n_users-1), metric='cosine')
knn.fit(R)
distances, indices = knn.kneighbors(R)
R_knn = np.zeros_like(R)
for u in range(n_users):
    neighbors = indices[u][1:]   # exclude self
    weights   = 1 - distances[u][1:]
    weights   = weights / (weights.sum() + 1e-9)
    R_knn[u]  = np.average(R[neighbors], axis=0, weights=weights)
R_knn = np.clip(R_knn, 0.5, 5.0)
rmse, mae = eval_cf(R_knn, R)
cf_results['KNN'] = {'rmse': rmse, 'mae': mae, 'R_pred': R_knn}
joblib.dump(knn, 'outputs/models/knn_recommender.pkl')
print(f"  KNN  → RMSE={rmse:.4f}  MAE={mae:.4f}")

# Pick best CF model (lowest RMSE)
best_name = min(cf_results, key=lambda x: cf_results[x]['rmse'])
R_best    = cf_results[best_name]['R_pred']
print(f"\n  Best CF model: {best_name}")

# ============================================================
# 5. HYBRID RECOMMENDER
# ============================================================
print("\n[5/6] Hybrid Recommender + Alpha Sensitivity...")

def recommend_hybrid(query_idx, top_k=5, alpha=0.6):
    cb_scores = cosine_similarity(emb_norm[query_idx].reshape(1,-1), emb_norm)[0]
    cb_scores[query_idx] = -1

    user_id   = int(df.iloc[query_idx]['kmeans_sbert'])
    cf_intent = R_best[user_id]   # shape (n_intents,)

    # Expand CF score to all rows by their intent
    cf_row = np.array([cf_intent[df.iloc[i]['intent_encoded']] for i in range(len(df))])
    cf_row_norm = (cf_row - cf_row.min()) / (cf_row.max() - cf_row.min() + 1e-9)

    hybrid = alpha * cb_scores + (1 - alpha) * cf_row_norm
    hybrid[query_idx] = -1
    top_idx = np.argsort(hybrid)[::-1][:top_k]
    return [(i, hybrid[i], df.iloc[i]['intent_category']) for i in top_idx]

# Alpha sensitivity
alphas       = np.round(np.arange(0.0, 1.1, 0.1), 1)
hybrid_prec  = []
for alpha in alphas:
    correct = sum(
        1 for idx in sample_idx[:50]
        if recommend_hybrid(idx, 1, alpha)[0][2] == df.iloc[idx]['intent_category']
    )
    hybrid_prec.append(correct / 50)

best_alpha = alphas[np.argmax(hybrid_prec)]
print(f"  Best alpha: {best_alpha:.1f}  →  Precision@1={max(hybrid_prec):.3f}")

# ============================================================
# 6. VISUALIZATIONS
# ============================================================
print("\n[6/6] Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('HealthBot-CRS | Recommender System (sklearn)',
             fontsize=17, fontweight='bold')

# Plot 1: CF RMSE & MAE
ax = axes[0, 0]
names = list(cf_results.keys())
rmses = [cf_results[n]['rmse'] for n in names]
maes  = [cf_results[n]['mae']  for n in names]
x, w  = np.arange(len(names)), 0.35
b1 = ax.bar(x - w/2, rmses, w, label='RMSE', color='#E74C3C', alpha=0.85)
b2 = ax.bar(x + w/2, maes,  w, label='MAE',  color='#3498DB', alpha=0.85)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
            f'{b.get_height():.3f}', ha='center', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_title('CF Models — RMSE & MAE', fontweight='bold')
ax.set_ylabel('Error'); ax.legend()

# Plot 2: Content-based cosine similarity distribution
ax = axes[0, 1]
ax.hist(cb_sims, bins=30, color='#2ECC71', edgecolor='white', alpha=0.85)
ax.axvline(cb_avg_sim, color='red', linestyle='--', label=f'Mean: {cb_avg_sim:.3f}')
ax.set_title('Content-Based Similarity Distribution', fontweight='bold')
ax.set_xlabel('Cosine Similarity'); ax.set_ylabel('Frequency'); ax.legend()

# Plot 3: Avg rating by intent
ax = axes[0, 2]
ir = df.groupby('intent_category')['rating'].mean().sort_values()
bars = ax.barh(ir.index, ir.values, color='#9B59B6', alpha=0.85, edgecolor='white')
for bar, val in zip(bars, ir.values):
    ax.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=9)
ax.set_title('Avg Interaction Rating by Intent', fontweight='bold')
ax.set_xlabel('Average Rating')

# Plot 4: Precision@k curve
ax = axes[1, 0]
k_vals = [1, 3, 5, 10]
precs  = [cb_p1, cb_p3, cb_p5, cb_p10]
ax.plot(k_vals, precs, 'bo-', linewidth=2, markersize=8)
for k, p in zip(k_vals, precs):
    ax.text(k, p+0.01, f'{p:.2f}', ha='center', fontsize=9)
ax.set_title('Content-Based Precision@k', fontweight='bold')
ax.set_xlabel('k'); ax.set_ylabel('Precision')
ax.set_ylim(0, 1.1); ax.grid(alpha=0.3)

# Plot 5: Heatmap cluster vs intent
ax = axes[1, 1]
pivot = df.groupby(['kmeans_sbert','intent_category'])['rating'].mean().unstack(fill_value=0)
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, annot=True, fmt='.1f',
            linewidths=0.5, cbar_kws={'shrink':0.8})
ax.set_title('Avg Rating: Cluster × Intent', fontweight='bold')
ax.set_xlabel('Intent'); ax.set_ylabel('Cluster')
ax.tick_params(axis='x', rotation=45, labelsize=7)

# Plot 6: Alpha sensitivity
ax = axes[1, 2]
ax.plot(alphas, hybrid_prec, 'rs-', linewidth=2, markersize=7)
ax.axvline(best_alpha, color='navy', linestyle='--', label=f'Best α={best_alpha:.1f}')
ax.set_title('Hybrid: Alpha Sensitivity', fontweight='bold')
ax.set_xlabel('Alpha (Content weight)'); ax.set_ylabel('Precision@1')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/05_recommender.png', dpi=150, bbox_inches='tight')
plt.close()
print(" Saved → outputs/plots/05_recommender.png")

# ============================================================
# SAVE METRICS
# ============================================================
rows = [
    {'model': 'Content-Based', 'metric': 'Precision@1',     'value': cb_p1},
    {'model': 'Content-Based', 'metric': 'Precision@3',     'value': cb_p3},
    {'model': 'Content-Based', 'metric': 'Precision@5',     'value': cb_p5},
    {'model': 'Content-Based', 'metric': 'Avg Cosine Sim',  'value': cb_avg_sim},
]
for n in cf_results:
    rows += [
        {'model': n, 'metric': 'RMSE', 'value': cf_results[n]['rmse']},
        {'model': n, 'metric': 'MAE',  'value': cf_results[n]['mae']},
    ]
rows.append({'model': f'Hybrid(α={best_alpha})', 'metric': 'Precision@1', 'value': max(hybrid_prec)})
pd.DataFrame(rows).to_csv('outputs/results/05_recommender_metrics.csv', index=False)

print(" Saved → outputs/results/05_recommender_metrics.csv")
print("\n" + "="*60)
print(" RECOMMENDER COMPLETE")
print("="*60)
print(f" Content-Based  Precision@1 : {cb_p1:.3f}")
print(f" Avg Cosine Similarity      : {cb_avg_sim:.3f}")
for n in cf_results:
    print(f" {n:5s}  RMSE={cf_results[n]['rmse']:.4f}  MAE={cf_results[n]['mae']:.4f}")
print(f" Hybrid(α={best_alpha}) Precision@1 : {max(hybrid_prec):.3f}")
print(f"\n Next → Run 06_evaluation.py")
print("="*60)
