# ============================================================
# HealthBot-CRS | Step 3: Clustering
# K-Means + DBSCAN + t-SNE Visualization
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import Counter

os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

print("=" * 60)
print(" HealthBot-CRS | Clustering (K-Means + DBSCAN + t-SNE)")
print("=" * 60)

# ============================================================
# 1. LOAD FEATURES
# ============================================================
print("\n[1/6] Loading features...")
df = pd.read_csv('data/healthbot_features.csv')
tfidf_matrix = np.load('outputs/results/tfidf_matrix.npy')
embeddings   = np.load('outputs/results/sbert_embeddings.npy')

le = LabelEncoder()
y_true = le.fit_transform(df['intent_category'])
intent_names = le.classes_

print(f" Loaded {len(df):,} rows")
print(f" TF-IDF shape  : {tfidf_matrix.shape}")
print(f" SBERT shape   : {embeddings.shape}")
print(f" Intent classes: {list(intent_names)}")

# Normalize for cosine-like distance
tfidf_norm = normalize(tfidf_matrix)
emb_norm   = normalize(embeddings)

N_CLUSTERS = 7   # matches number of intent categories

# ============================================================
# 2. K-MEANS CLUSTERING
# ============================================================
print("\n[2/6] K-Means Clustering...")

# --- Elbow + Silhouette sweep ---
k_range = range(2, 14)
inertias, sil_scores_tfidf, sil_scores_emb = [], [], []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_t = km.fit_predict(tfidf_norm)
    labels_e = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emb_norm)
    inertias.append(km.inertia_)
    sil_scores_tfidf.append(silhouette_score(tfidf_norm, labels_t, sample_size=1000))
    sil_scores_emb.append(silhouette_score(emb_norm,   labels_e, sample_size=1000))
    print(f"  k={k:2d}  Sil(TF-IDF)={sil_scores_tfidf[-1]:.3f}  Sil(SBERT)={sil_scores_emb[-1]:.3f}")

# Final K-Means with k=7
km_tfidf = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
km_labels_tfidf = km_tfidf.fit_predict(tfidf_norm)

km_emb = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
km_labels_emb = km_emb.fit_predict(emb_norm)

sil_t  = silhouette_score(tfidf_norm, km_labels_tfidf, sample_size=1000)
sil_e  = silhouette_score(emb_norm,   km_labels_emb,   sample_size=1000)
db_t   = davies_bouldin_score(tfidf_norm, km_labels_tfidf)
db_e   = davies_bouldin_score(emb_norm,   km_labels_emb)
ari_t  = adjusted_rand_score(y_true, km_labels_tfidf)
ari_e  = adjusted_rand_score(y_true, km_labels_emb)
nmi_t  = normalized_mutual_info_score(y_true, km_labels_tfidf)
nmi_e  = normalized_mutual_info_score(y_true, km_labels_emb)

print(f"\n K-Means (k=7) Results:")
print(f"  TF-IDF  → Silhouette={sil_t:.3f}  Davies-Bouldin={db_t:.3f}  ARI={ari_t:.3f}  NMI={nmi_t:.3f}")
print(f"  SBERT   → Silhouette={sil_e:.3f}  Davies-Bouldin={db_e:.3f}  ARI={ari_e:.3f}  NMI={nmi_e:.3f}")

df['kmeans_tfidf'] = km_labels_tfidf
df['kmeans_sbert'] = km_labels_emb

# ============================================================
# 3. DBSCAN CLUSTERING
# ============================================================
print("\n[3/6] DBSCAN Clustering on SBERT embeddings...")

# Reduce dims first for DBSCAN
pca50 = PCA(n_components=50, random_state=42)
emb_50 = pca50.fit_transform(emb_norm)

dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine', n_jobs=-1)
db_labels = dbscan.fit_predict(emb_50)

n_clusters_db  = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise_db     = list(db_labels).count(-1)
noise_pct      = 100 * n_noise_db / len(db_labels)

print(f"  DBSCAN clusters found : {n_clusters_db}")
print(f"  Noise points          : {n_noise_db} ({noise_pct:.1f}%)")

if n_clusters_db > 1:
    mask_valid = db_labels != -1
    sil_db = silhouette_score(emb_50[mask_valid], db_labels[mask_valid], sample_size=min(1000, mask_valid.sum()))
    ari_db = adjusted_rand_score(y_true[mask_valid], db_labels[mask_valid])
    print(f"  Silhouette (non-noise): {sil_db:.3f}")
    print(f"  ARI (non-noise)       : {ari_db:.3f}")
else:
    sil_db, ari_db = 0, 0
    print("  Not enough clusters for Silhouette.")

df['dbscan_label'] = db_labels

# ============================================================
# 4. t-SNE VISUALIZATION
# ============================================================
print("\n[4/6] Running t-SNE (this takes ~1-2 mins)...")

tsne = TSNE(n_components=2, random_state=42, perplexity=40,
            learning_rate=200, max_iter=1000, init='pca')
tsne_2d = tsne.fit_transform(emb_50)

df['tsne_x'] = tsne_2d[:, 0]
df['tsne_y'] = tsne_2d[:, 1]
print("  t-SNE done.")

# ============================================================
# 5. VISUALIZATIONS (2x3 grid)
# ============================================================
print("\n[5/6] Generating plots...")

COLORS = [
    '#E74C3C', '#E67E22', '#3498DB', '#9B59B6',
    '#2ECC71', '#1ABC9C', '#F39C12', '#34495E',
    '#E91E63', '#00BCD4', '#8BC34A', '#FF5722'
]
INTENT_COLOR = {name: COLORS[i] for i, name in enumerate(intent_names)}

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('HealthBot-CRS | Clustering Analysis', fontsize=17, fontweight='bold')

# --- Plot 1: Elbow curve ---
ax = axes[0, 0]
ax.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=6)
ax.axvline(N_CLUSTERS, color='red', linestyle='--', alpha=0.7, label=f'k={N_CLUSTERS}')
ax.set_title('Elbow Curve (K-Means Inertia)', fontweight='bold')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.legend()

# --- Plot 2: Silhouette comparison ---
ax = axes[0, 1]
ax.plot(list(k_range), sil_scores_tfidf, 'rs-', label='TF-IDF', linewidth=2, markersize=6)
ax.plot(list(k_range), sil_scores_emb,   'go-', label='SBERT',  linewidth=2, markersize=6)
ax.axvline(N_CLUSTERS, color='navy', linestyle='--', alpha=0.6, label=f'k={N_CLUSTERS}')
ax.set_title('Silhouette Score vs k', fontweight='bold')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Silhouette Score')
ax.legend()

# --- Plot 3: t-SNE coloured by TRUE intent ---
ax = axes[0, 2]
for intent in intent_names:
    mask = df['intent_category'] == intent
    ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
               c=INTENT_COLOR[intent], label=intent, alpha=0.5, s=8)
ax.set_title('t-SNE by True Intent', fontweight='bold')
ax.legend(fontsize=7, markerscale=2, loc='best')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# --- Plot 4: t-SNE coloured by K-Means (SBERT) ---
ax = axes[1, 0]
for k in range(N_CLUSTERS):
    mask = km_labels_emb == k
    ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
               c=COLORS[k], label=f'Cluster {k}', alpha=0.5, s=8)
ax.set_title(f't-SNE by K-Means (SBERT, k={N_CLUSTERS})', fontweight='bold')
ax.legend(fontsize=7, markerscale=2, loc='best')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# --- Plot 5: t-SNE coloured by DBSCAN ---
ax = axes[1, 1]
unique_db = sorted(set(db_labels))
for lbl in unique_db:
    mask = db_labels == lbl
    color = '#999999' if lbl == -1 else COLORS[lbl % len(COLORS)]
    name  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
    ax.scatter(tsne_2d[mask, 0], tsne_2d[mask, 1],
               c=color, label=name, alpha=0.5, s=8)
ax.set_title(f't-SNE by DBSCAN ({n_clusters_db} clusters)', fontweight='bold')
ax.legend(fontsize=6, markerscale=2, loc='best', ncol=2)
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

# --- Plot 6: Metrics bar chart ---
ax = axes[1, 2]
metrics = ['Sil (TF-IDF)', 'Sil (SBERT)', 'ARI (TF-IDF)', 'ARI (SBERT)', 'NMI (TF-IDF)', 'NMI (SBERT)']
values  = [sil_t, sil_e, ari_t, ari_e, nmi_t, nmi_e]
colors  = ['#3498DB', '#2ECC71'] * 3
bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='white')
for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_title('K-Means Clustering Metrics', fontweight='bold')
ax.set_xlabel('Score')
ax.set_xlim(0, max(values) * 1.2)

plt.tight_layout()
plt.savefig('outputs/plots/03_clustering.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Saved → outputs/plots/03_clustering.png")

# ============================================================
# 6. SAVE RESULTS
# ============================================================
print("\n[6/6] Saving results...")

df.to_csv('data/healthbot_clustered.csv', index=False)

results = {
    'kmeans_tfidf_silhouette':       sil_t,
    'kmeans_sbert_silhouette':        sil_e,
    'kmeans_tfidf_davies_bouldin':    db_t,
    'kmeans_sbert_davies_bouldin':    db_e,
    'kmeans_tfidf_ari':               ari_t,
    'kmeans_sbert_ari':               ari_e,
    'kmeans_tfidf_nmi':               nmi_t,
    'kmeans_sbert_nmi':               nmi_e,
    'dbscan_n_clusters':              n_clusters_db,
    'dbscan_noise_pct':               noise_pct,
    'dbscan_silhouette':              sil_db,
    'dbscan_ari':                     ari_db,
}
pd.Series(results).to_csv('outputs/results/03_clustering_metrics.csv', header=['value'])

print(f" → data/healthbot_clustered.csv")
print(f" → outputs/results/03_clustering_metrics.csv")
print(f" → outputs/plots/03_clustering.png")

print("\n" + "=" * 60)
print(" CLUSTERING COMPLETE")
print("=" * 60)
print(f"\n K-Means (k=7):")
print(f"   TF-IDF  Silhouette = {sil_t:.3f}  |  ARI = {ari_t:.3f}  |  NMI = {nmi_t:.3f}")
print(f"   SBERT   Silhouette = {sil_e:.3f}  |  ARI = {ari_e:.3f}  |  NMI = {nmi_e:.3f}")
print(f"\n DBSCAN:")
print(f"   Clusters = {n_clusters_db}  |  Noise = {noise_pct:.1f}%  |  ARI = {ari_db:.3f}")
print(f"\n Next step: Run 04_classification.py")
print("=" * 60)
