# kmeans_utils.py
"""
Utility functions for KMeans analysis:
- load file
- preprocess (numeric select, fill missing, standardize)
- manual KMeans implementation (iterative)
- evaluate (inertia, silhouette, calinski, davies)
- plotting helpers (heatmap, pairplot, PCA 2D/3D)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io, os

sns.set(style="darkgrid")

def load_file(path):
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    elif path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file type. Use .csv or .xlsx/.xls")

def preprocessing(df, fill_strategy="median"):
    # Pilih kolom numerik
    df_num = df.select_dtypes(include=[np.number]).copy()
    if fill_strategy == "median":
        df_num = df_num.fillna(df_num.median())
    else:
        df_num = df_num.fillna(df_num.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_num)
    return df_num, X_scaled, scaler

# Manual simple KMeans implementation (as requested)
class SimpleKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)
        idx = rng.choice(range(X.shape[0]), size=self.n_clusters, replace=False)
        self.centroids = X[idx].astype(float)

    def _compute_distances(self, X):
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return dists

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        self._init_centroids(X)
        labels = np.full(n_samples, -1, dtype=int)

        for it in range(self.max_iter):
            dists = self._compute_distances(X)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                # converged
                break
            labels = new_labels
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members) > 0:
                    self.centroids[k] = members.mean(axis=0)
                else:
                    # reinitialize empty centroid
                    self.centroids[k] = X[np.random.randint(0, n_samples)]
        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids[self.labels_])**2)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

# Evaluation helpers
def compute_elbow(X_scaled, k_range=range(1, 11)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    return list(k_range), inertias

def compute_silhouette(X_scaled, k_range=range(2, 11)):
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, labels))
    return list(k_range), sil_scores

def compute_other_scores(X_scaled, labels):
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    sil = silhouette_score(X_scaled, labels)
    return {"silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": db}

# Plotting utilities (save PNG and also return figure object)
def plot_elbow(k_vals, inertias, outpath=None):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(k_vals, inertias, '-o')
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method")
    ax.grid(True)
    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
    return fig

def plot_silhouette(k_vals, sil_scores, outpath=None):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(k_vals, sil_scores, '-o')
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette per k")
    ax.grid(True)
    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
    return fig

def plot_heatmap(df_num, outpath=None):
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df_num.corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax)
    ax.set_title("Correlation Heatmap")
    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
    return fig

def pca_2d_plot(X_scaled, labels, outpath=None):
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(6,5))
    sns.scatterplot(x=Xp[:,0], y=Xp[:,1], hue=labels, palette="tab10", ax=ax)
    ax.set_title("PCA 2D")
    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
    return fig

def pca_3d_plot(X_scaled, labels, outpath=None):
    from mpl_toolkits.mplot3d import Axes3D
    pca = PCA(n_components=3)
    Xp = pca.fit_transform(X_scaled)
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(Xp[:,0], Xp[:,1], Xp[:,2], c=labels, cmap='tab10', s=40)
    ax.set_title("PCA 3D")
    if outpath:
        fig.savefig(outpath, bbox_inches='tight')
    return fig

def save_cluster_results(df_original, labels, out_csv="hasil_kmeans.csv"):
    df_out = df_original.copy().reset_index(drop=True)
    df_out["cluster"] = labels
    df_out.to_csv(out_csv, index=False)
    return out_csv

def cluster_profile(df_original, labels):
    df = df_original.copy().reset_index(drop=True)
    df["cluster"] = labels
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    profile = df.groupby("cluster")[numeric_cols].mean()
    counts = df['cluster'].value_counts().sort_index()
    return profile, counts
