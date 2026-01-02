# kmeans_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
from kmeans_utils import load_file, preprocessing, compute_elbow, compute_silhouette, pca_2d_plot, pca_3d_plot, save_cluster_results, cluster_profile
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="KMeans Dashboard")

st.title("KMeans Dashboard (Dark Theme)")

uploaded = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx", "xls"])
if uploaded:
    # save temp
    df = load_file(uploaded) if isinstance(uploaded, str) else (pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded))
    st.write("Data preview:")
    st.dataframe(df.head())

    df_num, X_scaled, scaler = preprocessing(df)
    st.write("Numeric columns:", df_num.columns.tolist())

    st.sidebar.header("Clustering Options")
    k = st.sidebar.slider("Jumlah cluster (k)", 2, 12, 3)
    run = st.sidebar.button("Run Clustering")

    if run:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        df["cluster"] = labels
        st.success("Clustering selesai")

        # Elbow & Silhouette
        ks, inertias = compute_elbow(X_scaled, k_range=range(1,11))
        fig_elb = plt.figure()
        plt.plot(ks, inertias, marker="o"); plt.title("Elbow"); plt.xlabel("k"); plt.ylabel("Inertia")
        st.pyplot(fig_elb)

        ks2, sil = compute_silhouette(X_scaled, k_range=range(2,11))
        fig_sil = plt.figure()
        plt.plot(ks2, sil, marker="o"); plt.title("Silhouette"); plt.xlabel("k"); plt.ylabel("Score")
        st.pyplot(fig_sil)

        # PCA 2D
        fig_p2 = pca_2d_plot(X_scaled, labels)
        st.pyplot(fig_p2)

        # Show cluster profile
        profile, counts = cluster_profile(df, labels)
        st.write("Cluster profile (means):")
        st.dataframe(profile)
        st.write("Counts per cluster:")
        st.write(counts)

        # Export CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download results (CSV)", data=csv, file_name="hasil_kmeans.csv", mime="text/csv")
