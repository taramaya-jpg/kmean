import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Page config with icon
st.set_page_config(page_title="KMeans Clustering Explorer", page_icon="🧮", layout="centered")

# Load or simulate your dataset (replace with your own)
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'Feature1': np.random.normal(0, 1, 200),
        'Feature2': np.random.normal(5, 2, 200),
        'Feature3': np.random.normal(-3, 1, 200),
        'Feature4': np.random.normal(2, 1, 200)
    })
    return data

df = load_data()

# Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Title with icon
st.title("🧮 K-means Clustering Explorer")

st.markdown("""
Select the number of clusters (k), then run K-means clustering.
See silhouette scores for k=2 to 10 and find the best and worst values.
""")

# Sidebar for k selection
k_selected = st.sidebar.slider('Select k (number of clusters)', 2, 10, 3)

# Calculate silhouette scores for k=2 to 10
sil_scores = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df_scaled)
    sil = silhouette_score(df_scaled, labels)
    sil_scores[k] = sil

best_k = max(sil_scores, key=sil_scores.get)
worst_k = min(sil_scores, key=sil_scores.get)

# Display silhouette scores table with highlights
st.markdown("### 📊 Silhouette Scores for k=2 to 10")

scores_df = pd.DataFrame({
    'k': list(sil_scores.keys()),
    'Silhouette Score': list(sil_scores.values())
})

def highlight_best_worst(row):
    if row['k'] == best_k:
        return ['background-color: #a2d5f2; font-weight: bold;'] * len(row)  # Light blue
    elif row['k'] == worst_k:
        return ['background-color: #f28c8c; font-weight: bold;'] * len(row)  # Light red
    else:
        return [''] * len(row)

st.dataframe(scores_df.style.apply(highlight_best_worst, axis=1), height=300)

# Popup messages with icons for best/worst k selected
if k_selected == best_k:
    st.success(f"🎉 Best k = {best_k} with Silhouette Score = {sil_scores[best_k]:.4f}")
elif k_selected == worst_k:
    st.error(f"⚠️ Worst k = {worst_k} with Silhouette Score = {sil_scores[worst_k]:.4f}")

# Run K-means for selected k
kmeans = KMeans(n_clusters=k_selected, random_state=42)
labels = kmeans.fit_predict(df_scaled)
df['Cluster'] = labels

# Show cluster assignments
st.markdown(f"### 📋 Cluster Assignments for k={k_selected}")
st.dataframe(df.head(20))  # Show first 20 rows with cluster labels

# Show cluster centers
st.markdown(f"### 🎯 Cluster Centers (scaled features) for k={k_selected}")
centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=df.columns[:-1])
st.dataframe(centers_df.style.background_gradient(cmap='Blues'))

# PCA visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

st.markdown(f"### 📈 PCA Plot of Clusters (k={k_selected})")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='Set2', data=df, ax=ax, s=80)
ax.set_title('Clusters Visualized by PCA')
st.pyplot(fig)

# Custom CSS for hover effects and styling
st.markdown("""
<style>
[data-testid="stDataFrame"] tbody tr:hover {
    background-color: #f0f8ff !important;
}
h1, h2, h3 {
    color: #2c3e50;
}
.stButton>button {
    background-color: #2980b9;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #3498db;
    color: #ecf0f1;
}
</style>
""", unsafe_allow_html=True)
