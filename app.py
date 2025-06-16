import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie 
import requests
import time

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_clustering = load_lottieurl('https://assets4.lottiefiles.com/packages/lf20_xyadoh9h.json')
lottie_success = load_lottieurl('https://assets7.lottiefiles.com/packages/lf20_jbrw3hcz.json')
lottie_warning = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_kq5rGs.json')

# Page config
st.set_page_config(page_title="KMeans Clustering Explorer", page_icon="🧮", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main > div { padding: 2rem; border-radius: 10px; background: #ffffff; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
    .stButton>button { width: 100%; padding: 0.5rem; font-weight: 600; border: none; border-radius: 8px;
        background: linear-gradient(45deg, #2980b9, #3498db); color: white; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
        background: linear-gradient(45deg, #3498db, #2980b9); }
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    [data-testid="stDataFrame"] tbody tr:hover { background-color: #f0f8ff !important; }
    .css-1d391kg, .css-12w0qpk { background: linear-gradient(120deg, #f6d365 0%, #fda085 100%);
        padding: 1rem; border-radius: 8px; }
    h1, h2, h3 { color: #2c3e50; font-weight: 700; }
    .stProgress > div > div { background-color: #2ecc71; }
</style>
""", unsafe_allow_html=True)

# Title and Lottie animation
st.title("🧮 K-means Clustering Explorer")
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    ### 🚀 Welcome to the Interactive K-means Clustering Explorer!
    This app helps you understand how K-means clustering works with different values of k.
    - 📊 Analyze silhouette scores
    - 🎯 Visualize cluster centers
    - 📈 Explore PCA plots
    """)
with col2:
    st_lottie(lottie_clustering, height=200)

# File uploader
try:
    df = pd.read_csv("globalfood.csv", encoding='latin1')  # Added encoding fix
except FileNotFoundError:
    st.error("CSV file not found. Please make sure 'globalfood.csv' exists in the app folder.")
    st.stop()
except UnicodeDecodeError:
    st.error("Failed to decode CSV file. Try using a different encoding like 'latin1'.")
    st.stop()
df.dropna(inplace=True) 

# Standardize numeric columns only
df_numeric = df.select_dtypes(include=[np.number])
if df_numeric.shape[1] < 2:
    st.error("The dataset must contain at least two numeric columns for clustering.")
    st.stop()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Sidebar k selection
st.sidebar.markdown("### 🎛️ Configuration")
k_selected = st.sidebar.slider('Select number of clusters (k)', 2, 10, 3)

# Progress bar for computation
with st.spinner('Computing silhouette scores...'):
    progress_bar = st.progress(0)
    sil_scores = {}
    for i, k in enumerate(range(2, 11)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        sil_scores[k] = silhouette_score(df_scaled, labels)
        progress_bar.progress((i + 1) * 10)
    time.sleep(0.5)
    progress_bar.empty()

best_k = max(sil_scores, key=sil_scores.get)
worst_k = min(sil_scores, key=sil_scores.get)

# Show silhouette scores
st.markdown("### 📊 Silhouette Scores Analysis")
scores_df = pd.DataFrame({'k': list(sil_scores.keys()), 'Silhouette Score': list(sil_scores.values())})

def highlight_best_worst(row):
    if row['k'] == best_k:
        return ['background-color: #a2d5f2; font-weight: bold;'] * len(row)
    elif row['k'] == worst_k:
        return ['background-color: #f28c8c; font-weight: bold;'] * len(row)
    else:
        return [''] * len(row)

st.dataframe(scores_df.style.apply(highlight_best_worst, axis=1).format({'Silhouette Score': '{:.4f}'}), height=300)

# Feedback on selected k
if k_selected == best_k:
    st.success(f"🎉 Excellent choice! k={best_k} is the optimal number of clusters")
    st_lottie(lottie_success, height=150)
elif k_selected == worst_k:
    st.error(f"⚠️ Note: k={worst_k} might not be the best choice")
    st_lottie(lottie_warning, height=150)
elif sil_scores[k_selected] > np.mean(list(sil_scores.values())):
    st.info(f"👍 Good choice! k={k_selected} performs above average")
else:
    st.warning(f"🤔 k={k_selected} performs below average")

# Apply KMeans
kmeans = KMeans(n_clusters=k_selected, random_state=42, n_init=10)
labels = kmeans.fit_predict(df_scaled)
df['Cluster'] = labels

# Show cluster assignments
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"### 📋 Sample Cluster Assignments (k={k_selected})")
    st.dataframe(df.head(10).style.background_gradient(subset=['Cluster'], cmap='viridis'))
with col2:
    st.markdown(f"### 🎯 Cluster Centers")
    centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=df_numeric.columns)
    st.dataframe(centers_df.style.background_gradient(cmap='RdYlBu').format('{:.2f}'))

# PCA Visualization
st.markdown("### 📈 PCA Visualization")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', palette='rainbow', data=df, ax=ax, s=100)
ax.set_title(f'Cluster Visualization (k={k_selected})')
ax.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-family: Arial, sans-serif; color: #555;'>
    <p><span style="font-weight: 600;">K-means Clustering Explorer</span> &#128200;</p>
</div>
""", unsafe_allow_html=True)
