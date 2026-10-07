import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(
    page_title="Mobile Customer Analytics & Recommendation Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Load Preprocessed Data and Train Clustering / Recommendation Artifacts
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('preprocessed_mobile_reviews.csv')
    
    # Invert One-Hot Encoding
    brand_cols = [c for c in df.columns if c.startswith('brand_')]
    model_cols = [c for c in df.columns if c.startswith('model_')]
    
    df['brand'] = df[brand_cols].idxmax(axis=1).where(
        df[brand_cols].max(axis=1) == 1, 'brand_Apple'
    ).str.replace('brand_', '')
    
    df['model'] = df[model_cols].idxmax(axis=1).where(
        df[model_cols].max(axis=1) == 1, 'model_Edge 50'
    ).str.replace('model_', '')
    
    # K-Means 4-Cluster Segmentation
    cluster_features = [
        'price_usd_scaled', 
        'rating_scaled', 
        'helpful_votes_scaled', 
        'specifications_score_scaled'
    ]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df[cluster_features])
    
    # Mapping Segment Names from Standardized Centers
    cluster_means = df.groupby('cluster')[['price_usd_scaled', 'rating_scaled']].mean()
    def label_cluster(row):
        is_premium = row['price_usd_scaled'] > 0
        is_satisfied = row['rating_scaled'] > 0
        if is_premium and is_satisfied:
            return 'Premium Delighted (Champions)'
        elif is_premium and not is_satisfied:
            return 'Premium Flagship Detractors'
        elif not is_premium and is_satisfied:
            return 'Budget/Value Champions'
        else:
            return 'Budget/Mid-Range Detractors'
            
    segment_map = {cid: label_cluster(cluster_means.loc[cid]) for cid in range(4)}
    df['segment_name'] = df['cluster'].map(segment_map)
    
    # 2D PCA for visual projection
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(df[cluster_features])
    df['pca_dim1'] = pca_res[:, 0]
    df['pca_dim2'] = pca_res[:, 1]
    
    # Product catalog for recommendations
    spec_metrics = [
        'price_usd_scaled', 'rating_scaled', 'specifications_score_scaled',
        'battery_life_rating_scaled', 'camera_rating_scaled',
        'performance_rating_scaled', 'design_rating_scaled', 'display_rating_scaled'
    ]
    catalog = df.groupby(['brand', 'model'])[spec_metrics].mean().reset_index()
    
    # Normalization for similarity
    norm_mat = MinMaxScaler().fit_transform(catalog[spec_metrics])
    sim_mat = cosine_similarity(norm_mat)
    
    return df, catalog, sim_mat

df, catalog, sim_mat = load_and_process_data()

# Header & Overview
st.title("📱 Mobile Customer Analytics & Recommendation Engine")
st.markdown("Automated segmentation, exploratory trends, and similarity-based product alternatives based on **preprocessed_mobile_reviews.csv**.")

tab1, tab2, tab3 = st.tabs(["📊 Market Analytics", "🎯 Customer Segmentation", "💡 Recommendation Engine"])

# TAB 1: Market Analytics
with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analyzed Reviews", f"{len(df):,}")
    m2.metric("Tracked Brands", f"{df['brand'].nunique()}")
    m3.metric("Tracked Models", f"{df['model'].nunique()}")
    m4.metric("Avg Scaled Rating", f"{df['rating_scaled'].mean():.2f}")
    
    st.markdown("---")
    st.subheader("Standardized Price vs. Rating Distribution")
    sample_df = df.sample(3000, random_state=42)
    fig_scatter = px.scatter(
        sample_df,
        x='price_usd_scaled', 
        y='rating_scaled', 
        color='brand',
        hover_data=['model', 'segment_name'],
        labels={'price_usd_scaled': 'Standardized Price (Z-Score)', 'rating_scaled': 'Standardized Rating (Z-Score)'},
        title="Price vs. Rating by Brand"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 2: Customer Segmentation
with tab2:
    st.subheader("K-Means 4-Cluster Segmentation (PCA Projection)")
    sample_df = df.sample(3000, random_state=42)
    fig_pca = px.scatter(
        sample_df,
        x='pca_dim1', 
        y='pca_dim2', 
        color='segment_name',
        labels={'pca_dim1': 'PCA Dim 1 (Overall Quality & Satisfaction)', 'pca_dim2': 'PCA Dim 2 (Price & Tier)'},
        color_discrete_sequence=px.colors.qualitative.Set1,
        title="2D Principal Component Projection of Review Segments"
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    st.subheader("Cluster Metric Profiles (Z-Scores)")
    seg_summary = df.groupby('segment_name').agg(
        Review_Count=('rating_scaled', 'count'),
        Price_ZScore=('price_usd_scaled', 'mean'),
        Rating_ZScore=('rating_scaled', 'mean'),
        Specs_ZScore=('specifications_score_scaled', 'mean'),
        Helpful_Votes_ZScore=('helpful_votes_scaled', 'mean')
    ).reset_index()
    st.dataframe(seg_summary.style.format({
        'Price_ZScore': '{:.2f}', 'Rating_ZScore': '{:.2f}', 
        'Specs_ZScore': '{:.2f}', 'Helpful_Votes_ZScore': '{:.2f}'
    }), use_container_width=True)

# TAB 3: Recommendation Engine
with tab3:
    st.subheader("Device Alternative Matcher")
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen_model = st.selectbox("Select a target phone model:", catalog['model'].unique())
    with c2:
        top_k = st.slider("Number of recommendations:", 1, 5, 3)
        
    if st.button("Generate Recommendations", type="primary"):
        idx = catalog[catalog['model'] == chosen_model].index[0]
        scores = sorted(list(enumerate(sim_mat[idx])), key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_k]
        
        recs = catalog.iloc[[s[0] for s in scores]].copy()
        recs['Match_Confidence'] = [f"{s[1]*100:.1f}%" for s in scores]
        
        st.success(f"Top {top_k} recommendations matching **{chosen_model}**:")
        display_recs = recs[['brand', 'model', 'price_usd_scaled', 'rating_scaled', 'Match_Confidence']].rename(columns={
            'brand': 'Brand', 'model': 'Model', 'price_usd_scaled': 'Price Index (Z)', 'rating_scaled': 'Rating Index (Z)'
        })
        st.dataframe(display_recs, use_container_width=True)