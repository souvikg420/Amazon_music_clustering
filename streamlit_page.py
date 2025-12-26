import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Amazon Music Clustering",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Amazon Music Clustering (Unsupervised Learning)")
st.markdown("""
This application groups Amazon Music tracks into **4 clusters** based on **audio features**  
using **K-Means clustering**.
""")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    try:
        return pd.read_csv("amazon_music_clustered.csv")
    except FileNotFoundError:
        st.error(
            "❌ **amazon_music_clustered.csv not found!**\n\n"
            "Please make sure the file is in the same folder as this script."
        )
        st.stop()

df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Cluster Distribution", "Cluster Profiles", "PCA Visualization", "Genres & Artists","🎧 Song Recommendation"]
)

# -----------------------------
# OVERVIEW
# -----------------------------
if section == "Overview":
    st.subheader("📌 Project Overview")

    st.markdown("""
    **Objective:**  
    Automatically group similar songs based on audio characteristics such as  
    danceability, energy, loudness, tempo, etc.

    **Clustering Algorithm:**  
    - K-Means (k = 4)
    - Features scaled using StandardScaler

    **Why k = 4?**
    - Supported by Elbow Method
    - Strong Silhouette Score (~0.23)
    - Clear musical interpretability
    """)

    st.subheader("Dataset Snapshot")
    st.dataframe(df.head())

# -----------------------------
# CLUSTER DISTRIBUTION
# -----------------------------
elif section == "Cluster Distribution":
    st.subheader("📊 Cluster Size Distribution")

    cluster_counts = df["clusters"].value_counts(normalize=True) * 100
    st.write(cluster_counts.round(2))

    fig, ax = plt.subplots()
    sns.barplot(
        x=cluster_counts.index,
        y=cluster_counts.values,
        ax=ax
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage of Songs")
    ax.set_title("Cluster Distribution (%)")

    st.pyplot(fig)

# -----------------------------
# CLUSTER PROFILES
# -----------------------------
elif section == "Cluster Profiles":
    st.subheader("🎼 Cluster Audio Profiles")

    audio_features = [
        "duration_ms", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]

    cluster_profile = (
        df.groupby("clusters")[audio_features]
        .mean()
        .round(3)
    )

    st.dataframe(cluster_profile)

    st.markdown("""
    **Cluster Interpretation:**
    - **Cluster 0:** Instrumental / Focus Music  
    - **Cluster 1:** Party / Energetic Music  
    - **Cluster 2:** Rap / Spoken Content  
    - **Cluster 3:** Chill / Acoustic Music  
    """)

# -----------------------------
# PCA VISUALIZATION
# -----------------------------
elif section == "PCA Visualization":
    st.subheader("📉 PCA Cluster Visualization")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    audio_features = [
        "duration_ms", "danceability", "energy", "loudness",
        "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo"
    ]

    X = df[audio_features]

    # Scale again (safe for visualization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df["pca1"] = X_pca[:, 0]
    df["pca2"] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x="pca1",
        y="pca2",
        hue="clusters",
        data=df,
        palette="Set2",
        s=20,
        ax=ax
    )
    ax.set_title("Clusters Visualized using PCA")
    st.pyplot(fig)


# -----------------------------
# GENRES & ARTISTS
# -----------------------------
elif section == "Genres & Artists":
    st.subheader("🎧 Genres & Artists by Cluster")

    selected_cluster = st.selectbox(
        "Select Cluster",
        sorted(df["clusters"].unique())
    )

    st.markdown(f"### Top Genres in Cluster {selected_cluster}")
    genre_data = (
        df[df["clusters"] == selected_cluster]["genres"]
        .value_counts()
        .head(10)
    )
    st.write(genre_data)

    st.markdown(f"### Top Artists in Cluster {selected_cluster}")
    artist_data = (
        df[df["clusters"] == selected_cluster]["name_artists"]
        .value_counts()
        .head(10)
    )
    st.write(artist_data)
elif section == "🎧 Song Recommendation":
    st.subheader("🎵 Song Recommendation System")

    st.markdown("""
    Select a song and get recommendations based on **audio similarity**
    using **K-Means clustering**.
    """)

    # Make sure required columns exist
    required_cols = {"name_song", "name_artists", "clusters"}
    if not required_cols.issubset(df.columns):
        st.error("Required columns for recommendation not found.")
    else:
        # Song selector
        song = st.selectbox(
            "🎶 Choose a song",
            df["name_song"].sort_values().unique()
        )

        # Get selected song info
        song_row = df[df["name_song"] == song].iloc[0]
        song_cluster = song_row["clusters"]
        artist = song_row["name_artists"]

        st.markdown(
            f"""
            **Selected Song:** {song}  
            **Artist:** {artist}  
            **Cluster:** {song_cluster}
            """
        )

        # Recommend similar songs from same cluster
        recommendations = (
            df[
                (df["clusters"] == song_cluster) &
                (df["name_song"] != song)
            ]
            .sample(n=10, random_state=42)
            [["name_song", "name_artists"]]
        )

        st.subheader("🎧 Recommended Songs")
        st.dataframe(recommendations.reset_index(drop=True))


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit | Amazon Music Clustering Project**")

if __name__ == "__main__":
    import os
    os.system("streamlit run [streamlit_page.py](http://_vscodecontentref_/0)")