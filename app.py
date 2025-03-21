import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO
import logging

# Suppress Streamlit warnings
logging.getLogger("streamlit").setLevel(logging.ERROR)


# Load the dataset
@st.cache_data
def load_data():
    try:
        # Replace this with your actual dataset path
        data = pd.read_csv("anime_dataset_preprocessed.csv")
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


# Preprocess the data
def preprocess_data(data):
    if data.empty:
        return data

    # Ensure required columns exist
    required_columns = [
        "title",
        "main_pic",
        "score_count",
        "score_rank",
        "score_10_count",
        "score_09_count",
        "score_08_count",
        "score_07_count",
        "score_06_count",
        "score_05_count",
        "score_04_count",
        "score_03_count",
        "score_02_count",
        "score_01_count",
    ]
    for col in required_columns:
        if col not in data.columns:
            st.error(f"Column '{col}' is missing in the dataset.")
            return pd.DataFrame()

    # Fill missing numerical values with 0
    numerical_columns = [
        "score_count",
        "score_rank",
        "score_10_count",
        "score_09_count",
        "score_08_count",
        "score_07_count",
        "score_06_count",
        "score_05_count",
        "score_04_count",
        "score_03_count",
        "score_02_count",
        "score_01_count",
    ]
    data[numerical_columns] = data[numerical_columns].fillna(0)

    return data


# Compute similarity based on numerical features
def compute_similarity(data):
    if data.empty:
        return None

    try:
        # Select numerical features for similarity computation
        numerical_features = data[
            [
                "score_count",
                "score_rank",
                "score_10_count",
                "score_09_count",
                "score_08_count",
                "score_07_count",
                "score_06_count",
                "score_05_count",
                "score_04_count",
                "score_03_count",
                "score_02_count",
                "score_01_count",
            ]
        ]

        # Compute cosine similarity
        cosine_sim = cosine_similarity(numerical_features, numerical_features)
        return cosine_sim
    except Exception as e:
        st.error(f"Error computing similarity: {e}")
        return None


# Get recommendations based on anime title
def get_recommendations(title, data, cosine_sim, num_recommendations=5):
    if data.empty or cosine_sim is None:
        return pd.DataFrame()

    try:
        # Find the index of the anime
        idx = data[data["title"] == title].index[0]

        # Get pairwise similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top N recommendations (excluding itself)
        sim_scores = sim_scores[1 : num_recommendations + 1]

        # Get anime indices
        anime_indices = [i[0] for i in sim_scores]

        # Return top recommendations
        return data.iloc[anime_indices]
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return pd.DataFrame()


# Function to display anime image and title
def display_anime(title, image_url):
    try:
        st.write(f"**{title}**")
        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            st.image(img, caption=title, width=200)
        else:
            st.error(f"Failed to load image for {title}")
    except Exception as e:
        st.error(f"Error displaying anime: {e}")


# Streamlit app
def main():
    st.title("Anime Recommendation System (Content-Based)")

    # Load and preprocess data
    data = load_data()
    if data.empty:
        st.error("No dataset loaded. Please check the file path and try again.")
        return

    data = preprocess_data(data)

    # Compute similarity matrix
    cosine_sim = compute_similarity(data)
    if cosine_sim is None:
        st.error("Failed to compute similarity matrix. Please check your dataset.")
        return

    # User input for anime title
    anime_title = st.selectbox("Select an Anime", data["title"].unique())

    # Get recommendations
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(anime_title, data, cosine_sim)

        if recommendations.empty:
            st.warning("No recommendations found. Please check your dataset.")
        else:
            # Display top recommendations
            st.subheader("Top Recommendations:")
            for _, row in recommendations.iterrows():
                display_anime(row["title"], row["main_pic"])


# Run the app
if __name__ == "__main__":
    main()
