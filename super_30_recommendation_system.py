import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np

# --- Page Configuration ---
st.set_page_config(layout="wide")

# --- Function to fetch movie poster ---
def fetch_poster(movie_id):
    """Fetches a movie poster URL from the TMDB API using a secret key."""
    try:
        api_key = st.secrets["TMDB_API_KEY"]
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception as e:
        print(f"Error fetching poster: {e}") # Log error for debugging
    return "https://via.placeholder.com/500x750.png?text=No+Poster+Found"

# --- Function to load saved model assets ---
@st.cache_resource
def load_model_assets():
    """Loads the pre-saved movie DataFrame and cosine similarity matrix."""
    try:
        movies_df = joblib.load('movies_df.joblib')
        cosine_sim = joblib.load('cosine_sim_matrix.joblib')
        return movies_df, cosine_sim
    except FileNotFoundError:
        return None, None

# --- Recommendation Logic ---
def get_recommendations(title, movies_df, cosine_sim):
    """Generates movie recommendations for a given title."""
    indices = pd.Series(movies_df.index, index=movies_df['title'])
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx].astype(float)))
    
    def robust_sort_key(item):
        score = item[1]
        if isinstance(score, (int, float, np.number)) and np.isfinite(score):
            return score
        else:
            return -float('inf')

    sim_scores = sorted(sim_scores, key=robust_sort_key, reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices], movies_df['id'].iloc[movie_indices]

# --- Custom CSS and UI ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stSelectbox div[data-baseweb="select"] > div { background-color: #262730; }
    .stButton>button { background-color: #4B8BBE; color: white; border-radius: 8px; border: none; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; }
    .movie-title { font-size: 14px; font-weight: bold; text-align: center; margin-top: 8px; height: 50px; overflow: hidden; text-overflow: ellipsis; }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender System")
movies_df, cosine_sim = load_model_assets()

if movies_df is not None and cosine_sim is not None:
    st.sidebar.header("Find Your Next Movie")
    movie_options = movies_df['title'].sort_values().unique()
    selected_movie = st.sidebar.selectbox("Select a movie you've enjoyed:", movie_options)
    
    if st.sidebar.button("Recommend"):
        if selected_movie:
            with st.spinner('Curating a list of movies just for you...'):
                rec_titles, rec_ids = get_recommendations(selected_movie, movies_df, cosine_sim)
                
                # THE FIX IS HERE: Check the number of recommendations before displaying
                num_recs = len(rec_titles)
                
                if num_recs > 0:
                    st.subheader(f"Because you watched '{selected_movie}', you might like...")
                    
                    # Create a dynamic grid that adapts to the number of recommendations
                    cols = st.columns(5)
                    for i in range(num_recs):
                        col = cols[i % 5]
                        with col:
                            st.image(fetch_poster(rec_ids.iloc[i]))
                            st.markdown(f"<p class='movie-title'>{rec_titles.iloc[i]}</p>", unsafe_allow_html=True)
                else:
                    st.warning("Could not find any similar movies.")
else:
    st.error("🚨 Model files not found!")
    st.warning("Please run the updated training script first.")
