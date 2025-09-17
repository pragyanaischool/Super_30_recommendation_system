import streamlit as st
import pandas as pd
import joblib

# --- Function to load saved model assets ---
# The @st.cache_resource decorator ensures this function runs only once
@st.cache_resource
def load_model_assets():
    """
    Loads the pre-saved movie DataFrame and cosine similarity matrix.
    """
    try:
        movies_df = joblib.load('movies_df.joblib')
        cosine_sim = joblib.load('cosine_sim_matrix.joblib')
        return movies_df, cosine_sim
    except FileNotFoundError:
        # Return None if files are not found
        return None, None

# --- Recommendation Logic ---
def get_recommendations(title, movies_df, cosine_sim):
    """
    Generates movie recommendations for a given title using the pre-computed
    similarity matrix.
    """
    # Create a mapping of movie titles to their index in the DataFrame
    indices = pd.Series(movies_df.index, index=movies_df['title'])
    
    # Get the index of the movie that matches the input title
    idx = indices[title]
    
    # Get the pairwise similarity scores for that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on their similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies (excluding the input movie itself)
    sim_scores = sim_scores[1:11]
    
    # Get the indices of the top movies
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the titles of the top 10 most similar movies
    return movies_df['title'].iloc[movie_indices]

# --- Streamlit User Interface ---

# Set up the page title and a brief description
st.title("🎬 Movie Recommender System")
st.markdown("Select a movie from the dropdown below to get 10 similar movie recommendations.")

# Load the assets
movies_df, cosine_sim = load_model_assets()

# Check if the model files were loaded successfully
if movies_df is not None and cosine_sim is not None:
    # Create the user interface elements
    movie_options = movies_df['title'].sort_values().unique()
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        movie_options
    )

    if st.button("Recommend"):
        with st.spinner('Finding similar movies for you...'):
            try:
                recommendations = get_recommendations(selected_movie, movies_df, cosine_sim)
                st.subheader("Here are some movies you might enjoy:")
                for i, movie in enumerate(recommendations):
                    st.success(f"{i+1}. {movie}")
            except KeyError:
                st.error("Could not find the selected movie. Please try another.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
else:
    # Display an error message if the model files are missing
    st.error("🚨 Model files not found!")
    st.warning("Please run the training script first to generate 'movies_df.joblib' and 'cosine_sim_matrix.joblib'.")
