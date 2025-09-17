# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from nltk.stem.snowball import SnowballStemmer

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# --- Data Loading and Preprocessing with Caching ---
@st.cache_resource
def load_and_prepare_data():
    """
    Loads data, preprocesses it, computes the cosine similarity matrix,
    and prepares all necessary components for the recommender.
    This function is cached to run only once.
    """
    # Load datasets
    md = pd.read_csv('movies_metadata.csv', low_memory=False)
    credits = pd.read_csv('credits.csv')
    keywords = pd.read_csv('keywords.csv')

    # Data cleaning and merging
    md.drop([19730, 29503, 35587], inplace=True) # Drop bad rows
    md['id'] = pd.to_numeric(md['id'], errors='coerce').astype('int')
    keywords['id'] = pd.to_numeric(keywords['id'], errors='coerce').astype('int')
    credits['id'] = pd.to_numeric(credits['id'], errors='coerce').astype('int')
    
    md = md.merge(credits, on='id')
    md = md.merge(keywords, on='id')
    
    # Process genres, cast, crew, and keywords
    for feature in ['genres', 'cast', 'crew', 'keywords']:
        md[feature] = md[feature].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    md['director'] = md['crew'].apply(get_director)
    
    for feature in ['cast', 'keywords', 'genres']:
        md[feature] = md[feature].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    
    # Keep top 3 actors
    md['cast'] = md['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)

    # Clean and stem keywords
    stemmer = SnowballStemmer('english')
    md['keywords'] = md['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    
    # Sanitize and combine features into a "soup"
    def sanitize(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        elif isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

    for feature in ['cast', 'keywords', 'director', 'genres']:
        md[feature] = md[feature].apply(sanitize)

    # Give more weight to the director
    md['director'] = md['director'].apply(lambda x: [x, x, x])
    
    # Create the metadata soup
    md['soup'] = md['keywords'] + md['cast'] + md['director'] + md['genres']
    md['soup'] = md['soup'].apply(lambda x: ' '.join(x))

    # Create the count matrix and cosine similarity matrix
    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    count_matrix = count.fit_transform(md['soup'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    # Reset index for title mapping
    smd = md.reset_index()
    titles = smd['title']
    indices = pd.Series(smd.index, index=smd['title'])
    
    return smd, cosine_sim, titles, indices

# Load all assets
smd, cosine_sim, titles, indices = load_and_prepare_data()

# --- Weighted Rating Calculation ---
vote_counts = smd[smd['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = smd[smd['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()

def weighted_rating(x, m):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

# --- Improved Recommendation Function ---
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]  # Get top 25 similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    
    # Filter based on vote count percentile
    m = vote_counts.quantile(0.60)
    qualified = movies[movies['vote_count'] >= m]
    
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    
    qualified['wr'] = qualified.apply(lambda x: weighted_rating(x, m), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    
    return qualified

# --- Streamlit User Interface ---
st.title("🎬 Movie Recommender System")
st.markdown("Discover movies similar to your favorites! This app uses a content-based recommender that analyzes movie metadata like genre, director, cast, and keywords.")

movie_options = titles.sort_values().unique()
selected_movie = st.selectbox(
    "Choose a movie you like:",
    movie_options
)

if st.button("Recommend Movies"):
    with st.spinner('Finding recommendations...'):
        try:
            recommendations = improved_recommendations(selected_movie)
            st.subheader(f"Top 10 movies similar to '{selected_movie}':")
            
            # Display results in a clean format
            for index, row in recommendations.iterrows():
                st.success(f"**{row['title']}** ({int(row['year'])}) - Rating: {row['vote_average']}/10")

        except KeyError:
            st.error("Movie not found in the dataset. Please try another one.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
