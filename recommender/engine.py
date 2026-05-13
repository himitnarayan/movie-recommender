import pickle
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from pathlib import Path
import faiss
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ARTIFACTS = Path("recommender/artifacts")

# Global placeholders
tfidf = None
nn_tfidf = None
tfidf_matrix = None
movies = None
title_to_index = None
faiss_index = None
embedder = None

def load_model():
    global tfidf, nn_tfidf, tfidf_matrix, movies, title_to_index, faiss_index, embedder

    if tfidf is None:
        print("Loading model artifacts...")

        tfidf = pickle.load(open(ARTIFACTS / "tfidf.pkl", "rb"))
        nn_tfidf = pickle.load(open(ARTIFACTS / "nn_tfidf.pkl", "rb"))
        tfidf_matrix = load_npz(ARTIFACTS / "tfidf_vectors.npz")
        movies = pd.read_csv(ARTIFACTS / "movie_index.csv")
        
        # Load Neural Embeddings
        faiss_index = faiss.read_index(str(ARTIFACTS / "faiss_index.bin"))
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        title_to_index = pd.Series(
            movies.index, index=movies['title']
        ).drop_duplicates()

        # Configure Gemini for explainability
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

def generate_explanation(query_title, recommended_title, recommended_overview):
    try:
        if not os.environ.get("GOOGLE_API_KEY"):
            return "Similarity match."
            
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"In one short sentence, explain why someone who likes the movie '{query_title}' would also like the movie '{recommended_title}'. The overview of {recommended_title} is: {recommended_overview[:300]}. Keep it very brief and engaging."
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Because it shares similar themes, genres, or keywords."

def recommend(movie_title, top_n=10, algorithm="hybrid"):
    load_model()

    if movie_title not in title_to_index:
        return {"error": f"'{movie_title}' not found in database."}

    idx = title_to_index[movie_title]
    query_overview = movies.iloc[idx]['overview']

    scores = {}

    # 1. TF-IDF Search (Keyword Matching)
    distances_tfidf, indices_tfidf = nn_tfidf.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=top_n * 2
    )
    
    if algorithm in ["tfidf", "hybrid"]:
        for rank, i in enumerate(indices_tfidf[0]):
            if i == idx: continue
            weight = 1.0 if algorithm == "tfidf" else 0.4
            scores[i] = scores.get(i, 0) + (1.0 / (rank + 1)) * weight

    # 2. Semantic Search (FAISS)
    if algorithm == "hybrid":
        query_text = f"{movie_title}: {query_overview}"
        query_vector = embedder.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        distances_faiss, indices_faiss = faiss_index.search(query_vector, top_n * 2)
        
        for rank, i in enumerate(indices_faiss[0]):
            if i == idx: continue
            scores[i] = scores.get(i, 0) + (1.0 / (rank + 1)) * 0.6

    # Sort by score descending
    sorted_indices = sorted(scores, key=scores.get, reverse=True)[:top_n]

    results = []

    for i in sorted_indices:
        row = movies.iloc[i]

        poster_url = ""
        if isinstance(row['poster_path'], str):
            poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"

        # Generate Explainability Reason only for Hybrid mode to save API calls
        explanation = "Similarity match."
        if algorithm == "hybrid" and len(results) == 0:
             explanation = generate_explanation(movie_title, row['title'], row['overview'])

        results.append({
            "title": row['title'],
            "poster": poster_url,
            "link": f"https://www.themoviedb.org/movie/{int(row['id'])}",
            "explanation": explanation
        })

    return results
