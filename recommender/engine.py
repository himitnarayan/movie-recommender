import pickle
import pandas as pd
from scipy.sparse import load_npz
from pathlib import Path

ARTIFACTS = Path("recommender/artifacts")

# Global placeholders
tfidf = None
nn_model = None
tfidf_matrix = None
movies = None
title_to_index = None


def load_model():
    global tfidf, nn_model, tfidf_matrix, movies, title_to_index

    if tfidf is None:
        print("Loading model artifacts...")

        tfidf = pickle.load(open(ARTIFACTS / "tfidf.pkl", "rb"))
        nn_model = pickle.load(open(ARTIFACTS / "nn_model.pkl", "rb"))
        tfidf_matrix = load_npz(ARTIFACTS / "tfidf_vectors.npz")
        movies = pd.read_csv(ARTIFACTS / "movie_index.csv")

        title_to_index = pd.Series(
            movies.index, index=movies['title']
        ).drop_duplicates()


def recommend(movie_title, top_n=10):
    load_model()   # 👈 load only when needed

    if movie_title not in title_to_index:
        return {"error": f"'{movie_title}' not found in database."}

    idx = title_to_index[movie_title]

    distances, indices = nn_model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=top_n + 1
    )

    results = []

    for i in indices[0][1:]:
        row = movies.iloc[i]

        poster_url = ""
        if isinstance(row['poster_path'], str):
            poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"

        results.append({
            "title": row['title'],
            "poster": poster_url,
            "link": f"https://www.themoviedb.org/movie/{int(row['id'])}"
        })

    return results
