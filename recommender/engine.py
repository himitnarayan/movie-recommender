import pickle
import pandas as pd
from scipy.sparse import load_npz
from pathlib import Path
print("Loading recommender artifacts...")

# ---------- Load saved files ----------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"

tfidf = pickle.load(open(ARTIFACTS / "tfidf.pkl", "rb"))
nn_model = pickle.load(open(ARTIFACTS / "nn_model.pkl", "rb"))
tfidf_matrix = load_npz(ARTIFACTS / "tfidf_vectors.npz")
movies = pd.read_csv(ARTIFACTS / "movie_index.csv")
# Create title → index mapping
title_to_index = pd.Series(movies.index, index=movies['title']).drop_duplicates()

print("Recommender ready.")


# ---------- Recommendation Function ----------

def recommend(movie_title, top_n=10):
    movie_title = movie_title.strip()

    if movie_title not in title_to_index:
       return {"error": f"'{movie_title}' not found in database."}

    idx = title_to_index[movie_title]

    # Get nearest neighbors
    distances, indices = nn_model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=top_n + 1
    )

    results = []

    for i in indices[0][1:]:  # skip itself
        row = movies.iloc[i]

        poster_url = ""
        if isinstance(row['poster_path'], str):
            poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"

        results.append({
        "title": row['title'],
        "tmdb_id": int(row['id']),
        "poster": poster_url,
        "popularity": float(row['popularity']),
        "link": f"https://www.themoviedb.org/movie/{int(row['id'])}"  # ✅ added
    })

    return results



# ---------- Test locally ----------
if __name__ == "__main__":
    res = recommend("Inception")
    for r in res:
        print(r)
