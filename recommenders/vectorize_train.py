import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz

print("Loading soups...")
df = pd.read_csv("movies_soup.csv")

print("Number of movies:", len(df))

# ---------- TF-IDF Vectorization ----------

print("Vectorizing text with TF-IDF...")

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=50000   # VERY IMPORTANT for memory control
)

tfidf_matrix = tfidf.fit_transform(df['soup'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# ---------- Train Nearest Neighbors ----------

print("Training NearestNeighbors model...")

nn = NearestNeighbors(
    n_neighbors=11,
    metric='cosine',
    algorithm='brute'
)

nn.fit(tfidf_matrix)

print("Model training complete.")

# ---------- Save Artifacts ----------

print("Saving artifacts...")

# Save sparse vectors
save_npz("tfidf_vectors.npz", tfidf_matrix)

# Save model
pickle.dump(nn, open("nn_model.pkl", "wb"))

# Save TF-IDF object
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

# Save movie index data
df[['id', 'title', 'poster_path', 'popularity']].to_csv("movie_index.csv", index=False)

print("✅ Step 3 complete. Files saved:")
print("- tfidf_vectors.npz")
print("- nn_model.pkl")
print("- tfidf.pkl")
print("- movie_index.csv")
