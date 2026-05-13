import os
import glob
import pickle
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz
import mlflow
import faiss
from sentence_transformers import SentenceTransformer

ARTIFACT_DIR = "recommender/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------- Find CSV automatically ----------------
print("Searching for CSV dataset...")
csv_files = glob.glob("recommender/data/*.csv")

if not csv_files:
    raise FileNotFoundError("No CSV file found in recommender/data")

DATA_PATH = csv_files[0]
print("Using dataset:", DATA_PATH)

# ---------------- Load required columns ----------------
cols = ["id", "title", "overview", "genres", "keywords", "poster_path", "popularity"]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
df = df.dropna(subset=["title", "overview"]).drop_duplicates("id")

# ---------------- Reduce dataset for speed (IMPORTANT) ----------------
print("Reducing dataset size for fast processing...")
df = df.sort_values(by="popularity", ascending=False).head(10000)
df = df.reset_index(drop=True)
print("Dataset reduced to:", len(df))

# ---------------- Helper to parse JSON columns ----------------
def parse_names(text):
    try:
        items = literal_eval(text)
        return [i['name'].replace(" ", "").lower() for i in items]
    except:
        return []

print("Parsing genres and keywords...")
df['genres'] = df['genres'].apply(parse_names)
df['keywords'] = df['keywords'].apply(parse_names)
df['overview'] = df['overview'].str.lower()

# ---------------- Create metadata soup ----------------
print("Creating metadata soup...")
df['soup'] = (
    df['overview'] + " " +
    df['genres'].apply(lambda x: " ".join(x)) + " " +
    df['keywords'].apply(lambda x: " ".join(x))
)

# ==============================================================================
# MLOps: Start MLflow Tracking
# ==============================================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Movie_Recommender_System")

with mlflow.start_run(run_name="Hybrid_TFIDF_SentenceTransformers"):
    mlflow.log_param("dataset_size", len(df))
    mlflow.log_param("tfidf_max_features", 30000)
    mlflow.log_param("embedding_model", "all-MiniLM-L6-v2")

    # ---------------- 1. TF-IDF (memory safe) ----------------
    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=30000)
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    
    nn_tfidf = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_tfidf.fit(tfidf_matrix)

    # ---------------- 2. Neural Embeddings (Sentence Transformers) ----------------
    print("Generating Neural Embeddings (This may take a few minutes)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # For speed on CPU, we embed just the overview and genres instead of the full soup
    sentences = df['title'] + ": " + df['overview']
    embeddings = embedder.encode(sentences.tolist(), convert_to_numpy=True, show_progress_bar=True)
    
    # Create FAISS Index for fast vector search
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension) # Inner product (Cosine Similarity if normalized)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)

    # ---------------- Save artifacts ----------------
    print("Saving artifacts...")
    save_npz(f"{ARTIFACT_DIR}/tfidf_vectors.npz", tfidf_matrix)
    pickle.dump(nn_tfidf, open(f"{ARTIFACT_DIR}/nn_tfidf.pkl", "wb"))
    pickle.dump(tfidf, open(f"{ARTIFACT_DIR}/tfidf.pkl", "wb"))
    faiss.write_index(faiss_index, f"{ARTIFACT_DIR}/faiss_index.bin")
    
    df[['id', 'title', 'poster_path', 'popularity', 'overview']].to_csv(
        f"{ARTIFACT_DIR}/movie_index.csv", index=False
    )

    # Log basic artifact paths to mlflow
    mlflow.log_artifact(f"{ARTIFACT_DIR}/faiss_index.bin")
    mlflow.log_metric("features_extracted", tfidf_matrix.shape[1])
    
print("✅ Hybrid Model build complete.")
