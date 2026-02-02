import os
import glob
import pickle
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz

ARTIFACT_DIR = "recommender/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("Searching for CSV dataset...")
csv_files = glob.glob("recommender/data/*.csv")

if not csv_files:
    raise FileNotFoundError("No CSV file found in recommender/data")

DATA_PATH = csv_files[0]
print("Using dataset:", DATA_PATH)

# ---------------- Load only required columns ----------------
cols = [
    "id",
    "title",
    "overview",
    "genres",
    "keywords",
    "poster_path",
    "popularity"
]

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)

df = df.dropna(subset=["title", "overview"]).drop_duplicates("id")

# ---------------- Helper functions ----------------
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

# ---------------- TF-IDF (memory safe) ----------------
print("Vectorizing with TF-IDF...")
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=30000   # important for Render RAM
)

tfidf_matrix = tfidf.fit_transform(df['soup'])

# ---------------- Train Nearest Neighbors ----------------
print("Training NearestNeighbors...")
nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

# ---------------- Save artifacts ----------------
print("Saving artifacts...")
save_npz(f"{ARTIFACT_DIR}/tfidf_vectors.npz", tfidf_matrix)
pickle.dump(nn, open(f"{ARTIFACT_DIR}/nn_model.pkl", "wb"))
pickle.dump(tfidf, open(f"{ARTIFACT_DIR}/tfidf.pkl", "wb"))

df[['id', 'title', 'poster_path', 'popularity']].to_csv(
    f"{ARTIFACT_DIR}/movie_index.csv",
    index=False
)

print("✅ Model build complete.")
