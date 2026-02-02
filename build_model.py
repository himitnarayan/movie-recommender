import os
import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import save_npz
import pickle

DATA_PATH = "recommenders/data/TMDB_movie_dataset_v11.csv"
ARTIFACT_DIR = "recommender/artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("Loading dataset...")
cols = [
    "id", "title", "overview",
    "genres", "keywords",
    "poster_path", "popularity"
]

df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=False)
df = df.dropna(subset=["title", "overview"]).drop_duplicates("id")

print("Parsing metadata...")

def parse_names(text):
    try:
        items = literal_eval(text)
        return [i['name'].replace(" ", "").lower() for i in items]
    except:
        return []

df['genres'] = df['genres'].apply(parse_names)
df['keywords'] = df['keywords'].apply(parse_names)
df['overview'] = df['overview'].str.lower()

df['soup'] = (
    df['overview'] + " " +
    df['genres'].apply(lambda x: " ".join(x)) + " " +
    df['keywords'].apply(lambda x: " ".join(x))
)

print("Vectorizing...")

tfidf = TfidfVectorizer(stop_words='english', max_features=50000)
tfidf_matrix = tfidf.fit_transform(df['soup'])

print("Training NearestNeighbors...")

nn = NearestNeighbors(metric='cosine', algorithm='brute')
nn.fit(tfidf_matrix)

print("Saving artifacts...")

save_npz(f"{ARTIFACT_DIR}/tfidf_vectors.npz", tfidf_matrix)
pickle.dump(nn, open(f"{ARTIFACT_DIR}/nn_model.pkl", "wb"))
pickle.dump(tfidf, open(f"{ARTIFACT_DIR}/tfidf.pkl", "wb"))

df[['id', 'title', 'poster_path', 'popularity']].to_csv(
    f"{ARTIFACT_DIR}/movie_index.csv", index=False
)

print("✅ Model build complete.")
