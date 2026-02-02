import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "TMDB_movie_dataset_v11.csv"

# DATA_PATH = "C:\Users\Himit\OneDrive\Desktop\movie-recommender\recommender\data\TMDB_movie_dataset_v11.csv"  # change path

print("Loading dataset...")

cols_needed = [
    "id",
    "title",
    "overview",
    "genres",
    "keywords",
    "original_language",
    "popularity",
    "poster_path"
]

df = pd.read_csv(DATA_PATH, usecols=cols_needed, low_memory=False)

print("Initial shape:", df.shape)

# Drop rows missing critical text
df = df.dropna(subset=["title", "overview"])

# Remove duplicates
df = df.drop_duplicates(subset=["id"])

print("After cleaning:", df.shape)

# Save cleaned version
df.to_csv("clean_movies.csv", index=False)

print("✅ Saved clean_movies.csv")
