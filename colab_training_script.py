"""
GOOGLE COLAB TRAINING SCRIPT (v2 - Fixed Embedding Space)
-----------------------------------------------------------
IMPORTANT: This script uses Pinecone's own Inference API (llama-text-embed-v2)
to generate embeddings. This MUST match what the Django app uses at query time.

Steps:
1. Go to https://colab.research.google.com/
2. Create a new notebook.
3. Runtime -> Change runtime type -> T4 GPU (optional but faster).
4. Upload your kaggle.json to the Colab file browser.
5. Copy and paste this entire script into a cell and run it!

Get your free API key from https://www.pinecone.io/
"""

# --- 0. INSTALL DEPENDENCIES ---
# !pip install pandas pinecone pinecone-text kaggle tqdm

import os
import csv
import subprocess
import pandas as pd
from ast import literal_eval
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm

# --- 1. SETUP API KEYS ---
# Replace with your actual Pinecone API key
os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
INDEX_NAME = "movie-hybrid-search"
DIMENSION  = 384  # llama-text-embed-v2 with dimension=384

# --- 2. DOWNLOAD DATASET FROM KAGGLE ---
# Make sure you uploaded kaggle.json to the Colab file browser first!
print("Downloading Kaggle Dataset...")
subprocess.run("mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json", shell=True)
subprocess.run("kaggle datasets download -d asaniczka/tmdb-movies-dataset-2023-930k-movies --unzip", shell=True)

print("Loading dataset...")
df = pd.read_csv("TMDB_movie_dataset_v11.csv", engine='python', on_bad_lines='skip')
df.columns = df.columns.str.replace('"', '').str.strip()

# Clean data — drop missing titles/overviews and rows corrupted by CSV parsing
df['id'] = pd.to_numeric(df['id'], errors='coerce')
df = df.dropna(subset=["id", "title", "overview"]).drop_duplicates("id")
df['id'] = df['id'].astype(int).astype(str)
print(f"Loaded {len(df):,} clean movies.")

# Optional: Test with top 50,000 most popular first, then remove limit for full run
# df = df.sort_values(by="popularity", ascending=False).head(50000)

# --- 3. BUILD MOVIE "SOUP" (combined text for embedding) ---
def parse_names(text):
    try:
        items = literal_eval(text)
        return " ".join([i['name'] for i in items if isinstance(i, dict)])
    except:
        return ""

df['genres_text'] = df['genres'].apply(parse_names)
df['soup'] = (df['title'] + " " + df['genres_text'] + " " + df['overview'].fillna("")).str.strip()

# --- 4. INITIALIZE PINECONE ---
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Delete old index if it exists with wrong dimension, then recreate
existing = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME in existing:
    print(f"Deleting existing index '{INDEX_NAME}' to rebuild with correct model...")
    pc.delete_index(INDEX_NAME)

print(f"Creating new index '{INDEX_NAME}' with dimension={DIMENSION}...")
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric="dotproduct",  # Required for hybrid search
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

# Wait until index is ready
import time
while not pc.describe_index(INDEX_NAME).status['ready']:
    print("Waiting for index to be ready...")
    time.sleep(5)

index = pc.Index(INDEX_NAME)
print("Index is ready!")

# --- 5. FIT BM25 SPARSE MODEL ---
print("Fitting BM25 Sparse Model...")
bm25 = BM25Encoder()
bm25.fit(df['soup'])
# Download this file and place it in your Django app at recommender/artifacts/bm25_params.json
bm25.dump("bm25_params.json")
print("BM25 model saved to bm25_params.json — download and commit this to your repo!")

# --- 6. ENCODE AND UPLOAD TO PINECONE ---
print("Encoding and Uploading to Pinecone...")
# We use batches of 96 because Pinecone Inference API handles up to 96 inputs per call
batch_size = 96

for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i:i+batch_size]
    soups    = batch_df['soup'].tolist()
    ids      = batch_df['id'].tolist()

    # 1. Generate Dense Vectors via Pinecone Inference API (same model as Django app!)
    try:
        embeddings = pc.inference.embed(
            model="llama-text-embed-v2",
            inputs=soups,
            parameters={"input_type": "passage", "truncate": "END", "dimension": DIMENSION}
        )
        dense_vectors = [e.values for e in embeddings]
    except Exception as e:
        print(f"Embedding error at batch {i}: {e}")
        continue

    # 2. Generate Sparse Vectors (BM25)
    sparse_vectors = bm25.encode_documents(soups)

    # 3. Collect Metadata
    metadata = []
    for _, row in batch_df.iterrows():
        poster = row.get('poster_path', '')
        metadata.append({
            "title":       str(row['title']),
            "overview":    str(row['overview'])[:1000],  # Trim long overviews
            "poster_path": str(poster) if pd.notna(poster) else ""
        })

    # 4. Build Records and Upsert
    records = []
    for j in range(len(ids)):
        record = {
            "id":     str(ids[j]),
            "values": dense_vectors[j],
            "metadata": metadata[j]
        }
        # Only add sparse values if BM25 found valid tokens
        sv = sparse_vectors[j]
        if sv and len(sv.get("indices", [])) > 0:
            record["sparse_values"] = sv
        records.append(record)

    try:
        index.upsert(vectors=records)
    except Exception as e:
        print(f"Upsert error at batch {i}: {e}")

print(f"✅ Successfully uploaded {len(df):,} movies to Pinecone!")
print("📥 Don't forget to download bm25_params.json and commit it to recommender/artifacts/!")
