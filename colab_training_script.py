"""
GOOGLE COLAB TRAINING SCRIPT
-----------------------------
1. Go to https://colab.research.google.com/
2. Create a new notebook.
3. Go to Runtime -> Change runtime type -> Select T4 GPU.
4. Copy and paste this entire script into a cell and run it!

Make sure you get a free API key from https://www.pinecone.io/
"""

# !pip install pandas sentence-transformers pinecone pinecone-text kaggle

import os
import pandas as pd
from ast import literal_eval
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from tqdm.auto import tqdm

# --- 1. SETUP API KEYS ---
# Replace with your actual Pinecone API key
os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"
INDEX_NAME = "movie-hybrid-search"

# --- 2. DOWNLOAD DATASET ---
# Because this dataset is large, we download it directly using Kaggle
# You will need to upload your kaggle.json API key file to Colab first!
import subprocess
print("Downloading Kaggle Dataset...")
subprocess.run("mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json", shell=True)
subprocess.run("kaggle datasets download -d asaniczka/tmdb-movies-dataset-2023-930k-movies --unzip", shell=True)

print("Loading dataset...")
df = pd.read_csv("TMDB_movie_dataset_v11.csv")

# We drop rows without an overview or title to keep data clean
df = df.dropna(subset=["title", "overview"]).drop_duplicates("id")

# If you want to test fast, you can limit the rows to the top 100k most popular:
# df = df.sort_values(by="popularity", ascending=False).head(100000)


def parse_names(text):
    try:
        items = literal_eval(text)
        return [i['name'] for i in items]
    except:
        return []

df['genres_text'] = df['genres'].apply(lambda x: " ".join(parse_names(x)))
df['soup'] = df['title'] + " " + df['genres_text'] + " " + df['overview'].fillna("")

# --- 3. INITIALIZE PINECONE ---
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # Dimension for all-MiniLM-L6-v2
        metric="dotproduct", # Required for hybrid search
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# --- 4. PREPARE HYBRID MODELS ---
print("Loading Dense Embedding Model (GPU)...")
dense_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

print("Fitting BM25 Sparse Model...")
bm25 = BM25Encoder()
bm25.fit(df['soup'])
# Save this JSON file and put it in your Django app's recommender/artifacts/ folder!
bm25.dump("bm25_params.json") 

# --- 5. UPLOAD TO PINECONE (BATCHING) ---
print("Encoding and Uploading to Pinecone...")
batch_size = 100

for i in tqdm(range(0, len(df), batch_size)):
    batch_df = df.iloc[i:i+batch_size]
    
    # 1. Generate IDs
    ids = [str(x) for x in batch_df['id'].tolist()]
    
    # 2. Generate Dense Vectors
    dense_vectors = dense_model.encode(batch_df['soup'].tolist()).tolist()
    
    # 3. Generate Sparse Vectors
    sparse_vectors = bm25.encode_documents(batch_df['soup'].tolist())
    
    # 4. Generate Metadata
    metadata = [
        {
            "title": row['title'],
            "overview": row['overview'],
            "poster_path": row['poster_path']
        }
        for _, row in batch_df.iterrows()
    ]
    
    # 5. Assemble and Upsert
    records = []
    for j in range(len(ids)):
        records.append({
            "id": ids[j],
            "values": dense_vectors[j],
            "sparse_values": sparse_vectors[j],
            "metadata": metadata[j]
        })
        
    index.upsert(vectors=records)

print("✅ Successfully uploaded massive dataset to Pinecone Cloud!")
