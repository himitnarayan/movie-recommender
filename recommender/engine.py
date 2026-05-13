import os
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Global placeholders
pc_index = None
embedder = None
bm25 = None

def load_cloud_models():
    global pc_index, embedder, bm25

    if pc_index is None:
        print("Connecting to Pinecone Cloud Vector DB...")
        
        # 1. Connect to Pinecone
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            print("WARNING: PINECONE_API_KEY not found in environment.")
            return
            
        pc = Pinecone(api_key=pinecone_key)
        index_name = "movie-hybrid-search"
        
        # Ensure the index exists, then connect
        if index_name in [idx.name for idx in pc.list_indexes()]:
            pc_index = pc.Index(index_name)
        else:
            print(f"Index {index_name} does not exist in Pinecone.")
            return

        # 2. Load Neural Embedding Model (Small, runs fast in 512MB RAM)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 3. Load BM25 Sparse Encoder params (Created in Colab)
        bm25 = BM25Encoder()
        # In a real scenario, you download bm25_params.json from Colab and place it here
        # bm25.load("recommender/artifacts/bm25_params.json")
        # For this demo, we'll initialize a dummy one if file doesn't exist
        try:
            bm25.load("recommender/artifacts/bm25_params.json")
        except:
            print("Warning: bm25_params.json not found. Sparse search will be skipped.")

        # 4. Configure Gemini
        gemini_key = os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)

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
    load_cloud_models()

    if pc_index is None:
        return {"error": "Pinecone database is not connected. Add PINECONE_API_KEY."}

    # 1. Encode user query into dense vector
    dense_vector = embedder.encode(movie_title).tolist()
    
    # 2. Encode user query into sparse vector
    sparse_vector = None
    try:
        sparse_vector = bm25.encode_queries(movie_title)
    except:
        pass # If BM25 isn't loaded, skip it

    # 3. Query Pinecone (Cloud Hybrid Search)
    # Pinecone natively combines dense and sparse vectors via alpha weighting
    # Alpha = 0.5 means equal weight
    
    query_params = {
        "vector": dense_vector,
        "top_k": top_n,
        "include_metadata": True
    }
    
    if algorithm == "hybrid" and sparse_vector:
        query_params["sparse_vector"] = sparse_vector
        # In Pinecone, if both are provided, it does a dot-product hybrid search automatically
        
    try:
        response = pc_index.query(**query_params)
    except Exception as e:
        return {"error": f"Pinecone search failed: {e}"}

    results = []
    
    # 4. Process Cloud Results
    for match in response.matches:
        metadata = match.metadata
        
        poster_url = ""
        if isinstance(metadata.get('poster_path'), str):
            poster_url = f"https://image.tmdb.org/t/p/w500{metadata['poster_path']}"

        explanation = "High similarity match based on content and semantics."
        if algorithm == "hybrid" and len(results) == 0:
             explanation = generate_explanation(movie_title, metadata['title'], metadata.get('overview', ''))

        results.append({
            "title": metadata['title'],
            "poster": poster_url,
            "link": f"https://www.themoviedb.org/movie/{match.id}",
            "explanation": explanation
        })

    return results
