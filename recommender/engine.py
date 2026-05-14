import os
import requests
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Global placeholders
pc_index = None
bm25 = None

def load_cloud_models():
    global pc_index, bm25

    if pc_index is None:
        print("Connecting to Pinecone Cloud Vector DB...")
        
        # 1. Connect to Pinecone
        pinecone_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_key:
            print("WARNING: PINECONE_API_KEY not found in environment.")
            return
            
        pc = Pinecone(api_key=pinecone_key)
        index_name = "movie-hybrid-search"
        
        if index_name in [idx.name for idx in pc.list_indexes()]:
            pc_index = pc.Index(index_name)
        else:
            print(f"Index {index_name} does not exist in Pinecone.")
            return

        # 2. Load BM25 Sparse Encoder params (Lazy Load to save RAM)
        try:
            from pinecone_text.sparse import BM25Encoder
            bm25 = BM25Encoder()
            bm25.load("recommender/artifacts/bm25_params.json")
        except Exception as e:
            print(f"Warning: BM25 Sparse search will be skipped. {e}")

        # 3. Configure Gemini
        gemini_key = os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)

def get_dense_vector(text):
    # Using HuggingFace Free Inference API instead of local PyTorch to save 500MB of RAM!
    api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {}
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        
    try:
        response = requests.post(api_url, headers=headers, json={"inputs": [text], "options":{"wait_for_model":True}})
        if response.status_code == 200:
            return response.json()[0]
        else:
            print(f"HF API Error: {response.text}")
            return None
    except Exception as e:
        print(f"HF Request failed: {e}")
        return None

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

    # 1. Encode user query into dense vector via API
    dense_vector = get_dense_vector(movie_title)
    if not dense_vector:
        return {"error": "Failed to generate semantic embedding via API."}
    
    # 2. Encode user query into sparse vector
    sparse_vector = None
    if bm25 is not None:
        try:
            sparse_vector = bm25.encode_queries(movie_title)
        except Exception as e:
            print(f"BM25 Error: {e}")

    # 3. Query Pinecone (Cloud Hybrid Search)
    query_params = {
        "vector": dense_vector,
        "top_k": top_n,
        "include_metadata": True
    }
    
    if algorithm == "hybrid" and sparse_vector:
        query_params["sparse_vector"] = sparse_vector
        
    try:
        response = pc_index.query(**query_params)
    except Exception as e:
        return {"error": f"Pinecone search failed: {e}"}

    results = []
    for match in response.matches:
        metadata = match.metadata
        poster_url = ""
        if isinstance(metadata.get('poster_path'), str) and metadata['poster_path']:
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
