import os
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Global placeholders — loaded once on first request to save startup time
pc_index = None
pc_client = None
bm25 = None

def load_cloud_models():
    global pc_index, pc_client, bm25

    if pc_index is not None:
        return  # Already loaded

    print("Connecting to Pinecone Cloud Vector DB...")

    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        print("WARNING: PINECONE_API_KEY not found.")
        return

    pc_client = Pinecone(api_key=pinecone_key)
    index_name = "movie-hybrid-search"

    existing = [idx.name for idx in pc_client.list_indexes()]
    if index_name in existing:
        pc_index = pc_client.Index(index_name)
    else:
        print(f"Index '{index_name}' not found in Pinecone.")
        return

    # Load BM25 Sparse Encoder (optional — skip gracefully if file missing)
    try:
        from pinecone_text.sparse import BM25Encoder
        bm25 = BM25Encoder()
        bm25.load("recommender/artifacts/bm25_params.json")
        print("BM25 sparse encoder loaded.")
    except Exception as e:
        print(f"Warning: BM25 Sparse search will be skipped. {e}")

    # Configure Gemini for XAI
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        genai.configure(api_key=gemini_key)


def get_dense_vector(text):
    """
    Uses HuggingFace Inference API to generate 384-dim embeddings.
    Requires a free HF_TOKEN set in your environment variables.
    Get one free at: https://huggingface.co/settings/tokens
    """
    import requests
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        return None

    api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": [text], "options": {"wait_for_model": True}}
        )
        if response.status_code == 200:
            result = response.json()
            # API returns a nested list: [[vector]] — we want the inner list
            if isinstance(result, list) and isinstance(result[0], list):
                return result[0]
            return result
        else:
            print(f"HF API Error ({response.status_code}): {response.text[:200]}")
            return None
    except Exception as e:
        print(f"HF Request failed: {e}")
        return None


def generate_explanation(query_title, recommended_title, recommended_overview):
    try:
        if not os.environ.get("GOOGLE_API_KEY"):
            return "Similarity match based on themes and content."

        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = (
            f"In one short sentence, explain why someone who likes '{query_title}' "
            f"would also like '{recommended_title}'. "
            f"Overview: {recommended_overview[:300]}. Be brief and engaging."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Similar themes, genres, or mood."


def recommend(movie_title, top_n=10, algorithm="hybrid"):
    load_cloud_models()

    if pc_index is None:
        return {"error": "Pinecone database is not connected. Please add PINECONE_API_KEY."}

    # 1. Generate Dense Vector via Pinecone Inference API
    dense_vector = get_dense_vector(movie_title)
    if not dense_vector:
        return {"error": "Failed to generate semantic embedding. Check PINECONE_API_KEY."}

    # 2. Generate Sparse Vector (BM25) if available
    sparse_vector = None
    if bm25 is not None:
        try:
            sparse_vector = bm25.encode_queries(movie_title)
        except Exception as e:
            print(f"BM25 Error: {e}")

    # 3. Query Pinecone
    query_params = {
        "vector": dense_vector,
        "top_k": top_n,
        "include_metadata": True
    }

    if algorithm == "hybrid" and sparse_vector:
        if sparse_vector.get("indices"):
            query_params["sparse_vector"] = sparse_vector

    try:
        response = pc_index.query(**query_params)
    except Exception as e:
        return {"error": f"Pinecone query failed: {e}"}

    results = []
    for match in response.matches:
        metadata = match.metadata
        poster_url = ""
        if isinstance(metadata.get("poster_path"), str) and metadata["poster_path"].strip():
            poster_url = f"https://image.tmdb.org/t/p/w500{metadata['poster_path']}"

        explanation = "High similarity match based on content and semantics."
        if len(results) == 0:
            explanation = generate_explanation(
                movie_title,
                metadata.get("title", ""),
                metadata.get("overview", "")
            )

        results.append({
            "title": metadata.get("title", "Unknown"),
            "poster": poster_url,
            "link": f"https://www.themoviedb.org/movie/{match.id}",
            "explanation": explanation
        })

    return results
