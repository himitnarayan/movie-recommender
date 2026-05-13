# 🎬 Movie Recommender System — Advanced Cloud Hybrid Search Web App

A production-style **Hybrid Movie Recommendation System** built for extreme scale, demonstrating modern Data Science, MLOps, and Cloud Vector Search practices.

### 🌟 Key Data Science & Cloud Features
- ☁️ **Cloud Vector Database (Pinecone)**: Overcomes memory limitations by hosting the 930k embeddings on a free Pinecone serverless cluster.
- 🧠 **Hybrid Search Architecture**: Combines **BM25 Sparse Vectors** (for keyword matching) with **Dense Neural Embeddings** (SentenceTransformers) natively inside Pinecone.
- 🚀 **Google Colab Training Pipeline**: The massive dataset is downloaded, vectorized, and uploaded to Pinecone completely for free using a Colab T4 GPU (`colab_training_script.py`).
- 🤖 **LLM Explainability (XAI)**: Integrates with the **Google Gemini API** to generate one-sentence personalized explanations for *why* a movie was recommended.
- 🐳 **Dockerized Deployment**: Fully containerized with a `Dockerfile` and `docker-compose.yml`.
- 🌐 **Django REST API**: Extremely lightweight web backend (runs easily on 512MB RAM free tiers since the heavy lifting is offloaded to Pinecone).

---

## 🚀 Live Demo

👉 (https://movie-recommender-6a6h.onrender.com/)

---

## 🧠 How It Works (Cloud Architecture)

1. **Training (Colab GPU):**
   - The Kaggle TMDB dataset (~930k movies) is loaded.
   - `SentenceTransformers` generates Dense Vectors.
   - `BM25Encoder` generates Sparse Vectors.
   - Vectors are upserted into Pinecone.
2. **Inference (Django on Render):**
   - User types a query (e.g., "The Matrix").
   - Django encodes the query into Dense/Sparse vectors.
   - Django queries the Pinecone API for top 10 matches.
   - The **Gemini LLM** generates a brief explanation for the top recommendation.

---

## 🏗️ Architecture
`Colab GPU (Generate Vectors) -> Pinecone DB (Store & Search) -> Django API (LLM Explainability) -> UI`
