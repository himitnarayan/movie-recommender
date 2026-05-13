# 🎬 Movie Recommender System — Advanced Hybrid ML Web App

A production-style **Hybrid Movie Recommendation System** built for scale and accuracy, demonstrating modern Data Science and MLOps practices.

### 🌟 Key Data Science Features
- 🧠 **Hybrid Search Architecture**: Combines **TF-IDF Vectorization** (for keyword matching) with **Sentence Transformers** (Neural Embeddings for semantic meaning).
- ⚡ **Vector Database Integration**: Uses **FAISS (Facebook AI Similarity Search)** for high-performance dense vector retrieval.
- 📊 **MLOps & Tracking**: Implements **MLflow** for tracking model parameters, dataset sizes, and artifacts.
- 📈 **Offline Evaluation**: Includes a custom evaluation pipeline (`evaluation.py`) that calculates **NDCG@10** and **Precision@10** to rigorously test recommendation quality against ground-truth clusters.
- 🤖 **LLM Explainability (XAI)**: Integrates with the **Google Gemini API** to generate one-sentence personalized explanations for *why* a movie was recommended.
- 🌐 **Django REST API**: Serves the ML models via a robust web backend.
- ☁️ **Deployed on Render**: Automatically downloads the Kaggle TMDB dataset during build and processes embeddings on the fly.

> ⚡ This project follows **real ML engineering practices** — models are evaluated offline, tracked with MLflow, and artifacts are created during deployment, not stored in GitHub.

---

## 🚀 Live Demo

👉 (https://movie-recommender-6a6h.onrender.com/)

---

## 🧠 How It Works

1. Kaggle TMDB dataset (~930k movies) is downloaded during deploy.
2. Top **10,000 popular movies** are selected (for memory-safe dense embedding generation).
3. Metadata "soup" is created (Overview, Genres, Keywords, Director).
4. **TF-IDF Vectorization** (30,000 features) is trained and saved.
5. **Neural Embeddings** (`all-MiniLM-L6-v2`) generate dense vectors for semantic understanding.
6. A **FAISS Index** is built and saved for sub-millisecond similarity search.
7. During inference, the Django API performs a **Hybrid Search** (combining TF-IDF and FAISS distances).
8. The **Gemini LLM** generates a brief explanation for the top recommendation.

---

## 🏗️ Architecture
`Data Ingestion -> Vectorization (TF-IDF + Transformers) -> MLflow Tracking -> FAISS / NearestNeighbors -> Django API (with LLM Explainability) -> UI`
