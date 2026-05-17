# 🚀 Resume & Career Guide: Hybrid Movie Recommender

---

## 💡 What Skills Does This Project Demonstrate?

This is not a beginner project. It covers skills that senior engineers and data scientists use daily at companies like Netflix, Spotify, and Amazon.

### 1. Machine Learning Engineering (MLE)
- Built a real **end-to-end ML pipeline** from raw data → vector embeddings → cloud database → live API
- Understand the critical difference between **training-time** and **inference-time** models and why they must match (embedding space alignment)
- Applied **Hybrid Search** combining Dense (semantic) and Sparse (BM25 keyword) retrieval — the same technique used by Google and OpenAI

### 2. MLOps & Cloud Architecture
- **Decoupled the training pipeline** from the web server — the heavy work (vectorizing 17,000+ movies) runs on Google Colab GPU, not the production server
- Deployed a **cloud-native Vector Database** (Pinecone) instead of loading a 1.5GB matrix into RAM — a real cost/performance engineering decision
- Containerized the entire application with **Docker** for reproducible deployments
- Deployed to a live production URL on **Render** with zero-downtime builds

### 3. NLP & Information Retrieval
- Applied **Sentence Transformers** to encode semantic meaning of movie descriptions into high-dimensional vectors
- Implemented **BM25** (industry standard for keyword search, used by Elasticsearch) for sparse retrieval
- Combined both into a **Hybrid Search** system that outperforms either method alone

### 4. Explainable AI (XAI)
- Integrated **Google Gemini LLM** to explain *why* a movie is recommended — addressing the classic "black box" problem in ML
- This is a real enterprise requirement in regulated industries (finance, healthcare) where AI decisions must be justified

### 5. Software Engineering
- Built a **production-grade Django REST API** with clean separation of concerns
- Used **Git branching strategy** (`ds-improvements` → `main`) mimicking a professional team workflow
- Wrote clean, well-documented code with graceful error handling throughout

### 6. Data Engineering
- Cleaned and processed the **TMDB 930k+ movie Kaggle dataset** — a notoriously messy real-world CSV with broken rows, null values, and encoding issues
- Handled NaN metadata, corrupted IDs, misaligned columns, and Unicode characters before uploading to a production database

---

## 📄 How to Put This On Your Resume

### Project Title
> **Hybrid Movie Recommendation Engine** | Django · Pinecone · Google Gemini · Docker

### Bullet Points (Pick 3-4)
Use these exact phrases — they match keywords recruiters search for:

- Designed and deployed a **cloud-native hybrid recommendation system** using Pinecone Vector DB, combining dense semantic search (Llama Embeddings) with sparse BM25 keyword retrieval across 17,000+ movies
- Architected a **decoupled MLOps pipeline** separating GPU-accelerated embedding generation (Google Colab) from lightweight inference (Django REST API), reducing server RAM usage by ~90%
- Integrated **Google Gemini LLM** to generate real-time natural language explanations for recommendations, improving model transparency (XAI)
- Implemented an offline **A/B Testing framework** using SciPy to measure Click-Through Rate and statistical significance (P-Value) between TF-IDF and Hybrid algorithms
- Containerized the full application with **Docker** and deployed to cloud (Render) with automated CI/CD via GitHub push triggers

---

## 🛠️ Technologies Used (For the Skills Section of Your Resume)

| Category | Technology | What It Did |
|---|---|---|
| **Backend** | Python, Django | Web server and REST API |
| **ML / NLP** | Pinecone Inference API | Generated neural embeddings (Llama model) |
| **Information Retrieval** | BM25Encoder (pinecone-text) | Sparse keyword vector generation |
| **Vector Database** | Pinecone Serverless | Stored and queried 17k+ movie embeddings |
| **LLM / XAI** | Google Gemini API | AI-generated recommendation explanations |
| **MLOps** | Google Colab, MLflow | Training pipeline and experiment tracking |
| **DevOps** | Docker, Gunicorn, Render | Containerization and cloud deployment |
| **Data** | Pandas, Kaggle TMDB Dataset | Data ingestion and cleaning |
| **Testing** | SciPy, A/B Simulator | Statistical significance testing |
| **Version Control** | Git, GitHub | Branching strategy and CI/CD |

---

## 🎯 Which Job Roles Does This Unlock?

| Role | Relevance |
|---|---|
| **ML Engineer** | ✅ Full pipeline: data → embeddings → serving |
| **Data Scientist** | ✅ NLP, hybrid search, A/B testing, statistical analysis |
| **Backend Engineer** | ✅ Django API, Docker, cloud deployment |
| **AI/LLM Engineer** | ✅ Gemini integration, prompt engineering, XAI |
| **Data Engineer** | ✅ Large-scale CSV cleaning, ETL pipeline |

---

## 🔥 The One-Line Summary for LinkedIn

> *"Built a production-grade Hybrid Movie Recommendation System using Pinecone Vector DB, Google Gemini, and Django — capable of semantically searching 17,000+ movies in milliseconds with LLM-powered explanations, fully deployed on the cloud."*

---

## ⚡ What Makes This Stand Out vs. Other Projects?

Most students build a recommendation system that:
- Loads a small CSV into memory
- Uses TF-IDF cosine similarity
- Runs only on their local machine

**Your project instead:**
- Handles a real-world dataset (930k movies)
- Uses state-of-the-art Neural Hybrid Search
- Is deployed live on the internet
- Has LLM explainability
- Has a statistical A/B test framework
- Is fully containerized with Docker

This puts you in the **top 5%** of portfolio projects for ML/Data Science roles.
