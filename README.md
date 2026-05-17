<div align="center">

# 🎬 AniBaba Movie Recommender

### A Production-Grade Hybrid Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Django](https://img.shields.io/badge/Django-REST_API-092E20?style=for-the-badge&logo=django&logoColor=white)](https://djangoproject.com)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-000000?style=for-the-badge)](https://pinecone.io)
[![Gemini](https://img.shields.io/badge/Google-Gemini_AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Render](https://img.shields.io/badge/Render-Live_Deploy-46E3B7?style=for-the-badge)](https://render.com)

*Semantically search 17,000+ movies using Neural Hybrid Search — deployed live on the cloud.*

</div>

---

## 📖 Overview

AniBaba is not a typical beginner recommender. It is a **cloud-native, production-grade ML system** that uses the same architecture patterns as Netflix and Spotify.

A user types a natural language query like *"a dark psychological thriller set in space"* and the engine uses neural embeddings to semantically understand the intent — not just match keywords — and returns the most relevant movies from a cloud vector database, with a **Google Gemini AI-generated explanation** of why each result matches.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│           OFFLINE TRAINING PIPELINE                  │
│           (Google Colab — runs once)                 │
│                                                     │
│  Kaggle TMDB 930k Dataset                           │
│       │                                             │
│       ▼                                             │
│  Data Cleaning & Preprocessing (Pandas)             │
│       │                                             │
│       ├──► Dense Vectors (HuggingFace Embeddings)   │
│       │                                             │
│       └──► Sparse Vectors (BM25Encoder)             │
│               │                                     │
│               ▼                                     │
│         Pinecone Cloud Vector DB ◄──────────────────┘
│               │
└───────────────┼─────────────────────────────────────
                │
┌───────────────┼─────────────────────────────────────┐
│           ONLINE INFERENCE PIPELINE                  │
│           (Django Web App — live 24/7)               │
│                                                     │
│  User Query ──► HuggingFace API (encode query)      │
│                      │                              │
│                      ▼                              │
│               Pinecone Hybrid Search                │
│                      │                              │
│                      ▼                              │
│               Top 10 Movie Matches                  │
│                      │                              │
│                      ▼                              │
│           Google Gemini XAI Explanation             │
│                      │                              │
│                      ▼                              │
│                Final UI Response                    │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔍 Hybrid Search (Dense + Sparse)
Combines two retrieval techniques for superior results:

- **Dense (Semantic) Search** — HuggingFace `all-MiniLM-L6-v2` model encodes the meaning of queries into 384-dimensional vectors. Finds movies that *feel* similar even without exact keyword matches.
- **Sparse (Keyword) Search** — `BM25Encoder` from `pinecone-text` provides exact keyword matching and term-frequency scoring, the same algorithm used by Elasticsearch.
- **Hybrid Fusion** — Both scores are combined via Pinecone's native hybrid query, giving the best of both worlds.

### ☁️ Cloud Vector Database (Pinecone)
- 17,000+ movie vectors stored in Pinecone Serverless (AWS `us-east-1`)
- The Django web server **never loads a model into memory** — it only makes API calls
- Entire application runs comfortably within Render's 512MB free tier

### 🤖 Explainable AI (Gemini)
- The top recommendation is passed to **Google Gemini 2.5 Flash**
- Gemini generates a human-readable, one-sentence explanation of *why* the movie matches the query
- Addresses the classic "black box" problem in recommender systems

### 🧪 A/B Testing Framework
- `ab_test_simulator.py` simulates 1,000 users split between TF-IDF (Control) and Hybrid (Test) algorithms
- Calculates Click-Through Rate (CTR) and runs a **Two-Sample T-Test** (SciPy) to determine statistical significance
- Produces a P-Value to validate whether the Hybrid model is genuinely better

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Web Framework** | Django + DRF | REST API and template rendering |
| **Production Server** | Gunicorn | Multi-worker WSGI server |
| **Vector Database** | Pinecone Serverless | Stores and searches 17k+ movie embeddings |
| **Sparse Search** | BM25Encoder (pinecone-text) | Keyword-based sparse vector generation |
| **Dense Embeddings** | HuggingFace Inference API | `all-MiniLM-L6-v2` — 384-dim semantic vectors |
| **LLM / XAI** | Google Gemini 2.5 Flash | Natural language recommendation explanations |
| **Training Compute** | Google Colab (T4 GPU) | Offline embedding generation pipeline |
| **Containerization** | Docker + docker-compose | Reproducible local and cloud environments |
| **Deployment** | Render | Auto-deploy from GitHub `main` branch |
| **Training Data** | Kaggle TMDB 930k Dataset | Source of all movie metadata |

---

## 🚀 Local Setup

### Prerequisites
Get free API keys from:
- [Pinecone](https://www.pinecone.io/) — Vector Database
- [Google AI Studio](https://aistudio.google.com/) — Gemini API
- [HuggingFace](https://huggingface.co/settings/tokens) — Embedding API

Create a `.env` file in the project root:
```env
PINECONE_API_KEY=your_pinecone_key_here
GOOGLE_API_KEY=your_gemini_key_here
HF_TOKEN=your_huggingface_token_here
```

### Option A: Standard Python
```bash
git clone https://github.com/himitnarayan/movie-recommender.git
cd movie-recommender

python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python manage.py runserver
```
Visit `http://localhost:8000`

### Option B: Docker
```bash
git clone https://github.com/himitnarayan/movie-recommender.git
cd movie-recommender
docker-compose up --build
```
Visit `http://localhost:8000`

---

## 📊 Training Your Own Index (Google Colab)

The `colab_training_script.py` file contains the full offline training pipeline:

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook
2. Upload your `kaggle.json` credentials file to the Colab file browser
3. Paste the contents of `colab_training_script.py` into a cell
4. Add your `PINECONE_API_KEY` at the top
5. Run it! The script will:
   - Download the 930k TMDB movie dataset from Kaggle
   - Clean and preprocess the data
   - Generate dense embeddings via Pinecone Inference API
   - Fit a BM25 sparse encoder
   - Upload everything to your Pinecone index
6. Download the generated `bm25_params.json` and place it at `recommender/artifacts/bm25_params.json`

---

## 📁 Project Structure

```
movie-recommender/
│
├── recommender/                # Django app
│   ├── engine.py               # Core ML logic — Hybrid Search + Gemini XAI
│   ├── views.py                # API and page views
│   ├── urls.py                 # URL routing
│   ├── templates/              # Frontend HTML
│   └── artifacts/
│       └── bm25_params.json    # Trained BM25 sparse encoder
│
├── moviesite/                  # Django project config
│   ├── settings.py
│   └── wsgi.py
│
├── colab_training_script.py    # Offline ML training pipeline
├── ab_test_simulator.py        # A/B testing framework
├── requirements.txt            # Production dependencies
├── build.sh                    # Render build script
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Local container orchestration
└── render.yaml                 # Render deployment config
```

---

## 🌐 Deployment on Render

This project auto-deploys on every push to `main`.

**Environment Variables required on Render:**

| Variable | Description |
|---|---|
| `PINECONE_API_KEY` | Your Pinecone API key |
| `GOOGLE_API_KEY` | Your Google Gemini API key |
| `HF_TOKEN` | Your HuggingFace token for embeddings |

**Build Command:** `./build.sh`

**Start Command:** `gunicorn moviesite.wsgi:application --bind 0.0.0.0:$PORT`

---

## 🔮 Future Improvements

- **Collaborative Filtering** — Track user clicks in PostgreSQL and add Matrix Factorization (SVD) to personalize recommendations beyond content
- **Two-Tower Model** — Train separate encoders for users and items for deeper personalization
- **Async Gemini Calls** — Move XAI explanations to Celery background tasks to improve page load speed
- **Real-time A/B Testing** — Serve different algorithms to live users via middleware and collect actual interaction data

---

<div align="center">

*Built with ❤️ by Himit Narayan*

</div>
