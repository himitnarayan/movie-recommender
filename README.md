# 🎬 AniBaba Hybrid Recommender System

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Django](https://img.shields.io/badge/django-REST-green.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-lightgrey.svg)
![Gemini](https://img.shields.io/badge/Gemini-LLM_XAI-orange.svg)
![MLOps](https://img.shields.io/badge/MLflow-Tracking-blue.svg)

A production-grade, highly scalable **Hybrid Movie Recommendation Engine** designed to demonstrate modern Machine Learning Engineering, NLP, MLOps, and Cloud Architecture practices. 

Instead of relying on simple keyword matching, this engine uses a decoupled architecture to perform blazing-fast semantic searches across nearly **1 million movies**, providing LLM-powered explainability for its recommendations.

---

## 🌟 Key Features

### 1. Hybrid Search Architecture (Sparse + Dense)
Combines the precision of exact keyword matching with the deep contextual understanding of Neural Networks.
* **Dense Vectors:** Uses HuggingFace `SentenceTransformers` (`all-MiniLM-L6-v2`) to encode movie overviews and genres into 384-dimensional semantic embeddings. Captures the "meaning" of a search (e.g., matching "space adventure" to *Guardians of the Galaxy*).
* **Sparse Vectors:** Uses `BM25Encoder` to generate sparse arrays for traditional TF-IDF keyword matching.
* **Alpha Weighting:** Dynamically balances semantic meaning vs. exact keyword hits during retrieval.

### 2. Cloud Vector Database (Pinecone)
Overcomes the memory limitations of free-tier cloud hosting (like Render's 512MB RAM). 
* **Training Pipeline:** A Google Colab T4 GPU script (`colab_training_script.py`) is used to vectorize the 930,000+ TMDB movie dataset and upsert the 1.4GB matrix directly to a Pinecone Serverless cluster.
* **Inference Pipeline:** The Django backend is ultra-lightweight. It encodes the user's search query on the fly and hits the Pinecone API, returning results in milliseconds without storing any heavy indices in memory.

### 3. Explainable AI (XAI) via Google Gemini
Machine Learning models are often "black boxes." To build user trust, this system integrates the **Google Gemini 2.5 Flash API**.
* When a top recommendation is surfaced, the LLM analyzes the query and the recommended movie's metadata to generate a dynamic, one-sentence explanation of *why* the user will like it (e.g., *"You will like this because it shares the dark, gritty detective themes of The Batman."*)

### 4. A/B Testing Simulator
Includes an offline evaluation script (`ab_test_simulator.py`) to mathematically prove the superiority of the Hybrid model.
* Simulates 1,000 users interacting with Algorithm A (TF-IDF Control) and Algorithm B (Hybrid Test).
* Calculates Click-Through Rate (CTR) based on relevant cluster matching.
* Uses a Two-Sample T-Test to generate a P-Value, proving Statistical Significance for business stakeholders.

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Offline Training Pipeline (Google Colab GPU)
        A[Kaggle TMDB Dataset 930k] --> B(Data Cleaning & Preprocessing)
        B --> C{Vectorization}
        C -->|SentenceTransformers| D[Dense Embeddings]
        C -->|BM25Encoder| E[Sparse Embeddings]
        D --> F[(Pinecone Cloud Vector DB)]
        E --> F
    end

    subgraph Online Inference Pipeline (Django Web App)
        G[User Search Query] --> H(Query Encoder)
        H --> I(Pinecone API Search)
        F -.-> I
        I --> J[Top 10 Movie Matches]
        J --> K(Google Gemini API)
        K -->|Generates Explainability| L[Final UI Payload]
    end
```

---

## 💻 Tech Stack

* **Backend:** Python, Django
* **Machine Learning:** HuggingFace `SentenceTransformers`, Scikit-Learn, Pandas, NumPy
* **Vector Database:** Pinecone (Serverless)
* **LLM & XAI:** Google Generative AI (Gemini)
* **MLOps:** MLflow (Experiment Tracking), Docker (Containerization)
* **Deployment:** Render (Web Service), Google Colab (Training Compute)

---

## 🚀 How to Run Locally

### 1. Prerequisites
You will need API keys for the following free services:
* [Pinecone](https://www.pinecone.io/) (Vector Database)
* [Google AI Studio](https://aistudio.google.com/) (Gemini API)

Create a `.env` file in the root directory:
```env
PINECONE_API_KEY=your_pinecone_key_here
GOOGLE_API_KEY=your_gemini_key_here
```

### 2. Standard Setup
```bash
# Clone the repository
git clone https://github.com/himitnarayan/movie-recommender.git
cd movie-recommender

# Create a virtual environment & install dependencies
python -m venv venv
source venv/Scripts/activate # (On Windows: .\venv\Scripts\Activate.ps1)
pip install -r requirements.txt

# Start the Django server
python manage.py runserver
```

### 3. Docker Setup
Alternatively, you can run the application instantly using Docker Compose:
```bash
docker-compose up --build
```
Navigate to `http://localhost:8000` to interact with the engine.

---

## 📈 Future Improvements
* **Collaborative Filtering:** Implement a Matrix Factorization (SVD) layer based on user rating history to personalize results beyond content-based filtering.
* **Celery Workers:** Move the Gemini XAI API calls to asynchronous background tasks to improve initial UI render speeds.
* **Real-time A/B Testing:** Build a middleware to randomly assign active web sessions to different ranking algorithms and store actual interaction data in PostgreSQL.

---
*Designed & Engineered for scale by Himit Narayan.*
