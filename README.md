#🎬 Movie Recommender System (1M TMDB Dataset)

A scalable Content-Based Movie Recommendation System built using:

🧠 TF-IDF Vectorization

📐 Cosine Similarity + Nearest Neighbors

🌐 Django REST API

🎨 Minimal Web UI

☁️ Deployed on Render (free tier)

📦 Kaggle dataset auto-download during build


##🚀 Live Demo

👉 (https://movie-recommender-6a6h.onrender.com/)



##🧠 How it works
1. Kaggle TMDB dataset (~930k movies) is downloaded during deploy

2. Top 60,000 popular movies are selected (performance optimization)

3. Metadata soup created using:

    Overview

    Genres

    Keywords

4. TF-IDF vectorization (30,000 features)

5. Nearest Neighbors trained with cosine similarity

6. Django API serves recommendations with movie posters and TMDB links


##Project Structure
build_model.py        → ML pipeline (runs during deploy)

build.sh              → Render build script

recommender/
    engine.py         → Recommendation engine
    templates/
        index.html    → UI
moviesite/

requirements.txt

##⚙️ Run Locally (Step by Step)
1️⃣ Clone the repo
