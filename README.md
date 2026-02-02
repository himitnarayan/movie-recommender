# 🎬 Movie Recommender System — Scalable Content-Based ML Web App

A production-style **Content-Based Movie Recommendation System** built with:

- 🧠 TF-IDF Vectorization on movie metadata
- 📐 Cosine Similarity via Nearest Neighbor vector search
- 🌐 Django REST API
- 🎨 Minimal Web UI
- ☁️ Deployed on Render (Free Tier)
- 📦 Kaggle dataset auto-download during build

> ⚡ This project follows **real ML engineering practices** — model artifacts are created during deployment, not stored in GitHub.

---

## 🚀 Live Demo

👉 (https://movie-recommender-6a6h.onrender.com/)

---

## 🧠 How It Works

1. Kaggle TMDB dataset (~930k movies) is downloaded during deploy
2. Top **60,000 popular movies** are selected (performance optimization)
3. Metadata "soup" created using:
   - Overview
   - Genres
   - Keywords
4. TF-IDF vectorization (30,000 features)
5. Nearest Neighbors trained with cosine similarity
6. Django API serves recommendations with posters and TMDB links

---

## 🏗️ Architecture

