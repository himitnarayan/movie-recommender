# 🎯 Interview Preparation Guide: Hybrid Movie Recommender System

This guide will help you confidently present your project to a hiring manager, Principal Data Scientist, or ML Engineering lead.

---

## 1. The 2-Minute Elevator Pitch
*When the interviewer asks: "Walk me through a recent project you built."*

> "I built a production-grade Hybrid Movie Recommendation System capable of searching nearly 1 million movies in milliseconds. 
>
> I initially started with a standard TF-IDF keyword approach, but I quickly realized it failed to understand semantic meaning—for example, a user searching for 'space adventure' wouldn't find *Guardians of the Galaxy* if those exact words weren't in the synopsis. 
> 
> To solve this, I designed a **decoupled cloud architecture**. I wrote a Google Colab pipeline that uses a HuggingFace `SentenceTransformer` to generate dense neural embeddings for the entire Kaggle TMDB dataset, alongside a `BM25Encoder` for sparse keyword vectors. Because this resulted in a massive 1.4GB matrix that would crash a standard web server, I upserted the embeddings to a **Pinecone Serverless Vector Database**.
> 
> Finally, I built a lightweight Django backend that encodes user queries on-the-fly, queries Pinecone for a hybrid sparse-dense match, and then uses the **Google Gemini LLM** to generate a one-sentence personalized explanation of *why* the user will like the top result. The whole system is fully Dockerized for easy deployment."

---

## 2. Visual Architecture Explanation

If they ask how the system works, draw this or explain it step-by-step:

```mermaid
graph TD
    subgraph Offline ML Pipeline (Heavy Compute)
        A[Kaggle Dataset 930k] -->|Clean & Parse| B(Data Preprocessing)
        B --> C{Vectorization}
        C -->|Dense: SentenceTransformers| D[Semantic Embeddings 384-dim]
        C -->|Sparse: BM25Encoder| E[Keyword Embeddings]
        D --> F[(Pinecone Serverless Vector DB)]
        E --> F
    end

    subgraph Online Inference Pipeline (Lightweight)
        G[User Types Query] --> H(Django API Endpoint)
        H --> I(Encode Query: Sparse + Dense)
        I --> J{Pinecone Hybrid Search}
        J --> K[Top 10 Movie Results]
        K --> L(Google Gemini XAI Prompt)
        L --> M[UI Displays Movies + Explanation]
    end
```

**Key talking points for the architecture:**
1. **Decoupled System:** Emphasize that you *purposefully* separated the heavy ML training (Google Colab GPU) from the web server (Django) to keep hosting costs at $0 while achieving infinite scalability.
2. **Hybrid Search:** Explain that Pinecone uses an *Alpha Parameter* to balance between exact keyword hits (BM25 Sparse) and semantic meaning (Neural Dense). 
3. **Explainability (XAI):** Mention that recommender systems are often "Black Boxes," so you used Gemini to increase user trust by explaining the recommendations.

---

## 3. Technology Stack Breakdown

Be prepared to explain *why* you chose these specific tools:

| Technology | What it does | Why you chose it over alternatives |
| :--- | :--- | :--- |
| **SentenceTransformers** | Generates Dense Vectors | I used `all-MiniLM-L6-v2` because it is extremely fast and lightweight while retaining high accuracy, making it perfect for generating 1 million vectors quickly. |
| **BM25Encoder** | Generates Sparse Vectors | BM25 is the industry standard for keyword search (better than basic TF-IDF) because it handles term saturation and document length normalization better. |
| **Pinecone** | Cloud Vector Database | I chose Pinecone over local FAISS because local FAISS requires loading the entire index into RAM (which would crash a 512MB free tier server). Pinecone allows serverless API querying. |
| **Google Gemini API** | LLM Explainability | Extremely fast inference times compared to local open-source LLMs, ensuring the web page loads instantly. |
| **Django / Docker** | Web Framework & Ops | Django provides robust API routing, and Docker ensures the project runs identically on any cloud provider. |

---

## 4. Common Interview Questions & How to Answer Them

### Q1: "How did you handle the memory limitations of processing 1 million movies?"
**Your Answer:** "When I first built it, generating a FAISS index and TF-IDF matrix locally consumed over 1.5GB of RAM. If I deployed that, it would instantly crash standard free-tier cloud instances. I solved this by treating the project like an enterprise MLOps pipeline: I moved the heavy vector generation to a Google Colab GPU notebook, and then shipped only the final vectors to a Pinecone cloud database. Now my Django app just makes API calls and uses less than 100MB of RAM."

### Q2: "What is Hybrid Search and why is it better than just Semantic Search?"
**Your Answer:** "Semantic search (Dense vectors) is great at understanding meaning, but it sometimes fails at exact matching. For example, if a user searches for the exact title *'The Matrix Reloaded'*, a pure semantic search might return *'Inception'* because the 'vibes' are similar. By using a Hybrid approach, the Sparse Vector (BM25) guarantees that exact keyword matches are heavily weighted, while the Dense Vector handles the thematic matching. It gives the best of both worlds."

### Q3: "How do you evaluate if your new Hybrid algorithm is actually better than the old one?"
**Your Answer:** "I built an offline **A/B Testing Simulator**. I created ground-truth clusters of movies, and wrote a script that simulates 1,000 users. Half are routed to the TF-IDF algorithm, and half to the Hybrid algorithm. I calculated the Click-Through Rate (CTR) based on how often relevant movies appeared in the Top 5. Finally, I used a Two-Sample T-Test via SciPy to generate a P-Value, proving statistical significance."

### Q4: "What were some challenges you faced while dealing with the dataset?"
**Your Answer:** "The Kaggle TMDB dataset was massive and contained a lot of malformed rows. When parsing the CSV with Pandas, I encountered `EOF inside string` errors because of unclosed quotation marks inside movie overviews. To fix it, I had to force Pandas to use the `python` engine and skip bad lines. Additionally, because the genres were formatted as JSON strings, dropping quotes completely shattered the dataframe columns. I had to carefully sanitize the data, particularly coercing the `id` column to numeric to drop corrupted rows before pushing to Pinecone, as Pinecone strictly requires valid ASCII strings for IDs."

### Q5: "If we hired you, how would you improve this system for production?"
**Your Answer:** "Right now, it's a Content-Based Filtering system. In production, I would upgrade it to a full **Two-Tower architecture** or add **Collaborative Filtering**. I would track actual user clicks in a PostgreSQL database, run Matrix Factorization (like SVD) to find similarities between *users*, and use that data to personalize the home page before the user even types a search query."
