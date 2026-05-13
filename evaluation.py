import numpy as np
import pandas as pd
from recommender.engine import recommend

# Ground truth test cases for offline evaluation
# Format: "Query Movie": ["Expected Relevant Movie 1", "Expected Relevant Movie 2"]
TEST_CASES = {
    "The Dark Knight": ["Batman Begins", "The Dark Knight Rises", "Joker", "Batman"],
    "Iron Man": ["Iron Man 2", "Iron Man 3", "The Avengers", "Captain America: Civil War"],
    "Toy Story": ["Toy Story 2", "Toy Story 3", "Finding Nemo", "Monsters, Inc."],
    "The Matrix": ["The Matrix Reloaded", "The Matrix Revolutions", "Inception"]
}

def calculate_precision_at_k(recommended, relevant, k=10):
    recommended_k = [r['title'] for r in recommended[:k]]
    relevant_set = set(relevant)
    hits = sum(1 for rec in recommended_k if rec in relevant_set)
    return hits / float(k)

def calculate_dcg_at_k(recommended, relevant, k=10):
    recommended_k = [r['title'] for r in recommended[:k]]
    relevant_set = set(relevant)
    dcg = 0.0
    for i, rec in enumerate(recommended_k):
        if rec in relevant_set:
            dcg += 1.0 / np.log2(i + 2) # i is 0-indexed, so rank is i+1, log2(rank+1) -> log2(i+2)
    return dcg

def calculate_ndcg_at_k(recommended, relevant, k=10):
    dcg = calculate_dcg_at_k(recommended, relevant, k)
    
    # Ideal DCG (all relevant items at the top)
    idcg = 0.0
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def run_evaluation():
    print("Starting Offline Evaluation...")
    precisions = []
    ndcgs = []
    
    for query, relevant in TEST_CASES.items():
        print(f"\nQuerying: {query}")
        results = recommend(query, top_n=10)
        
        if isinstance(results, dict) and "error" in results:
            print(f"Skipping (not in db): {results['error']}")
            continue
            
        p_at_10 = calculate_precision_at_k(results, relevant, k=10)
        ndcg_at_10 = calculate_ndcg_at_k(results, relevant, k=10)
        
        print(f"Precision@10: {p_at_10:.4f} | NDCG@10: {ndcg_at_10:.4f}")
        
        precisions.append(p_at_10)
        ndcgs.append(ndcg_at_10)
        
    if precisions:
        mean_precision = np.mean(precisions)
        mean_ndcg = np.mean(ndcgs)
        print("\n==================================")
        print("OVERALL METRICS")
        print(f"Mean Precision@10: {mean_precision:.4f}")
        print(f"Mean NDCG@10:      {mean_ndcg:.4f}")
        print("==================================")
    else:
        print("No metrics computed. Check if test case movies exist in DB.")

if __name__ == "__main__":
    run_evaluation()
