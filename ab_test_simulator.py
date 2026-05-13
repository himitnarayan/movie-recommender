import numpy as np
import pandas as pd
from scipy import stats
from recommender.engine import recommend
import random

# Ground truth test cases for A/B testing
TEST_CASES = {
    "The Dark Knight": ["Batman Begins", "The Dark Knight Rises", "Joker", "Batman"],
    "Iron Man": ["Iron Man 2", "Iron Man 3", "The Avengers", "Captain America: Civil War"],
    "Toy Story": ["Toy Story 2", "Toy Story 3", "Finding Nemo", "Monsters, Inc."],
    "The Matrix": ["The Matrix Reloaded", "The Matrix Revolutions", "Inception"]
}

def simulate_user_click(recommendations, relevant_movies):
    """
    Simulates a user click based on whether a relevant movie is in the recommendations.
    If a relevant movie is shown, they have an 80% chance of clicking it.
    If no relevant movie is shown, they have a 5% chance of randomly clicking something.
    """
    relevant_set = set(relevant_movies)
    titles = [r['title'] for r in recommendations]
    
    # Check if any recommended movie is in the relevant set
    has_relevant = any(title in relevant_set for title in titles)
    
    if has_relevant:
        return 1 if random.random() < 0.80 else 0
    else:
        return 1 if random.random() < 0.05 else 0

def run_ab_test_simulation(num_users=1000):
    print(f"Starting A/B Test Simulation with {num_users} simulated users...")
    print("Algorithm A (Control): TF-IDF Keyword Search")
    print("Algorithm B (Test): Hybrid Semantic + TF-IDF Search\n")
    
    clicks_a = []
    clicks_b = []
    
    queries = list(TEST_CASES.keys())
    
    for i in range(num_users):
        # Pick a random query for the user
        query = random.choice(queries)
        relevant = TEST_CASES[query]
        
        # User is randomly assigned to Group A (Control) or Group B (Test)
        if random.random() < 0.5:
            # Group A: TF-IDF
            results = recommend(query, top_n=5, algorithm="tfidf")
            if not isinstance(results, dict):
                click = simulate_user_click(results, relevant)
                clicks_a.append(click)
        else:
            # Group B: Hybrid
            results = recommend(query, top_n=5, algorithm="hybrid")
            if not isinstance(results, dict):
                click = simulate_user_click(results, relevant)
                clicks_b.append(click)
                
    # Calculate Results
    ctr_a = np.mean(clicks_a) * 100 if clicks_a else 0
    ctr_b = np.mean(clicks_b) * 100 if clicks_b else 0
    
    print("=== A/B TEST RESULTS ===")
    print(f"Control Group A (TF-IDF): {len(clicks_a)} users | CTR: {ctr_a:.2f}%")
    print(f"Test Group B (Hybrid):    {len(clicks_b)} users | CTR: {ctr_b:.2f}%\n")
    
    # Statistical Significance (Two-sample proportion z-test / t-test approximation)
    t_stat, p_value = stats.ttest_ind(clicks_a, clicks_b, equal_var=False)
    
    print("=== STATISTICAL SIGNIFICANCE ===")
    print(f"P-Value: {p_value:.5f}")
    if p_value < 0.05:
        print("✅ Result is STATISTICALLY SIGNIFICANT. Algorithm B outperforms Algorithm A.")
    else:
        print("❌ Result is NOT statistically significant. Cannot declare a winner.")

if __name__ == "__main__":
    run_ab_test_simulation()
