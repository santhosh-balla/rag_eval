"""
Kappa Score Calculation Module - Standalone Version
====================================================

This module provides inter-rater reliability functions (Cohen's Kappa, Fleiss' Kappa)
without requiring the full RAG pipeline dependencies.

Functions:
  - create_manual_scoring_template()
  - calculate_kappa_scores(team_ratings)
  - save_ratings_to_csv(team_ratings, filename)
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

try:
    from statsmodels.stats.inter_rater import fleiss_kappa
except ImportError:
    fleiss_kappa = None


def create_manual_scoring_template() -> Dict:
    """
    Create a template for manual scoring by multiple raters (1-5 scale).
    
    This allows a team of 5 people to independently score each answer
    and compute Cohen's kappa for inter-rater agreement.
    
    Scale:
    1 = Poor (completely incorrect, irrelevant)
    2 = Fair (mostly incorrect, some relevance)
    3 = Good (mostly correct, some issues)
    4 = Very Good (correct, minor issues)
    5 = Excellent (completely correct, clear explanation)
    
    Returns: Dictionary structure for team ratings
    """
    print("\n" + "="*80)
    print("MANUAL SCORING INSTRUCTIONS FOR TEAM OF 5 RATERS")
    print("="*80)
    print("""
Each rater should independently score each answer on a scale of 1-5:
    
    1 = Poor      - Completely incorrect, irrelevant to the question
    2 = Fair      - Mostly incorrect, some relevance
    3 = Good      - Mostly correct, may have minor issues
    4 = Very Good - Correct with minor issues or gaps
    5 = Excellent - Completely accurate and well-explained
    
Team Members:
    1. Rater 1 (your initials)
    2. Rater 2 (your initials)
    3. Rater 3 (your initials)
    4. Rater 4 (your initials)
    5. Rater 5 (your initials)

Example output format:
{
    "question_1": {
        "rater_1": 4,
        "rater_2": 4,
        "rater_3": 3,
        "rater_4": 4,
        "rater_5": 5
    },
    ...
}
    """)
    
    # Create template with 10 questions and 5 raters
    questions = [f"question_{i+1}" for i in range(10)]
    raters = [f"rater_{i+1}" for i in range(5)]
    
    template = {q: {r: None for r in raters} for q in questions}
    
    return template


def calculate_kappa_scores(team_ratings: Dict) -> Dict:
    """
    Calculate inter-rater reliability metrics (Cohen's Kappa and Fleiss' Kappa).
    
    Input: team_ratings Dict in format:
    {
        "question_1": {"rater_1": 5, "rater_2": 4, ...},
        "question_2": {"rater_1": 4, "rater_2": 4, ...},
        ...
    }
    
    Returns: Dict with:
        - pairwise_kappas: Cohen's Kappa for each rater pair
        - fleiss_kappa: Overall Fleiss' Kappa for all raters
        - average_ratings: Mean and std dev per question
    """
    
    # Get raters and questions
    first_question = team_ratings[list(team_ratings.keys())[0]]
    raters = sorted(first_question.keys())
    questions = sorted(team_ratings.keys())
    
    # 1. CALCULATE COHEN'S KAPPA (Pairwise)
    pairwise_kappas = {}
    cohen_scores = []
    
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            rater_i = raters[i]
            rater_j = raters[j]
            
            # Get scores for both raters
            scores_i = [team_ratings[q].get(rater_i) for q in questions]
            scores_j = [team_ratings[q].get(rater_j) for q in questions]
            
            # Calculate Cohen's Kappa
            kappa = cohen_kappa_score(scores_i, scores_j)
            pairwise_kappas[f"{rater_i} vs {rater_j}"] = kappa
            cohen_scores.append(kappa)
    
    # Average Cohen's Kappa
    avg_cohen = np.mean(cohen_scores) if cohen_scores else None
    
    # 2. CALCULATE FLEISS' KAPPA (Overall)
    fleiss_value = None
    
    if fleiss_kappa is not None:
        # Convert to matrix format for Fleiss' Kappa
        # Rows = questions, Columns = rating values (1-5)
        ratings_matrix = []
        
        for question in questions:
            row = [0] * 5  # 5 rating levels (1-5)
            for rater in raters:
                rating = team_ratings[question].get(rater)
                if rating is not None:
                    # Convert to 0-indexed (rating - 1)
                    row[int(rating) - 1] += 1
            ratings_matrix.append(row)
        
        ratings_matrix = np.array(ratings_matrix)
        fleiss_value = fleiss_kappa(ratings_matrix)
    
    # 3. PRINT ONLY COHEN AND FLEISS KAPPA SCORES
    print(f"\nCohen's Kappa (Average): {avg_cohen:.4f}")
    print(f"Fleiss' Kappa: {fleiss_value:.4f}")
    
    # Calculate average ratings (for internal use)
    average_ratings = {}
    for question in questions:
        scores = [team_ratings[question][rater] for rater in raters 
                  if team_ratings[question][rater] is not None]
        
        if scores:
            avg = np.mean(scores)
            std = np.std(scores)
            average_ratings[question] = {"avg": avg, "std": std}
    
    # Return results
    return {
        "pairwise_kappas": pairwise_kappas,
        "fleiss_kappa": fleiss_value,
        "average_ratings": average_ratings,
        "cohen_average": avg_cohen
    }


def save_ratings_to_csv(team_ratings: Dict, filename: str = "team_ratings.csv"):
    """
    Export team ratings to CSV file for sharing and archiving.
    
    Args:
        team_ratings: Dict with all rater scores
        filename: Output CSV filename (default: team_ratings.csv)
    
    Returns: Pandas DataFrame
    """
    
    # Convert dict to dataframe
    questions = sorted(team_ratings.keys())
    raters = sorted(team_ratings[questions[0]].keys())
    
    data = []
    for question in questions:
        row = {"Question": question}
        for rater in raters:
            row[rater.replace("_", " ").title()] = team_ratings[question][rater]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"\n✓ Team ratings saved to {filename}")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Testing kappa_scoring module...")
    
    # Create template
    template = create_manual_scoring_template()
    
    # Example ratings
    example_ratings = {
        "question_1": {"rater_1": 5, "rater_2": 4, "rater_3": 5, "rater_4": 4, "rater_5": 5},
        "question_2": {"rater_1": 4, "rater_2": 4, "rater_3": 4, "rater_4": 3, "rater_5": 4},
        "question_3": {"rater_1": 3, "rater_2": 2, "rater_3": 3, "rater_4": 2, "rater_5": 3},
        "question_4": {"rater_1": 5, "rater_2": 5, "rater_3": 5, "rater_4": 5, "rater_5": 4},
        "question_5": {"rater_1": 2, "rater_2": 3, "rater_3": 2, "rater_4": 1, "rater_5": 2},
        "question_6": {"rater_1": 4, "rater_2": 4, "rater_3": 3, "rater_4": 4, "rater_5": 4},
        "question_7": {"rater_1": 3, "rater_2": 3, "rater_3": 4, "rater_4": 3, "rater_5": 3},
        "question_8": {"rater_1": 1, "rater_2": 1, "rater_3": 2, "rater_4": 1, "rater_5": 1},
        "question_9": {"rater_1": 4, "rater_2": 5, "rater_3": 4, "rater_4": 5, "rater_5": 4},
        "question_10": {"rater_1": 3, "rater_2": 4, "rater_3": 3, "rater_4": 4, "rater_5": 3},
    }
    
    # Calculate kappa
    results = calculate_kappa_scores(example_ratings)
    
    # Save to CSV
    save_ratings_to_csv(example_ratings, "test_ratings.csv")
    
    print("\n✓ Module tested successfully!")
