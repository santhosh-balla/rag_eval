"""
QUICK START: Manual Scoring & Kappa Analysis
=============================================

Run this file to calculate Cohen Kappa and Fleiss Kappa scores.
"""

from kappa_scoring import calculate_kappa_scores


def main():
    # Print Cohen's Kappa Formula and Guidance
    print("\n" + "="*80)
    print("COHEN'S KAPPA SCORE - FORMULA, GUIDANCE & RESULTS")
    print("="*80)
    
    print("\n📐 FORMULA:")
    print("-" * 80)
    print("""
    Cohen's Kappa = (P_o - P_e) / (1 - P_e)
    
    Where:
      P_o = Observed Agreement (proportion of times two raters agree)
      P_e = Expected Agreement (proportion of agreement by chance)
    
    The formula measures inter-rater reliability by accounting for 
    agreement that could occur by random chance alone.
    """)
    
    print("\n📖 GUIDANCE:")
    print("-" * 80)
    print("""
    Cohen's Kappa ranges from -1 to +1:
    
      0.81 - 1.00  = Almost Perfect Agreement (✅ Excellent)
      0.61 - 0.80  = Substantial Agreement (✓ Good)
      0.41 - 0.60  = Moderate Agreement (⚠️ Fair)
      0.21 - 0.40  = Fair Agreement (⚠️ Needs improvement)
      0.00 - 0.20  = Slight Agreement (❌ Poor)
      < 0.00       = Poor/No Agreement (❌ Unacceptable)
    
    Interpretation:
      - Use this metric when you have 2 raters scoring the same items
      - Higher scores mean raters are in better agreement
      - Scores > 0.7 indicate reliable/trustworthy ratings
    """)
    
    # Highly aligned team ratings (0.81-1.00 range)
    example_ratings = {
        "question_1": {"rater_1": 5, "rater_2": 5, "rater_3": 5, "rater_4": 5, "rater_5": 5},
        "question_2": {"rater_1": 4, "rater_2": 4, "rater_3": 4, "rater_4": 4, "rater_5": 4},
        "question_3": {"rater_1": 5, "rater_2": 5, "rater_3": 5, "rater_4": 5, "rater_5": 5},
        "question_4": {"rater_1": 4, "rater_2": 4, "rater_3": 4, "rater_4": 5, "rater_5": 4},
        "question_5": {"rater_1": 5, "rater_2": 5, "rater_3": 5, "rater_4": 5, "rater_5": 5},
        "question_6": {"rater_1": 3, "rater_2": 3, "rater_3": 3, "rater_4": 3, "rater_5": 3},
        "question_7": {"rater_1": 4, "rater_2": 4, "rater_3": 4, "rater_4": 4, "rater_5": 4},
        "question_8": {"rater_1": 5, "rater_2": 5, "rater_3": 5, "rater_4": 5, "rater_5": 5},
        "question_9": {"rater_1": 4, "rater_2": 4, "rater_3": 4, "rater_4": 4, "rater_5": 4},
        "question_10": {"rater_1": 5, "rater_2": 5, "rater_3": 5, "rater_4": 5, "rater_5": 5},
    }
    
    print("\n📊 RESULTS:")
    print("-" * 80)
    results = calculate_kappa_scores(example_ratings)
    
    print("\n🔍 INTERPRETATION:")
    print("-" * 80)
    
    cohen_avg = results.get("cohen_average")
    if cohen_avg:
        if cohen_avg >= 0.81:
            level = "Almost Perfect Agreement ✅"
            meaning = "Raters are highly aligned. Ratings are reliable and trustworthy."
        elif cohen_avg >= 0.61:
            level = "Substantial Agreement ✓"
            meaning = "Raters mostly agree. Ratings are generally reliable."
        elif cohen_avg >= 0.41:
            level = "Moderate Agreement ⚠️"
            meaning = "Raters have mixed agreement. Some discussion needed."
        elif cohen_avg >= 0.21:
            level = "Fair Agreement ⚠️"
            meaning = "Raters often disagree. Team needs alignment on criteria."
        else:
            level = "Slight/Poor Agreement ❌"
            meaning = "Raters rarely agree. Scoring criteria must be revised."
        
        print(f"  Score: {cohen_avg:.4f}")
        print(f"  Level: {level}")
        print(f"  Meaning: {meaning}")


if __name__ == "__main__":
    main()
