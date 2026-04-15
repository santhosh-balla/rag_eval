"""
MANUAL SCORING EXAMPLE - Team of 5 Raters
==========================================

This script demonstrates how to:
1. Collect manual ratings from 5 team members
2. Calculate inter-rater reliability (Cohen's Kappa, Fleiss' Kappa)
3. Measure agreement between raters

Run this after evaluating the RAG pipeline to manually score the answers.
"""

import sys
from kappa_scoring import (
    create_manual_scoring_template,
    calculate_kappa_scores,
    save_ratings_to_csv
)


def get_team_ratings_from_user():
    """
    Interactively collect ratings from team of 5 raters.
    Each rater rates all 10 questions on a scale of 1-5.
    """
    print("\n" + "="*80)
    print("INTER-RATER RELIABILITY SCORING")
    print("="*80)
    
    # Initialize template
    team_ratings = create_manual_scoring_template()
    
    # For demonstration, using sample ratings
    # In real scenario, each rater would enter their own ratings
    sample_ratings = {
        "question_1": {"rater_1": 5, "rater_2": 4, "rater_3": 5, "rater_4": 4, "rater_5": 5},  # High agreement
        "question_2": {"rater_1": 4, "rater_2": 4, "rater_3": 4, "rater_4": 3, "rater_5": 4},  # High agreement
        "question_3": {"rater_1": 3, "rater_2": 2, "rater_3": 3, "rater_4": 2, "rater_5": 3},  # Moderate agreement
        "question_4": {"rater_1": 5, "rater_2": 5, "rater_3": 5, "rater_4": 5, "rater_5": 4},  # Very high agreement
        "question_5": {"rater_1": 2, "rater_2": 3, "rater_3": 2, "rater_4": 1, "rater_5": 2},  # Moderate agreement
        "question_6": {"rater_1": 4, "rater_2": 4, "rater_3": 3, "rater_4": 4, "rater_5": 4},  # High agreement
        "question_7": {"rater_1": 3, "rater_2": 3, "rater_3": 4, "rater_4": 3, "rater_5": 3},  # High agreement
        "question_8": {"rater_1": 1, "rater_2": 1, "rater_3": 2, "rater_4": 1, "rater_5": 1},  # Very high agreement
        "question_9": {"rater_1": 4, "rater_2": 5, "rater_3": 4, "rater_4": 5, "rater_5": 4},  # High agreement
        "question_10": {"rater_1": 3, "rater_2": 4, "rater_3": 3, "rater_4": 4, "rater_5": 3}, # High agreement
    }
    
    return sample_ratings


def manual_scoring_workflow():
    """Complete workflow for manual scoring and kappa calculation."""
    
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         MANUAL SCORING WORKFLOW - INTER-RATER RELIABILITY ANALYSIS        ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Collect ratings
    print("\n📝 STEP 1: Collecting Ratings from 5 Team Members")
    print("-" * 80)
    print("Each rater independently scores each answer (1-5 scale)")
    print("1 = Poor | 2 = Fair | 3 = Good | 4 = Very Good | 5 = Excellent")
    
    # In production, you would collect ratings interactively:
    # team_ratings = {}
    # for i in range(5):
    #     print(f"\n🔄 Enter ratings for Rater {i+1}:")
    #     for question_num in range(1, 11):
    #         while True:
    #             try:
    #                 rating = int(input(f"   Question {question_num} (1-5): "))
    #                 if 1 <= rating <= 5:
    #                     team_ratings[f"question_{question_num}"][f"rater_{i+1}"] = rating
    #                     break
    #                 else:
    #                     print("   ⚠️  Please enter a number between 1 and 5")
    #             except ValueError:
    #                 print("   ⚠️  Please enter a valid number")
    
    # For this example, using sample data
    print("✓ Using sample ratings for demonstration")
    team_ratings = get_team_ratings_from_user()
    
    # Step 2: Save ratings to CSV
    print("\n📊 STEP 2: Saving Ratings to CSV")
    print("-" * 80)
    df = save_ratings_to_csv(team_ratings, "team_ratings.csv")
    print("\n✓ Ratings saved to team_ratings.csv")
    print("\n Sample data:")
    print(df.head(10).to_string())
    
    # Step 3: Calculate kappa scores
    print("\n🔍 STEP 3: Calculating Inter-Rater Reliability")
    print("-" * 80)
    kappa_results = calculate_kappa_scores(team_ratings)
    
    # Step 4: Summary and recommendations
    print("\n" + "="*80)
    print("SUMMARY & INTERPRETATION")
    print("="*80)
    
    if kappa_results:
        print("\n✓ Kappa Score Results:")
        
        # Pairwise kappas
        print("\nPairwise Agreement Levels:")
        for pair, kappa in kappa_results["pairwise_kappas"].items():
            print(f"  {pair}: {kappa}")
        
        # Fleiss kappa
        if kappa_results["fleiss_kappa"] is not None:
            fleiss_val = kappa_results["fleiss_kappa"]
            print(f"\nOverall Team Agreement (Fleiss' Kappa): {fleiss_val}")
            
            if fleiss_val >= 0.81:
                print("  ✅ Excellent - Team is highly aligned on scoring")
            elif fleiss_val >= 0.61:
                print("  ✓ Good - Team shows substantial agreement")
            elif fleiss_val >= 0.41:
                print("  ⚠️  Fair - Team shows moderate agreement (may need discussion)")
            else:
                print("  ❌ Poor - Team should discuss and align on scoring criteria")
        
        # Average ratings
        print("\nPer-Question Average Ratings:")
        for question, stats in kappa_results["average_ratings"].items():
            avg = stats["avg"]
            std = stats["std"]
            print(f"  {question}: {avg:.2f} ± {std:.2f}")


def create_scoring_instructions():
    """Create a detailed instruction document for raters."""
    
    instructions = """
╔════════════════════════════════════════════════════════════════════════════╗
║               MANUAL SCORING INSTRUCTIONS FOR RATERS                      ║
╚════════════════════════════════════════════════════════════════════════════╝

OVERVIEW:
---------
You are one of 5 team members rating the quality of RAG-generated answers about 
Honeywell products. Your independent ratings will be combined to measure inter-rater 
reliability and validate our RAG system.

RATING SCALE (1-5):
-------------------

5 = EXCELLENT
   ✓ Answer is completely accurate
   ✓ All key information is included
   ✓ Explanation is clear and well-structured
   ✓ Directly addresses the question

4 = VERY GOOD
   ✓ Answer is mostly accurate with only minor issues
   ✓ Most key information is included
   ✓ Explanation is generally clear
   ✓ Minor gaps or improvements possible

3 = GOOD
   ✓ Answer has some correct information
   ✓ Some key information missing
   ✓ Explanation could be clearer
   ✓ Basic question requirements met

2 = FAIR
   ✗ Answer has significant errors or omissions
   ✗ Most key information incomplete
   ✗ Explanation is unclear or confusing
   ✗ Question not fully addressed

1 = POOR
   ✗ Answer is completely incorrect or irrelevant
   ✗ No useful information provided
   ✗ Fails to address the question
   ✗ Unusable for end users

SCORING CRITERIA:
-----------------
When scoring each answer, consider:

1. ACCURACY (40%)
   - Are the facts correct?
   - Are specifications accurate?
   - Are there any hallucinations or false information?

2. COMPLETENESS (30%)
   - Does it answer all parts of the question?
   - Are all important details included?
   - Is anything critical missing?

3. CLARITY (20%)
   - Is the explanation easy to understand?
   - Is it well-organized?
   - Would end users understand this?

4. RELEVANCE (10%)
   - Is the answer focused on the question?
   - Is there unnecessary or off-topic information?

IMPORTANT NOTES:
----------------
• Score independently without discussing with other raters
• Use the numeric scale only (1-5)
• Rate all 10 questions for consistency
• If uncertain, rate honestly - disagreement is valuable data
• Inter-rater agreement (kappa score) measures our scoring consistency

HOW TO RECORD YOUR SCORES:
--------------------------
Please provide your ratings in this format:

Question 1: [your rating 1-5]
Question 2: [your rating 1-5]
...
Question 10: [your rating 1-5]

Or in CSV format:
question_1,4
question_2,5
question_3,3
...

QUESTIONS TO EVALUATE:
----------------------
1. What are the key features and specifications of the Honeywell Home T4 Pro?
2. What are the power requirements for the Honeywell Wi-Fi RTH6500WF?
3. What is the Honeywell BACnet TB3026B and its communication specs?
4. What sensor technologies does the Honeywell XNX Universal Transmitter support?
5. What are the output module options for the Honeywell XNX?
6. What are the specifications of the Honeywell Xenon 1902 Scanner?
7. What temperature control modes does the Honeywell T3 Pro offer?
8. What are the installation requirements for the Honeywell BACnet TB3026B?
9. What is the power source for the Honeywell RTH2300/RTH221?
10. What are the industrial applications of the Honeywell XNX Universal Transmitter?

INTER-RATER RELIABILITY (KAPPA SCORE):
---------------------------------------
After all 5 raters have scored, the team calculates:

• Cohen's Kappa: Measures pairwise agreement between each pair of raters
• Fleiss' Kappa: Measures overall agreement among all 5 raters

Interpretation:
- 0.81-1.00: Almost Perfect Agreement ✅
- 0.61-0.80: Substantial Agreement ✓
- 0.41-0.60: Moderate Agreement ⚠️
- 0.21-0.40: Fair Agreement ⚠️
- < 0.20: Slight/Poor Agreement ❌

If kappa is LOW (< 0.41):
  → Discuss scoring differences
  → Clarify scoring criteria
  → May need second round of scoring with alignment

If kappa is HIGH (> 0.60):
  → Your team's ratings are consistent
  → Results are reliable for system evaluation
    """
    
    return instructions


if __name__ == "__main__":
    # Run the manual scoring workflow
    manual_scoring_workflow()
    
    # Optionally print detailed instructions
    print("\n" + "="*80)
    print("For detailed scoring instructions, see above or run:")
    print("  python -c \"from manual_scoring_example import create_scoring_instructions; print(create_scoring_instructions())\"")
    print("="*80)
