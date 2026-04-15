"""
RAG Pipeline with Groq & RAGAs Evaluation
==========================================
This script builds a baseline RAG (Retrieval-Augmented Generation) pipeline using:
- Groq API (llama-3.1-8b-instant) for LLM inference
- OpenAI embeddings (text-embedding-3-small) for vector embeddings
- Chroma for local vector storage
- RAGAs for automated evaluation metrics

Domain data includes: Java Textbook, Terms & Conditions, and Product Catalog.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, context_recall
from datasets import Dataset

# Import for inter-rater reliability (kappa scores)
from sklearn.metrics import cohen_kappa_score
try:
    from statsmodels.stats.inter_rater import fleiss_kappa
except ImportError:
    fleiss_kappa = None

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ ERROR: GROQ_API_KEY not found!")
    print("Please create a .env file with GROQ_API_KEY variable.")
    sys.exit(1)

# Configuration flags
SKIP_METRICS_EVALUATION = os.getenv("SKIP_METRICS", "false").lower() == "true"

# Paths to domain text files
DOMAIN_FILES = {
    "java_textbook": "java_textbook.txt",
    "terms_conditions": "terms_conditions.txt",
    "product_catalog": "honeywell_products.txt",
}

CHROMA_DB_PATH = "./chroma_db"

# ============================================================================
# PHASE 1: VECTOR STORE SETUP
# ============================================================================

def load_domain_documents() -> str:
    """
    Load and concatenate all domain text files into a single corpus.
    Returns concatenated text from all domain files.
    """
    print("\n📄 Loading domain documents...")
    corpus = ""
    
    for domain_name, filename in DOMAIN_FILES.items():
        filepath = Path(filename)
        if not filepath.exists():
            print(f"⚠️  Warning: {filename} not found. Skipping {domain_name}.")
            continue
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            corpus += f"\n--- {domain_name.upper()} ---\n{content}\n"
            print(f"✓ Loaded: {domain_name} ({len(content)} characters)")
    
    if not corpus:
        print("❌ No domain documents found!")
        sys.exit(1)
    
    print(f"✓ Total corpus size: {len(corpus)} characters")
    return corpus


def create_chroma_store(reset: bool = False) -> Chroma:
    """
    Create and populate a Chroma vector store with domain documents.
    
    Args:
        reset: If True, delete existing Chroma database and start fresh.
    
    Returns:
        Chroma vector store instance.
    """
    print("\n🔧 Setting up Chroma vector store...")
    
    # Reset if requested
    if reset and Path(CHROMA_DB_PATH).exists():
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)
        print("✓ Existing Chroma database cleared")
    
    # Load domain documents
    corpus = load_domain_documents()
    
    # Split text into chunks
    print("\n✂️  Splitting corpus into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(corpus)
    print(f"✓ Created {len(chunks)} chunks")
    
    # Initialize HuggingFace embeddings (free, local)
    print("\n🔐 Initializing HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create Chroma vector store
    print("\n💾 Creating Chroma vector store...")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )
    print(f"✓ Vector store created with {len(chunks)} chunks")
    
    return vector_store


def load_chroma_store() -> Chroma:
    """Load existing Chroma vector store from disk."""
    print("\n📂 Loading Chroma vector store from disk...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    print(f"✓ Vector store loaded")
    return vector_store


# ============================================================================
# PHASE 2: RAG PIPELINE
# ============================================================================

def get_rag_response(query: str, vector_store: Chroma) -> Tuple[str, List[str]]:
    """
    Execute RAG pipeline: retrieve top 5 chunks and generate answer using Groq.
    
    Args:
        query: User query string.
        vector_store: Chroma vector store instance.
    
    Returns:
        Tuple of (generated_answer, list_of_retrieved_contexts)
    """
    print(f"\n🔍 Processing query: '{query}'")
    
    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=GROQ_API_KEY
    )
    
    # Create RAG prompt template
    rag_prompt = PromptTemplate(
        template="""You are a helpful AI assistant. Answer the following question based on the provided context.
        
Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )
    
    # Retrieve top 5 chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(query)
    
    # Extract context strings
    context_strings = [doc.page_content for doc in retrieved_docs]
    context_text = "\n---\n".join(context_strings)
    
    print(f"✓ Retrieved {len(context_strings)} chunks")
    
    # Generate answer using Groq
    prompt = rag_prompt.format(context=context_text, question=query)
    message = HumanMessage(content=prompt)
    response = llm.invoke([message])
    answer = response.content
    
    print(f"✓ Answer generated (length: {len(answer)} chars)")
    
    return answer, context_strings


# ============================================================================
# PHASE 3: GOLD-STANDARD EVALUATION DATASET
# ============================================================================

def get_evaluation_dataset() -> List[Dict]:
    """
    Define gold-standard evaluation triplets (Question, Context, Ground Truth).
    These represent the reference standards for RAGAs evaluation.
    
    Returns:
        List of evaluation triplets.
    """
    evaluation_data = [
        {
            "question": "What are the key features and specifications of the Honeywell Home T4 Pro Programmable Thermostat?",
            "ground_truth": "The Honeywell Home T4 Pro (TH4110U2005) is a 7-day programmable thermostat with four programmable time periods per day. It features adaptive intelligent recovery that learns how long the HVAC system takes to reach target temperature. It includes compressor protection and is compatible with 24-volt heating/cooling systems only. Temperature range is 40-90°F for heat and 50-99°F for cool. It runs on two AA alkaline batteries and includes keypad lockout with a default unlock password of 1234.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What are the power requirements and system compatibility for the Honeywell Wi-Fi 7-Day Programmable Thermostat (RTH6500WF)?",
            "ground_truth": "The Honeywell RTH6500WF Wi-Fi thermostat requires a C (common) wire to supply 24 VAC continuous power and will NOT work on 120/240 volt systems. It has an 1800 mAh lithium-ion battery and connects to the home Wi-Fi network for remote access through the Honeywell Home app. Default energy-saving programs can reduce heating and cooling expenses by as much as 33%. It supports Heat, Cool, Off, and EM HEAT modes for heat pumps with auxiliary heat.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What is the Honeywell BACnet Fixed Function Thermostat (TB3026B) and what are its communication specifications?",
            "ground_truth": "The Honeywell TB3026B is a configurable commercial thermostat with 19 pre-loaded applications for fan coil, heat pump, and conventional rooftop HVAC systems. It features built-in temperature and humidity sensors and operates as a fully functioning BACnet controller. It communicates via BACnet MS/TP over EIA-485 twisted-pair at speeds of 9.6, 19.2, 38.4, or 76.8 Kbps. The network supports up to 128 master devices per network and 255 total devices. Maximum segment length is 4000 feet with repeaters required beyond this distance.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What sensor technologies and gas detection ranges does the Honeywell XNX Universal Transmitter support?",
            "ground_truth": "The Honeywell XNX Universal Transmitter is a high-specification gas transmitter compatible with Honeywell Analytics detectors. It supports three sensor technologies: Electrochemical (EC), Infrared (IR), and Catalytic Bead (MPD). The combustible gas detection range is 0 to 100% LEL/LFL. It offers more than 200 unique configurations and is certified for hazardous areas including Zone 1, 2, 21, 22, and North American Class I and II Division 1 or 2. Operating temperature ranges from -40°C to +65°C depending on sensor type.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What are the output module options available for the Honeywell XNX Universal Transmitter?",
            "ground_truth": "The Honeywell XNX Universal Transmitter offers the following mutually exclusive output module options: 4-20 mA with HART 6.0 protocol (standard), Relay Module with three user-configurable relays (2 SPCO alarm relays, 1 SPCO fault relay), Modbus Module with isolated RS-485 output for multi-drop Modbus RTU networks, and Foundation Fieldbus Module for multi-drop Foundation Fieldbus H1 network connections. The standard 4-20 mA/HART output is compatible with all optional modules.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What are the specifications and capabilities of the Honeywell Xenon 1902 Cordless Area-Imaging Scanner?",
            "ground_truth": "The Honeywell Xenon 1902 is a cordless handheld area-imaging barcode scanner using Bluetooth wireless interface. It features an 1800 mAh lithium-ion rechargeable battery capable of up to 50,000 reads per charge. Standby power consumption is 0.5W with input voltage of 5 VDC. The scanner weighs 214 grams, has an IP41 protection rating (protection against objects 1mm or greater and vertically falling water drops), and features a rugged shockproof design.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What temperature control range and system modes does the Honeywell Home T3 Pro Smart Thermostat offer?",
            "ground_truth": "The Honeywell Home T3 Pro is a programmable thermostat designed for both conventional and heat pump HVAC systems. It provides scheduling capabilities, a backlit display, and simple three-button control. The thermostat is powered by two AA batteries (included) and includes a UWP mounting system with a decorative cover plate (4.72 inches H x 5.9 inches W). It is designed as an entry-level smart home device with straightforward operation suitable for residential applications.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What are the installation requirements and network specifications for the Honeywell BACnet TB3026B?",
            "ground_truth": "The Honeywell TB3026B requires installation by a trained, experienced service technician with power disconnected before installation. It must be mounted approximately 4 feet above the floor in an area with good air circulation at average temperature, complying with Americans with Disabilities Act requirements. It requires 24 VAC on terminal 1. The network uses shielded twisted-pair cable with characteristic impedance between 100-130 ohms, distributed capacitance less than 30 pF/foot, 18 AWG wire gauge (minimum 22 AWG acceptable), and matched precision terminating resistors (1/4 W, +/-1%, 80-130 ohms) at each segment end.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What is the power source and physical specification for the Honeywell RTH2300/RTH221 Series Programmable Thermostat?",
            "ground_truth": "The Honeywell RTH2300/RTH221 Series is an entry-level 5-2 day programmable thermostat that is pre-programmed and ready to operate out of the box - users only need to set the time and day. It features a backlight that stays lit for 12 seconds when any button is pressed. The thermostat includes compressor protection that engages when 'Cool On' flashes on the display. It comes with a one-year limited warranty against manufacturing defects (excluding battery). Warranty does not cover removal/reinstallation costs or damage caused by the consumer.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What are the industrial applications and safety features of the Honeywell XNX Universal Transmitter?",
            "ground_truth": "The Honeywell XNX Universal Transmitter is designed for industrial applications including upstream and downstream oil and gas, chemical, wastewater, and other industrial settings. The housing is painted LM25 aluminum standard with 316 stainless steel marine-grade coating optional. It has 24 VDC power, weighs 2.8 kg, and has five threaded cable entry ports available as 5x M25 (ATEX/IECEx) or 5x 3/4\" NPT (UL/CSA). It features a 2.5-inch high-resolution backlit LCD display and non-intrusive four-button magnetic interface. Remote sensor option extends up to 50 feet from transmitter with optional wireless capabilities.",
            "context_domain": "product_catalog"
        },
    ]
    
    return evaluation_data


# ============================================================================
# INTER-RATER RELIABILITY (Manual Team Scoring)
# ============================================================================

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
    
    # Template for 10 questions with 5 raters
    scoring_template = {
        f"question_{i}": {
            "rater_1": None,
            "rater_2": None,
            "rater_3": None,
            "rater_4": None,
            "rater_5": None
        }
        for i in range(1, 11)
    }
    
    return scoring_template


def calculate_kappa_scores(team_ratings: Dict) -> Dict:
    """
    Calculate inter-rater reliability using Cohen's Kappa and Fleiss' Kappa.
    
    Args:
        team_ratings: Dictionary with ratings from 5 raters per question
        
    Returns:
        Dictionary with kappa scores and agreement statistics
        
    Kappa Interpretation:
    - 0.81-1.00: Almost Perfect Agreement
    - 0.61-0.80: Substantial Agreement
    - 0.41-0.60: Moderate Agreement
    - 0.21-0.40: Fair Agreement
    - 0.00-0.20: Slight Agreement
    - < 0.00: Poor/No Agreement
    """
    
    print("\n" + "="*80)
    print("INTER-RATER RELIABILITY ANALYSIS (KAPPA SCORES)")
    print("="*80)
    
    ratings_array = []
    questions_list = []
    
    # Convert ratings to array format: (n_questions, n_raters)
    for question, raters_dict in team_ratings.items():
        ratings = [
            raters_dict.get("rater_1"),
            raters_dict.get("rater_2"),
            raters_dict.get("rater_3"),
            raters_dict.get("rater_4"),
            raters_dict.get("rater_5")
        ]
        
        # Skip if any rating is missing
        if None in ratings:
            continue
            
        ratings_array.append(ratings)
        questions_list.append(question)
    
    if not ratings_array:
        print("⚠️  No complete ratings available for kappa calculation")
        return None
    
    ratings_array = np.array(ratings_array)
    
    results = {
        "questions_scored": len(questions_list),
        "pairwise_kappas": {},
        "fleiss_kappa": None
    }
    
    # Calculate pairwise Cohen's Kappa between each pair of raters
    rater_names = ["Rater 1", "Rater 2", "Rater 3", "Rater 4", "Rater 5"]
    print("\n📊 Pairwise Cohen's Kappa Scores (Agreement between two raters):")
    print("-" * 80)
    
    for i in range(5):
        for j in range(i+1, 5):
            rater_i_scores = ratings_array[:, i]
            rater_j_scores = ratings_array[:, j]
            
            try:
                kappa = cohen_kappa_score(rater_i_scores, rater_j_scores)
                pair_name = f"{rater_names[i]} vs {rater_names[j]}"
                results["pairwise_kappas"][pair_name] = round(kappa, 4)
                
                # Interpret agreement level
                if kappa >= 0.81:
                    agreement_level = "Almost Perfect"
                elif kappa >= 0.61:
                    agreement_level = "Substantial"
                elif kappa >= 0.41:
                    agreement_level = "Moderate"
                elif kappa >= 0.21:
                    agreement_level = "Fair"
                elif kappa >= 0:
                    agreement_level = "Slight"
                else:
                    agreement_level = "Poor"
                
                print(f"  {pair_name}: {kappa:.4f} ({agreement_level})")
            except Exception as e:
                print(f"  Error calculating {rater_names[i]} vs {rater_names[j]}: {e}")
    
    # Calculate Fleiss' Kappa (overall agreement among all 5 raters)
    if fleiss_kappa is not None:
        print("\n📊 Fleiss' Kappa Score (Overall agreement among all 5 raters):")
        print("-" * 80)
        try:
            # Convert to format required by fleiss_kappa: (n_subjects, n_categories)
            # We need to convert numerical ratings to category counts
            max_rating = 5
            n_questions = ratings_array.shape[0]
            
            # Create contingency table
            contingency_table = np.zeros((n_questions, max_rating))
            for i in range(n_questions):
                for rating in ratings_array[i, :]:
                    contingency_table[i, int(rating)-1] += 1
            
            fleiss_k = fleiss_kappa(contingency_table)
            results["fleiss_kappa"] = round(fleiss_k, 4)
            
            if fleiss_k >= 0.81:
                agreement_level = "Almost Perfect"
            elif fleiss_k >= 0.61:
                agreement_level = "Substantial"
            elif fleiss_k >= 0.41:
                agreement_level = "Moderate"
            elif fleiss_k >= 0.21:
                agreement_level = "Fair"
            elif fleiss_k >= 0:
                agreement_level = "Slight"
            else:
                agreement_level = "Poor"
            
            print(f"  Fleiss' Kappa: {fleiss_k:.4f} ({agreement_level})")
            print(f"  Questions Evaluated: {n_questions}/10")
            
        except Exception as e:
            print(f"  Error calculating Fleiss' Kappa: {e}")
    else:
        print("\n⚠️  statsmodels not installed. Install with: pip install statsmodels")
    
    # Calculate average rating per question
    print("\n📈 Average Rating per Question:")
    print("-" * 80)
    avg_ratings = {}
    for idx, question in enumerate(questions_list):
        avg_rating = np.mean(ratings_array[idx, :])
        std_rating = np.std(ratings_array[idx, :])
        avg_ratings[question] = {"avg": round(avg_rating, 2), "std": round(std_rating, 2)}
        print(f"  {question}: {avg_rating:.2f} ± {std_rating:.2f}")
    
    results["average_ratings"] = avg_ratings
    
    return results


def save_ratings_to_csv(team_ratings: Dict, filename: str = "team_ratings.csv"):
    """Save team ratings to CSV file for easier sharing and management."""
    df_data = []
    
    for question, raters_dict in team_ratings.items():
        row = {
            "Question": question,
            "Rater_1": raters_dict.get("rater_1"),
            "Rater_2": raters_dict.get("rater_2"),
            "Rater_3": raters_dict.get("rater_3"),
            "Rater_4": raters_dict.get("rater_4"),
            "Rater_5": raters_dict.get("rater_5"),
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df.to_csv(filename, index=False)
    print(f"\n✓ Team ratings saved to {filename}")
    return df



# ============================================================================
# PHASE 4: RAGAs EVALUATION
# ============================================================================

def evaluate_rag_pipeline(vector_store: Chroma, evaluation_data: List[Dict]) -> Dict:
    """
    Evaluate RAG pipeline using RAGAs metrics.
    Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall.
    
    Args:
        vector_store: Chroma vector store instance.
        evaluation_data: List of evaluation triplets.
    
    Returns:
        Dictionary with RAGAs evaluation scores.
    """
    print("\n🧪 Evaluating RAG pipeline with RAGAs...")
    
    # Initialize Groq as the judge LLM for RAGAs
    judge_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )
    
    # Generate pipeline answers for all evaluation queries
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in evaluation_data:
        question = item["question"]
        gt = item["ground_truth"]
        
        # Get RAG response
        answer, retrieved_contexts = get_rag_response(question, vector_store)
        
        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)  # RAGAs expects list of context strings
        ground_truths.append(gt)
    
    print(f"✓ Generated {len(answers)} answers")
    
    # Create RAGAs dataset
    rag_eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    # Skip metrics evaluation if flag is set
    if SKIP_METRICS_EVALUATION:
        print("\n⏭️  Skipping RAGAs metrics evaluation (SKIP_METRICS=true)")
        return None, questions, answers, contexts
    
    # Configure RAGAs metrics with Groq judge
    print("\n📊 Computing RAGAs metrics (Faithfulness, Context Precision, Context Recall)...")
    print("⚠️  Using simplified metrics to work with smaller LLM models (n=1 only)")
    
    # Create metric instances configured with Groq
    # Note: Removed 'answer_relevancy' as it requires n>1 sampling unsupported by small models
    metrics_to_evaluate = [
        faithfulness,           # ✅ Works with n=1
        context_precision,      # ✅ Works with n=1
        context_recall          # ✅ Works with n=1
        # answer_relevancy     # ❌ Removed - causes "n must be at most 1" error
    ]
    
    try:
        # Run evaluation with simplified settings for smaller models
        results = evaluate(
            dataset=rag_eval_dataset,
            metrics=metrics_to_evaluate,
            llm=judge_llm,
            embeddings=HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            ),
            batch_size=1  # Process one at a time to avoid rate limits
        )
        
        print("✓ Evaluation complete")
        return results, questions, answers, contexts
    
    except Exception as e:
        print(f"⚠️  Error during RAGAs evaluation: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This can happen with smaller models that have limited token budgets.")
        print("Returning partial results without metrics...")
        return None, questions, answers, contexts


# ============================================================================
# PHASE 5: OUTPUT FORMATTING
# ============================================================================

def print_results(results: Dict, questions: List[str], answers: List[str], contexts: List[List[str]]):
    """
    Print formatted evaluation results and sample Q&A.
    
    Args:
        results: RAGAs evaluation results dictionary (or None if metrics skipped).
        questions: List of questions evaluated.
        answers: List of generated answers.
        contexts: List of retrieved context strings for each query.
    """
    print("\n" + "="*80)
    print("RAG PIPELINE EVALUATION RESULTS")
    print("="*80)
    
    # Print metrics if available
    if results:
        print("\n✅ RAGAS AUTOMATED METRICS")
        print("-" * 80)
        # Extract metric scores from RAGAs EvaluationResult object
        try:
            # RAGAs returns an EvaluationResult object with metric attributes
            metrics_dict = {}
            
            # Try to access as attributes first (RAGAs EvaluationResult object)
            if hasattr(results, 'faithfulness'):
                metrics_dict["Faithfulness"] = float(results.faithfulness) if results.faithfulness is not None else 0
                metrics_dict["Context Precision"] = float(results.context_precision) if results.context_precision is not None else 0
                metrics_dict["Context Recall"] = float(results.context_recall) if results.context_recall is not None else 0
                # answer_relevancy removed - not computed
            # Fallback: treat as dictionary
            elif isinstance(results, dict):
                metrics_dict = {
                    "Faithfulness": results.get("faithfulness", 0) or 0,
                    "Context Precision": results.get("context_precision", 0) or 0,
                    "Context Recall": results.get("context_recall", 0) or 0
                }
            
            # Create metrics dataframe and display
            if metrics_dict:
                metrics_df = pd.DataFrame([metrics_dict])
                
                print("\n📈 RAGAs Metrics Table:")
                print("-" * 80)
                print(metrics_df.to_string())
                print("-" * 80)
                
                # Calculate and print average score (excluding NaN values)
                valid_scores = [v for v in metrics_dict.values() if not np.isnan(v)]
                if valid_scores:
                    avg_score = np.mean(valid_scores)
                    print(f"\n📊 Average Score: {avg_score:.4f}")
                    print(f"📝 Note: NaN values indicate metrics that couldn't be computed with the current model")
        
        except Exception as e:
            print(f"⚠️  Could not format metrics table: {e}")
            print(f"Raw results: {results}")
            # Try to print raw results if available
            if hasattr(results, '__dict__'):
                print(f"Results attributes: {results.__dict__}")
    else:
        print("\n⏭️  RAGAs Metrics Skipped (SKIP_METRICS=true)")
        print("-" * 80)
        print("📌 To enable automated metrics evaluation:")
        print("   1. Set SKIP_METRICS=false in .env file")
        print("   2. Or run: export SKIP_METRICS=false && python rag_pipeline.py")
        print("   3. Ensure GROQ_API_KEY is set with sufficient quota")
    
    # Print evaluation summary (always show this)
    if questions and answers:
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        print(f"✓ Total Questions Evaluated: {len(questions)}")
        print(f"✓ Answers Generated: {len(answers)}")
        print(f"✓ Context Chunks Retrieved: {sum(len(c) for c in contexts)}")
        if contexts:
            print(f"✓ Average Context Chunks per Query: {sum(len(c) for c in contexts) / len(contexts):.1f}")
    
    # Print sample query with answer and contexts
    if questions and answers and contexts:
        print("\n" + "="*80)
        print("SAMPLE QUERY DEMO (Query 1 of {})".format(len(questions)))
        print("="*80)
        
        sample_idx = 0  # First query as demo
        
        print(f"\n❓ Question:")
        print(f"   {questions[sample_idx]}\n")
        
        print(f"💬 Generated Answer:")
        print(f"   {answers[sample_idx]}\n")
        
        print(f"📚 Retrieved Context Chunks ({len(contexts[sample_idx])} chunks):")
        print("-" * 80)
        for i, context in enumerate(contexts[sample_idx], 1):
            print(f"\n[Chunk {i}]")
            print(context[:300] + ("..." if len(context) > 300 else ""))
        print("-" * 80)
        
        # Show how to enable manual scoring for team evaluation
        print("\n" + "="*80)
        print("📋 NEXT STEPS: Manual Team Evaluation")
        print("="*80)
        print("To conduct inter-rater reliability analysis with your team:")
        print("  1. Run: python QUICK_START.py")
        print("  2. Or use: from kappa_scoring import calculate_kappa_scores")
        print("  3. See: SCORING_README.md for complete workflow")
        print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main orchestration function."""
    print("\n" + "="*80)
    print(" RAG PIPELINE WITH GROQ & RAGAs EVALUATION")
    print("="*80)
    
    results = None
    questions = []
    answers = []
    contexts = []
    
    try:
        # Step 1: Create/Load Vector Store
        if Path(CHROMA_DB_PATH).exists():
            print("\n🔄 Chroma database exists. Using existing database...")
            vector_store = load_chroma_store()
        else:
            print("\n🆕 Creating new Chroma vector store...")
            vector_store = create_chroma_store(reset=True)
        
        # Step 2: Load evaluation dataset
        print("\n📋 Loading evaluation dataset...")
        eval_data = get_evaluation_dataset()
        print(f"✓ Loaded {len(eval_data)} evaluation triplets")
        
        # Step 3: Evaluate RAG pipeline
        results, questions, answers, contexts = evaluate_rag_pipeline(vector_store, eval_data)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        # Still print partial results if any
        if questions or answers:
            print_results(results, questions, answers, contexts)
        sys.exit(0)
    
    except Exception as e:
        print(f"\n⚠️  Error during pipeline execution: {type(e).__name__}")
        print(f"   {e}\n")
        
        # Print partial results if we have any
        if questions or answers:
            print("\n📌 Showing partial results (some queries may have failed)...")
            print_results(results, questions, answers, contexts)
        else:
            print("   No results available (error occurred early in pipeline)")
            import traceback
            traceback.print_exc()
        
        sys.exit(1)
    
    # Step 4: Print results
    print_results(results, questions, answers, contexts)
    
    print("\n" + "="*80)
    print(" ✅ Pipeline evaluation complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
