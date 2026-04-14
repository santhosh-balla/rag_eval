"""
RAG Pipeline with Groq & RAGAs Evaluation
==========================================
This script builds a baseline RAG (Retrieval-Augmented Generation) pipeline using:
- Groq API (llama-3.1-70b-versatile) for LLM inference
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall
from datasets import Dataset

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GROQ_API_KEY or not OPENAI_API_KEY:
    print("❌ ERROR: API keys not found!")
    print("Please create a .env file with GROQ_API_KEY and OPENAI_API_KEY variables.")
    sys.exit(1)

# Paths to domain text files
DOMAIN_FILES = {
    "java_textbook": "java_textbook.txt",
    "terms_conditions": "terms_conditions.txt",
    "product_catalog": "product_catalog.txt",
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
    
    # Initialize OpenAI embeddings
    print("\n🔐 Initializing OpenAI embeddings (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
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
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
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
        model="llama-3.1-70b-versatile",
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
    retrieved_docs = retriever.get_relevant_documents(query)
    
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
            "question": "What is inheritance in Java and how does it work?",
            "ground_truth": "Inheritance is a mechanism where a new class (derived class) inherits properties and behaviors from an existing class (base class) using the 'extends' keyword. In Java, a class can directly inherit from only one base class but can implement multiple interfaces. The derived class inherits all non-private members of the base class.",
            "context_domain": "java_textbook"
        },
        {
            "question": "What are the limitations and disclaimers mentioned in the Terms and Conditions?",
            "ground_truth": "SoftwareHub provides materials 'as is' without warranties. They disclaim all warranties including merchantability, fitness for a particular purpose, and non-infringement. They are not liable for damages including loss of data or profit, even if notified of possible damage. They do not warrant accuracy, completeness, or currentness of materials.",
            "context_domain": "terms_conditions"
        },
        {
            "question": "What are the key features offered by CloudHost Business Plan?",
            "ground_truth": "CloudHost Business Plan includes 500 GB SSD storage, unlimited bandwidth, 24/7 monitoring and incident response, database management for MySQL, PostgreSQL, and MongoDB, and API rate of 1 million requests per month included. It offers 99.99% uptime SLA with automatic load balancing and auto-scaling.",
            "context_domain": "product_catalog"
        },
        {
            "question": "What access modifiers exist in Java and what do they control?",
            "ground_truth": "Java provides four access levels: public (accessible from anywhere), protected (accessible within same package and subclasses), default/no modifier (accessible only within same package), and private (accessible only within the same class). These modifiers control the visibility and accessibility of classes, methods, and variables.",
            "context_domain": "java_textbook"
        },
        {
            "question": "What is the price of CodeMaster Pro 2024 and what platforms does it support?",
            "ground_truth": "CodeMaster Pro 2024 is priced at $99.99 per year. It supports Windows 10+, macOS 10.14+, and Linux (Ubuntu 18.04+). It includes features like AI-powered code completion, cross-platform debugging tools, version control integration, and a plugin ecosystem with 500+ community extensions.",
            "context_domain": "product_catalog"
        },
    ]
    
    return evaluation_data


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
        model="llama-3.1-70b-versatile",
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
    
    # Configure RAGAs metrics with Groq judge
    print("\n📊 Computing RAGAs metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall)...")
    
    # Create metric instances configured with Groq
    # Note: RAGAs metrics can be configured with custom LLMs
    metrics_to_evaluate = [
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall
    ]
    
    try:
        # Run evaluation
        results = evaluate(
            dataset=rag_eval_dataset,
            metrics=metrics_to_evaluate,
            llm=judge_llm,
            embeddings=OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=OPENAI_API_KEY
            ),
            batch_size=1  # Process one at a time to avoid rate limits
        )
        
        print("✓ Evaluation complete")
        return results, questions, answers, contexts
    
    except Exception as e:
        print(f"⚠️  Error during RAGAs evaluation: {e}")
        print("Attempting fallback evaluation...")
        return None, questions, answers, contexts


# ============================================================================
# PHASE 5: OUTPUT FORMATTING
# ============================================================================

def print_results(results: Dict, questions: List[str], answers: List[str], contexts: List[List[str]]):
    """
    Print formatted evaluation results and sample Q&A.
    
    Args:
        results: RAGAs evaluation results dictionary.
        questions: List of questions evaluated.
        answers: List of generated answers.
        contexts: List of retrieved context strings for each query.
    """
    print("\n" + "="*80)
    print("RAG PIPELINE EVALUATION RESULTS")
    print("="*80)
    
    if results:
        # Extract metric scores
        try:
            metrics_dict = {
                "Faithfulness": results.get("faithfulness", 0),
                "Answer Relevance": results.get("answer_relevance", 0),
                "Context Precision": results.get("context_precision", 0),
                "Context Recall": results.get("context_recall", 0)
            }
            
            # Handle case where results might be aggregated differently
            if isinstance(results, dict) and "scores" in results:
                metrics_dict = results["scores"]
            
            # Create metrics table
            metrics_df = pd.DataFrame([metrics_dict])
            
            print("\n📈 RAGAs Metrics Table:")
            print("-" * 80)
            print(metrics_df.to_string())
            print("-" * 80)
            
            # Calculate and print average score
            avg_score = np.mean(list(metrics_dict.values()))
            print(f"\n📊 Average Score: {avg_score:.4f}")
        
        except Exception as e:
            print(f"⚠️  Could not format metrics table: {e}")
            print(f"Raw results: {results}")
    else:
        print("⚠️  No evaluation results available (check API keys and rate limits)")
    
    # Print sample query with answer and contexts
    if questions and answers and contexts:
        print("\n" + "="*80)
        print("SAMPLE QUERY DEMO")
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main orchestration function."""
    print("\n" + "="*80)
    print(" RAG PIPELINE WITH GROQ & RAGAs EVALUATION")
    print("="*80)
    
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
        
        # Step 4: Print results
        print_results(results, questions, answers, contexts)
        
        print("\n" + "="*80)
        print(" ✅ Pipeline evaluation complete!")
        print("="*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Process interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
