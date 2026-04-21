# RAG Pipeline with Groq & RAGAs Evaluation

A baseline Retrieval-Augmented Generation (RAG) pipeline using Groq API, local HuggingFace embeddings, and RAGAs for automated evaluation.

## 📋 Quick Summary

This project implements:

- **RAG Pipeline**: Retrieves relevant text chunks from domain documents and generates answers using Groq's Llama 3.1 70B
- **Vector Store**: Local Chroma database with local HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **Evaluation**: 5 gold-standard Q&A pairs evaluated with RAGAs metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall)
- **Domains**: Java Textbook, Terms & Conditions, Product Catalog

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Python Version**: 3.9+

### 2. Configure API Keys

Create a `.env` file in the workspace root (copy from `.env.example`):

```bash
GROQ_API_KEY=your_groq_api_key
```

**Get API Keys**:

- **Groq API**: https://console.groq.com/keys

No OpenAI key is required.

### 3. Verify Domain Files

The following files must be present:

- `java_textbook.txt` — Sample Java OOP concepts
- `terms_conditions.txt` — SoftwareHub T&C clauses
- `product_catalog.txt` — TechStore product listings

All files are pre-populated with realistic content.

## ▶️ Running the Pipeline

```bash
python rag_pipeline.py
```

### Expected Output

1. **Initialization**: Loads domain documents, creates Chroma vector store
2. **Evaluation**: Runs 5 gold-standard questions through the RAG pipeline
3. **Results**: Displays RAGAs metrics table and one sample Q&A with retrieved contexts

### Common Output Sections

```
📄 Loading domain documents...
✓ Loaded: java_textbook (X characters)
✓ Loaded: terms_conditions (X characters)
✓ Loaded: product_catalog (X characters)

🔧 Setting up Chroma vector store...
✂️  Splitting corpus into chunks...
✓ Created 123 chunks

💾 Creating Chroma vector store...
✓ Vector store created with 123 chunks

🧪 Evaluating RAG pipeline with RAGAs...
📊 Computing RAGAs metrics...
✓ Evaluation complete

📈 RAGAs Metrics Table:
   Faithfulness  Answer Relevance  Context Precision  Context Recall
0        0.8234            0.7912            0.8456           0.7890

📊 Average Score: 0.8123

💬 Generated Answer:
   [Full answer to first sample query]

📚 Retrieved Context Chunks:
   [Top 5 chunks for the sample query]
```

## 📁 File Structure

```
rag_eval/
├── requirements.txt               # Python dependencies
├── .env.example                   # API key template (rename to .env)
├── rag_pipeline.py               # Main RAG + evaluation script
├── java_textbook.txt             # Domain data: Java OOP
├── terms_conditions.txt          # Domain data: Software T&C
├── product_catalog.txt           # Domain data: Product descriptions
└── chroma_db/                    # Vector store (created on first run)
```

## 🔧 Configuration

### Modify Chunk Size

Edit `rag_pipeline.py`, function `create_chroma_store()`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Change this
    chunk_overlap=50,    # And/or this
)
```

### Change Groq Model

Edit the `get_rag_response()` function:

```python
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Or another Groq model
    temperature=0.3,
)
```

### Adjust Retrieval Parameters

Edit `get_rag_response()`:

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Change k to retrieve more/fewer chunks
```

## 📊 Understanding the Metrics

- **Faithfulness**: How well the answer is grounded in the retrieved context (avoids hallucination)
- **Answer Relevance**: How relevant the answer is to the asked question
- **Context Precision**: Among retrieved chunks, what fraction is relevant to the question
- **Context Recall**: Among all relevant chunks in the database, what fraction was retrieved

Higher scores (closer to 1.0) indicate better performance.

## ⚠️ Troubleshooting

### "API key not found" Error

- Ensure `.env` file exists with `GROQ_API_KEY`
- Check that keys are valid and have sufficient quota/credits

### Slow First Run

- The first run may download the local embedding model (`sentence-transformers/all-MiniLM-L6-v2`)
- Subsequent runs use the cached model

### "Domain file not found" Error

- Verify `java_textbook.txt`, `terms_conditions.txt`, and `product_catalog.txt` exist in the working directory

### Slow Evaluation

- Evaluation may take 2-5 minutes per query because RAGAs calls the LLM multiple times
- Groq API calls are processed sequentially to avoid rate limits

### Empty Vector Store

- Delete the `chroma_db/` folder and run the script again to rebuild

## 🔄 Reset & Rebuild

To clear and rebuild the vector store:

```bash
rm -r chroma_db/          # Or: rmdir /s chroma_db (Windows)
python rag_pipeline.py
```

## 📈 Extending the Pipeline

### Add More Evaluation Questions

Edit `get_evaluation_dataset()` in `rag_pipeline.py`:

```python
{
    "question": "Your question here?",
    "ground_truth": "Expected answer here",
    "context_domain": "attribute for tracking"
}
```

### Add New Domain Data

1. Create a new `.txt` file (e.g., `api_docs.txt`)
2. Add to `DOMAIN_FILES` dict in `rag_pipeline.py`
3. Run the script (it will auto-load the new domain)

### Switch Embeddings Provider

Replace `HuggingFaceEmbeddings()` with another LangChain-supported embeddings provider if needed.

## 📚 Additional Resources

- [LangChain Docs](https://python.langchain.com/)
- [Groq API Docs](https://console.groq.com/docs)
- [RAGAs Docs](https://docs.ragas.io/)
- [Chroma Docs](https://docs.trychroma.com/)

## 📝 License

This project is for educational purposes.

---

**Questions or issues?** Review the troubleshooting section or check the API documentation links above.

---

## Week 2: LLM-as-a-Judge Evaluation

Week 2 adds a second evaluation layer on top of the untouched Week 1 baseline: an LLM-as-a-judge scores each RAG output on a 4-metric rubric, a consistency test measures robustness to paraphrases, and a comparison pipeline cross-checks the judge against the Week 1 RAGAs scores.

### How to run

Place the Week 1 artifacts in a folder named `week1_artifacts/` at the repo root:

```
week1_artifacts/
├── rag_outputs.json                  # 10 entries: question, ground_truth, context_domain, rag_answer, retrieved_contexts
├── ragas_per_question_scores.csv     # per-question RAGAs scores, keyed by question
└── ragas_domain_scorecard.csv        # Week 1 domain scorecard (kept for reference)
```

Then from the repo root:

```bash
pip install -r requirements.txt
python run_week2_pipeline.py
```

`run_week2_pipeline.py` is the single entry point. It runs four phases in order and prints timing for each one.

### Rubric and judge model

The judge scores each RAG output on four metrics using an integer Likert scale from 1 to 5: **Faithfulness** (every claim supported by the retrieved context), **Answer Relevance** (directly addresses the question), **Correctness** (matches the ground truth semantically), and **Context Relevance** (retrieved chunks are on-topic). Each metric includes explicit anchor descriptions at 1 / 3 / 5 so scores are comparable across runs, and the judge must also output a one-sentence `reasoning` field justifying the lowest score. The judge model is **`llama-3.1-8b-instant`** (constant `JUDGE_MODEL` in `llm_judge.py`), deliberately different from the Week 1 generator (`llama-3.3-70b-versatile`) to reduce self-evaluation bias. All judge calls run at `temperature=0` and require strict JSON output; parse failures are retried once with a stricter reminder and record null scores on persistent failure rather than crashing the pipeline.

### Scale note

Judge scores are on a 1-5 integer scale. When joined against RAGAs (which returns 0-1 floats), the judge scores are also presented in normalized form (`judge_*_norm = judge_* / 5`) so deltas and correlations are meaningful. Both raw and normalized columns are kept in the comparison CSV.

### Produced files

- `llm_judge_outputs.json` — full per-question judge records (scores, reasoning, raw response).
- `llm_judge_scores.csv` — flat table (`question, domain, faithfulness, answer_relevance, correctness, context_relevance`), keyed by question for joining with Week 1 RAGAs scores.
- `consistency_test_results.csv` — all 18 paraphrase runs (3 base questions × 6 variants): question text, RAG answer, and 4 judge scores per row.
- `consistency_summary.csv` — per base question, per metric: mean and std across the 6 variants.
- `week2_comparison.csv` — per-question side-by-side: `ragas_faithfulness`, `judge_faithfulness_norm`, `delta_faithfulness`, `ragas_context_precision`, `judge_context_relevance_norm`, `delta_context`, `ragas_context_recall` (reference), `judge_answer_relevance_norm`, `judge_correctness_norm`.
- `week2_domain_scorecard.csv` — per-domain means of the six comparable metrics above.
- `correlations.csv` — Pearson and Spearman `r` and `p` for `ragas_faithfulness` vs `judge_faithfulness_norm` and for `ragas_context_precision` vs `judge_context_relevance_norm`.
- `divergence_cases.csv` — top 3 questions where the judge and RAGAs disagree most (|delta| ≥ 0.3 on the normalized scale), including the full RAG answer and the judge's reasoning.
- `figures/fig_ragas_vs_judge_scatter.png` — scatter of RAGAs vs judge faithfulness with y=x reference, colored by domain.
- `figures/fig_domain_comparison_bar.png` — grouped bar chart of per-domain faithfulness means (RAGAs vs judge).
- `figures/fig_consistency_heatmap.png` — heatmap of std across 6 variants (lower = more consistent) per base question × metric.

### Files added in Week 2 (for reference)

- `llm_judge.py` — judge module (prompt, `score_one`, `score_batch`, model constant).
- `run_judge_on_week1.py` — runs the judge over the 10 Week 1 RAG outputs.
- `consistency_test.py` — 3 base questions × 6 paraphrase variants, uses `get_rag_response` + `get_or_create_chroma_store` from `rag_pipeline.py`.
- `compare_ragas_vs_judge.py` — joins on `question`, normalizes judge scores, computes deltas, correlations, and divergences.
- `visualize_week2.py` — matplotlib figures (dpi=150, `tight_layout`).
- `run_week2_pipeline.py` — orchestrator.

