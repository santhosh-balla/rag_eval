# RAG Pipeline with Groq & RAGAs Evaluation

A baseline Retrieval-Augmented Generation (RAG) pipeline using Groq API, HuggingFace embeddings, and RAGAs for automated evaluation.

## 📋 Quick Summary

This project implements:

- **RAG Pipeline**: Retrieves relevant text chunks from domain documents and generates answers using Groq's Llama 3.1 8B
- **Vector Store**: Local Chroma database with HuggingFace embeddings (all-MiniLM-L6-v2) - free, local, no quota limits
- **Evaluation**: 5 gold-standard Q&A pairs evaluated with RAGAs metrics (Faithfulness, Answer Relevance, Context Precision, Context Recall)
- **Domains**: Java Textbook, Terms & Conditions, Honeywell Products Catalog

## 🚀 Quick Start

### Fast Mode (Recommended for Testing)
```bash
SKIP_METRICS=true python rag_pipeline.py
```
⏱️ **Takes ~30 seconds** - Tests RAG without slow metrics

### Full Mode (Includes All Metrics)
```bash
python rag_pipeline.py
```
⏱️ **Takes ~15 minutes** - Includes RAGAs evaluation metrics

**Python Version**: 3.9+

### 2. Configure API Keys

Create a `.env` file in the workspace root (copy from `.env.example`):

```bash
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
```

**Get API Keys**:

- **Groq API**: https://console.groq.com/keys
- **OpenAI API**: https://platform.openai.com/api-keys

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

- Ensure `.env` file exists with `GROQ_API_KEY` and `OPENAI_API_KEY`
- Check that keys are valid and have sufficient quota/credits

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

Replace `OpenAIEmbeddings()` with alternatives:

- HuggingFace Embeddings (free, local)
- Cohere Embeddings (commercial)
- Other LangChain-supported embeddings

## 📚 Additional Resources

- [LangChain Docs](https://python.langchain.com/)
- [Groq API Docs](https://console.groq.com/docs)
- [RAGAs Docs](https://docs.ragas.io/)
- [Chroma Docs](https://docs.trychroma.com/)

## 📝 License

This project is for educational purposes.

---

**Questions or issues?** Review the troubleshooting section or check the API documentation links above.
