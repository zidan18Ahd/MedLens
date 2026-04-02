# MedLens: Medical RAG System

A knowledge-graph enhanced retrieval-augmented generation system for clinical documents.

MedLens is a Medical QA and RAG system designed to process clinical PDFs, extract 
key medical entities, and answer healthcare-related questions through fused semantic 
and graph-based retrieval.

---

## Core Overview

MedLens brings together semantic embeddings, medical entity extraction, a knowledge 
graph, and LLM-based answer generation into a unified Streamlit application.

It allows medical professionals, researchers, and students to upload medical documents 
(PDFs), automatically extract relevant entities (medications, dosages, conditions), 
and query them in natural language to receive synthesized answers.

---

## Key Features

### Entity Extraction
- Rule-based medical entity recognition for medications, dosages, conditions, and frequency patterns.
- Extracts entities using domain-aware regex patterns with 50-character context windows around each match.

### Semantic Retrieval
- Utilizes SentenceTransformers (all-MiniLM-L6-v2) for sentence-level embeddings.
- Performs cosine similarity search across chunked medical texts to retrieve top-k relevant passages.

### Knowledge Graph Integration
- Builds a NetworkX-powered graph connecting co-occurring medical entities as nodes and edges.
- Graph neighbor entities trigger a second retrieval pass, fusing semantic and graph signals 
  into a unified context before answer generation.

### LLM Answer Generation
- Retrieved context from both vector search and graph expansion is passed to Mistral via API.
- Produces fluent, grounded answers based strictly on document content.
- Falls back to raw context if the LLM call fails.

### Streamlit Interface
- Interactive single-page app for document upload, question input, and live result visualization.
- No backend/frontend separation — everything runs in one Python environment.

---

## Architecture Overview

| Component | Description |
|---|---|
| MedicalRAG Class | Core logic handling PDF ingestion, entity extraction, embedding generation, and graph building. |
| SentenceTransformer Model | Generates semantic embeddings for cosine similarity search. |
| NetworkX Graph | Stores entities and co-occurrence relationships; neighbors expand the retrieval context. |
| Mistral API | Reads fused context and generates a synthesized natural language answer. |
| Streamlit UI | Provides a user-friendly interface for uploads, querying, and visualization. |

---

## Query Pipeline
```
question
    |
vector search + graph neighbor expansion   <- fused retrieval
    |
merged, deduplicated context
    |
Mistral LLM (mistral-small-latest)         <- answer generation
    |
fluent answer grounded in document content
```

---

## Quick Start

### Prerequisites
- Python 3.8+
- Streamlit installed
- Mistral API key (free tier at console.mistral.ai)
- Git (optional)

### Installation

Clone the repository:
```bash
git clone https://github.com/zidan18Ahd/MedLens.git
cd MedLens/backend
```

### Configuration

Add your Mistral API key in medical_rag_light.py:
```python
"Authorization": "Bearer YOUR_MISTRAL_API_KEY"
```

---

## Known Limitations

- Entity extraction is regex-based — brand names, misspellings, and new drugs outside 
  the pattern list will be missed. A BioBERT or scispaCy model would improve recall significantly.
- Chunking has no overlap between windows — relationships split across chunk boundaries 
  may not be retrieved. Sliding window chunking is a planned improvement.
- Embeddings are stored in memory and reset on restart. FAISS or ChromaDB would add 
  persistence and scale to larger document sets.
- PDF ingestion only supports text-layer PDFs. Scanned documents require OCR (Tesseract) 
  as a fallback.
