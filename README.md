#  MedLens: Streamlit-Based Medical RAG System
Advanced Knowledge-Graph Enhanced Retrieval-Augmented Generation System for Clinical Documents

MedLens is an intelligent **Medical QA and RAG (Retrieval-Augmented Generation)** system designed to process clinical PDFs, extract key medical entities, and answer healthcare-related questions through semantic similarity and graph-based reasoning.

---

## 🧠 Core Overview
MedLens brings together **semantic embeddings**, **medical entity extraction**, and a **knowledge graph architecture** into a unified Streamlit application.  
It allows medical professionals, researchers, and students to upload medical documents (PDFs), automatically extract relevant entities (medications, dosages, and conditions), and query them in natural language.

---

## ⚙️ Key Features

###  Entity Extraction
- **Rule-based medical entity recognition** for medications, dosages, conditions, and frequency patterns.
- Extracts entities using domain-aware regex patterns.

###  Semantic Retrieval
- Utilizes **SentenceTransformers (`all-MiniLM-L6-v2`)** for sentence-level embeddings.
- Performs **vector similarity search** across chunked medical texts.

###  Knowledge Graph Integration
- Builds a **NetworkX-powered graph** connecting related entities by contextual co-occurrence.
- Enables **relationship-aware retrieval** for enriched answers.

###  Intelligent Question Answering
- Combines semantic similarity with entity relationships for context-aware responses.
- Returns related chunks, entity types, and graph connections.

###  Streamlit Interface
- Interactive single-page app for document upload, question input, and live result visualization.
- No backend/frontend separation — everything runs in one Python environment.

---

##  Architecture Overview

| Component | Description |
|------------|-------------|
| **MedicalRAG Class** | Core logic handling PDF ingestion, entity extraction, embedding generation, and graph building. |
| **SentenceTransformer Model** | Generates semantic embeddings for similarity search. |
| **NetworkX Graph** | Stores entities and their relationships for contextual query expansion. |
| **Streamlit UI** | Provides a user-friendly interface for uploads, querying, and visualization. |

---

##  Quick Start

### **1️⃣ Prerequisites**
- Python 3.8+
- Streamlit installed
- Git (optional)

---

### **2️⃣ Installation**
Clone the repository:
```bash
git clone https://github.com/zidan18Ahd/MedLens.git
cd MedLens/backend
