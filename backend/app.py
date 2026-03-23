import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any
import networkx as nx

# Configure the page
st.set_page_config(
    page_title="Medical RAG System",
    page_icon="🩺",
    layout="wide"
)

@dataclass
class MedicalEntity:
    name: str
    type: str
    context: str

class MedicalRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = []
        self.knowledge_graph = nx.Graph()
    
    def extract_medical_entities(self, text: str) -> List[MedicalEntity]:
        """Simple rule-based medical entity extraction"""
        entities = []
        
        # Medical patterns
        patterns = {
            'MEDICATION': [
                r'\b(?:lisinopril|metformin|atorvastatin|amlodipine|metoprolol|omeprazole|simvastatin|losartan|albuterol|aspirin|ibuprofen|paracetamol|amoxicillin|vitamin)\b',
                r'\b[A-Z][a-z]*(?:statin|pril|mycin|cycline|pam|zolam|azole)\b'
            ],
            'DOSAGE': [
                r'\b\d+\s*mg\b',
                r'\b\d+\s*mcg\b', 
                r'\b\d+\s*ml\b',
                r'\b\d+\s*times?\s*(?:a|per)\s*(?:day|week|month)\b'
            ],
            'CONDITION': [
                r'\b(?:hypertension|diabetes|asthma|arthritis|depression|anxiety|migraine|headache|fever|cough|COVID|flu|pneumonia)\b'
            ],
            'FREQUENCY': [
                r'\b(?:once|twice|thrice)\s*(?:daily|a day)\b',
                r'\bevery\s*\d+\s*(?:hours|days|weeks)\b'
            ]
        }
        
        for entity_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    entity = MedicalEntity(
                        name=match.group(),
                        type=entity_type,
                        context=context
                    )
                    entities.append(entity)
        
        return entities
    
    def process_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chunk_text(self, text: str, chunk_size: int = 400) -> List[str]:
        """Split text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            current_chunk.append(word)
            current_size += len(word) + 1  # +1 for space
            
            if current_size >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """Process and add PDF document to the system"""
        # Extract text from PDF
        text = self.process_pdf(file_path)
        if text.startswith("Error:"):
            return {"error": text}
        
        # Chunk the text
        chunks = self.chunk_text(text)
        
        # Extract entities
        entities = self.extract_medical_entities(text)
        
        # Create embeddings for chunks
        for chunk in chunks:
            self.chunks.append(chunk)
            embedding = self.model.encode(chunk)
            self.embeddings.append(embedding)
        
        # Build knowledge graph
        for entity in entities:
            self.knowledge_graph.add_node(entity.name, type=entity.type)
        
        # Add some basic relationships based on co-occurrence
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:i+3]:  # Look at next 2 entities
                if entity1.name != entity2.name:
                    self.knowledge_graph.add_edge(entity1.name, entity2.name)
        
        return {
            "chunks_processed": len(chunks),
            "entities_found": len(entities),
            "text_length": len(text)
        }
    
    def search_similar(self, query: str, top_k: int = 5) -> List[str]:
        """Find similar chunks using vector search"""
        if not self.embeddings:
            return []
        
        query_embedding = self.model.encode(query)
        similarities = []
        
        for i, embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity and return top chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.chunks[idx] for idx, _ in similarities[:top_k]]
    
    def get_related_entities(self, query: str) -> List[Dict[str, str]]:
        """Get entities related to the query"""
        related = []
        query_lower = query.lower()
        
        for node in self.knowledge_graph.nodes():
            if query_lower in node.lower():
                # Get neighbors of this node
                neighbors = list(self.knowledge_graph.neighbors(node))
                related.append({
                    "entity": node,
                    "type": self.knowledge_graph.nodes[node].get('type', 'UNKNOWN'),
                    "neighbors": neighbors[:3]  # Top 3 neighbors
                })
        
        return related
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Main query function"""
        # Vector search for similar content
        similar_chunks = self.search_similar(question, top_k)
        
        # Entity search
        related_entities = self.get_related_entities(question)
        
        # Generate answer
        if similar_chunks:
            # Simple answer generation from context
            context = " ".join(similar_chunks[:2])
            answer = f"Based on the medical documents: {context[:300]}..."
        else:
            answer = "No relevant information found in the documents."
        
        return {
            "answer": answer,
            "relevant_chunks": similar_chunks,
            "entities": related_entities,
            "chunks_searched": len(similar_chunks)
        }

# Initialize the RAG system
if 'rag' not in st.session_state:
    st.session_state.rag = MedicalRAG()

st.title("🩺 Medical RAG System")
st.write("Upload medical PDFs and ask questions about medications, dosages, and conditions")

# Sidebar for uploads
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the PDF
        with st.spinner("Processing PDF..."):
            result = st.session_state.rag.add_document("temp.pdf")
        
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success(f"PDF processed! Found {result['entities_found']} medical entities across {result['chunks_processed']} text chunks.")
        
        # Clean up
        os.remove("temp.pdf")

# Main area for querying
st.header("Ask Medical Questions")

question = st.text_input("Enter your question about medications, conditions, etc:")

if st.button("Get Answer") and question:
    with st.spinner("Searching for answers..."):
        result = st.session_state.rag.query(question)
    
    st.subheader("Answer")
    st.write(result["answer"])
    
    if result["entities"]:
        st.subheader("Related Medical Entities")
        for entity in result["entities"]:
            with st.expander(f"{entity['entity']} ({entity['type']})"):
                st.write(f"Neighbors: {', '.join(entity['neighbors']) if entity['neighbors'] else 'None'}")
    
    if result["relevant_chunks"]:
        st.subheader("Relevant Text Chunks")
        for i, chunk in enumerate(result["relevant_chunks"]):
            st.text_area(f"Chunk {i+1}", chunk, height=100)

# System status
st.sidebar.header("System Status")
st.sidebar.write(f"Chunks loaded: {len(st.session_state.rag.chunks)}")
st.sidebar.write(f"Entities in graph: {st.session_state.rag.knowledge_graph.number_of_nodes()}")
st.sidebar.write(f"Relationships in graph: {st.session_state.rag.knowledge_graph.number_of_edges()}")
