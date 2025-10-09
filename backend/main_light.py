from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from medical_rag_light import MedicalLightRAG
import asyncio

app = FastAPI(title="Medical Light RAG", version="1.0.0")

# CORS middleware - important for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize our RAG system
rag_system = MedicalLightRAG(working_dir="./data")

# Create upload directory
os.makedirs("data/uploads", exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Medical Light RAG System - PDF Only"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "chunks_loaded": len(rag_system.chunks),
        "entities_found": rag_system.knowledge_graph.number_of_nodes()
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Save uploaded file
        file_path = f"data/uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            # Copy the uploaded file to the destination
            shutil.copyfileobj(file.file, buffer)
        
        # Process the PDF
        result = await rag_system.add_document(file_path, file.filename)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content={
            "status": "success", 
            "message": "PDF processed successfully",
            "result": result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/query")
async def query_medical_data(question: str = Form(...)):
    """Query the medical RAG system"""
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = await rag_system.query(question)
        
        return JSONResponse(content={
            "status": "success",
            "question": question,
            "result": result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_chunks": len(rag_system.chunks),
        "total_entities": rag_system.knowledge_graph.number_of_nodes(),
        "total_relationships": rag_system.knowledge_graph.number_of_edges()
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting Medical Light RAG Server...")
    print("Access the API at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)