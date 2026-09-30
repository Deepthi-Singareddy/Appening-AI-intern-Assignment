import os
from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.graph import build_rag_graph
from src.ingestion import run_ingestion
from src.utils import logger, ensure_knowledge_base_pdf

# Ensure knowledge base PDF exists on startup
ensure_knowledge_base_pdf()

# Pre-ingest document locally if needed
run_ingestion()

# Initialize FastAPI App
app = FastAPI(
    title="Agentic AI LangGraph & Pinecone RAG Chatbot API",
    description="Enterprise Retrieval-Augmented Generation API built with LangGraph, Pinecone, and OpenAI.",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LangGraph State Machine instance
graph = build_rag_graph(index_name=settings.PINECONE_INDEX_NAME)


class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "What is the core definition of Agentic AI as outlined in the eBook?"})


class QueryResponse(BaseModel):
    query: str = Field(..., description="Original query asked by user")
    final_answer: str = Field(..., description="Synthesized grounded answer")
    retrieved_context_chunks: List[str] = Field(..., description="Top retrieved context chunks from vector DB")
    confidence_score: float = Field(..., description="Groundedness / confidence score between 0.0 and 1.0")


class HealthResponse(BaseModel):
    status: str
    pinecone_configured: bool
    openai_configured: bool
    pdf_document_present: bool


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "LangGraph & Pinecone RAG Chatbot API is operational.",
        "docs_url": "/docs",
        "chat_endpoint": "/chat"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    return HealthResponse(
        status="healthy",
        pinecone_configured=settings.is_pinecone_configured,
        openai_configured=settings.is_openai_configured,
        pdf_document_present=settings.DEFAULT_PDF_PATH.exists()
    )


@app.post("/chat", response_model=QueryResponse, tags=["RAG Chatbot"])
async def chat_endpoint(request: QueryRequest):
    """
    Executes the LangGraph RAG workflow over the provided user query.
    Returns structured JSON with query, final answer, retrieved context chunks, and confidence score.
    """
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    logger.info(f"Received API query request: '{query_text}'")

    initial_state = {
        "question": query_text,
        "context": [],
        "answer": "",
        "score": 0.0,
        "is_grounded": False
    }

    try:
        result = graph.invoke(initial_state)
        
        response_payload = QueryResponse(
            query=query_text,
            final_answer=result.get("answer", "No answer generated."),
            retrieved_context_chunks=result.get("context", []),
            confidence_score=result.get("score", 0.0)
        )
        logger.info(f"Successfully processed query. Confidence Score: {response_payload.confidence_score}")
        return response_payload

    except Exception as e:
        logger.error(f"Error executing LangGraph RAG workflow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution error during RAG workflow: {str(e)}"
        )


@app.post("/ingest", tags=["ETL Pipeline"])
async def trigger_ingestion():
    """Triggers re-ingestion of the knowledge base PDF document."""
    try:
        run_ingestion()
        return {"status": "success", "message": "Knowledge base document ingested successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=settings.HOST, port=settings.PORT, reload=True)
