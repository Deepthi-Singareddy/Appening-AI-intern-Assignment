import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from typing import List, Optional, Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import settings
from src.utils import logger, ensure_knowledge_base_pdf


# Global in-memory document store cache for local offline fallback
_LOCAL_INGESTED_CHUNKS: List[Document] = []


def load_and_split_pdf(pdf_path: Path) -> List[Document]:
    """
    Loads PDF document and splits text into standard chunks with overlap.
    """
    if not pdf_path.exists():
        pdf_path = ensure_knowledge_base_pdf()

    logger.info(f"Loading document from: {pdf_path}")
    docs: List[Document] = []
    
    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        logger.info(f"Successfully loaded {len(docs)} pages from PDF.")
    except Exception as e:
        logger.warning(f"PyPDFLoader encountered an issue: {e}. Checking text fallback...")
        txt_path = pdf_path.with_suffix(".txt")
        if txt_path.exists():
            loader = TextLoader(str(txt_path), encoding="utf-8")
            docs = loader.load()
            logger.info(f"Loaded text document fallback with {len(docs)} sections.")
        else:
            raise RuntimeError(f"Could not load PDF or fallback text file at {pdf_path}")

    # Chunking pipeline
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Segmented document into {len(chunks)} text chunks (size={settings.CHUNK_SIZE}, overlap={settings.CHUNK_OVERLAP}).")
    return chunks


def setup_pinecone_index(index_name: str, dimension: int = 1536):
    """
    Verifies or creates Pinecone index using ServerlessSpec.
    """
    if not settings.is_pinecone_configured:
        logger.info("Pinecone API key not set. Skipping Pinecone index creation.")
        return

    try:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            logger.info(f"Pinecone index '{index_name}' not found. Creating index with dimension={dimension}...")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_ENVIRONMENT)
            )
            logger.info(f"Successfully created Pinecone index '{index_name}'.")
        else:
            logger.info(f"Pinecone index '{index_name}' already exists.")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone index '{index_name}': {e}")


def run_ingestion(pdf_path: Optional[str] = None, index_name: Optional[str] = None) -> Any:
    """
    Executes complete ETL Ingestion & Vector Storage Pipeline:
    1. Parse source PDF document.
    2. Divide text into overlapping chunks.
    3. Generate OpenAI embeddings and index in Pinecone.
    4. Store fallback cache for local execution.
    """
    global _LOCAL_INGESTED_CHUNKS

    target_pdf = Path(pdf_path) if pdf_path else settings.DEFAULT_PDF_PATH
    target_index = index_name or settings.PINECONE_INDEX_NAME

    # 1. Load and chunk document
    chunks = load_and_split_pdf(target_pdf)
    _LOCAL_INGESTED_CHUNKS = chunks

    # 2. Pinecone & OpenAI Embeddings setup
    if settings.is_openai_configured and settings.is_pinecone_configured:
        try:
            from langchain_openai import OpenAIEmbeddings
            logger.info("Initializing OpenAI Embeddings (text-embedding-3-small)...")
            embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY
            )

            setup_pinecone_index(target_index, dimension=1536)

            try:
                from langchain_pinecone import PineconeVectorStore
                logger.info(f"Upserting {len(chunks)} vectors into Pinecone index '{target_index}' via PineconeVectorStore...")
                vector_store = PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    index_name=target_index,
                    pinecone_api_key=settings.PINECONE_API_KEY
                )
            except ImportError:
                from pinecone import Pinecone
                logger.info(f"Upserting {len(chunks)} vectors into Pinecone index '{target_index}' via native Pinecone SDK...")
                pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                index = pc.Index(target_index)
                vectors = []
                for i, chunk in enumerate(chunks):
                    vec = embeddings.embed_query(chunk.page_content)
                    vectors.append({
                        "id": f"chunk-{i}",
                        "values": vec,
                        "metadata": {"text": chunk.page_content}
                    })
                index.upsert(vectors=vectors)

            logger.info("Ingestion completed successfully in Pinecone Vector DB.")
            return chunks

        except Exception as e:
            logger.error(f"Error during Pinecone ingestion: {e}. Falling back to local chunk cache.")
            return chunks
    else:
        logger.warning(
            "API keys for OpenAI/Pinecone not fully configured in environment. "
            "Ingestion completed locally in memory for testing."
        )
        return chunks


if __name__ == "__main__":
    run_ingestion()
