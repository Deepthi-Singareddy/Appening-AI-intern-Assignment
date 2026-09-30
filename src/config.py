import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure project root is in sys.path
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Load environment variables from .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Centralized System Configuration Management."""
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "3"))
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    DATA_DIR: Path = BASE_DIR / "data"
    DEFAULT_PDF_PATH: Path = DATA_DIR / "Ebook-Agentic-AI.pdf"
    
    @property
    def is_pinecone_configured(self) -> bool:
        return bool(
            self.PINECONE_API_KEY 
            and self.PINECONE_API_KEY.strip() != "" 
            and self.PINECONE_API_KEY != "your_pinecone_api_key_here"
        )
        
    @property
    def is_openai_configured(self) -> bool:
        return bool(
            self.OPENAI_API_KEY 
            and self.OPENAI_API_KEY.strip() != "" 
            and self.OPENAI_API_KEY != "your_openai_api_key_here"
        )


settings = Settings()
