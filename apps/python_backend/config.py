from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    database_url: str = Field(
        default="postgresql+asyncpg://admin:password@localhost:5432/log_triage",
        description="Database connection string"
    )
    
    # Worker Settings
    worker_concurrency: int = Field(
        default=4,
        description="Number of concurrent background workers pulling from the queue"
    )
    worker_batch_size: int = Field(
        default=50,
        description="Maximum number of logs to batch process together"
    )
    worker_batch_timeout: float = Field(
        default=0.1,
        description="Timeout in seconds to wait for accumulating a batch"
    )
    
    # Chunking Settings
    chunk_size: int = Field(
        default=500,
        description="Fixed-size character chunking size"
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap between chunks"
    )
    
    # Model Settings
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformers model for embeddings"
    )
    embedding_dimensions: int = Field(
        default=384,
        description="Dimensions for the selected embedding model"
    )
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
