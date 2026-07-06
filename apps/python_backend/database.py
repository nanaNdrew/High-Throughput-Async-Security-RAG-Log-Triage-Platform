from typing import AsyncGenerator
import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, func, text
from pgvector.sqlalchemy import Vector

from config import settings

logger = logging.getLogger(__name__)

# Strict connection pooling setup
engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30.0,
    pool_recycle=1800,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

class LogEntry(Base):
    __tablename__ = "log_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[DateTime] = mapped_column(DateTime(timezone=True), default=func.now(), index=True)
    severity: Mapped[str] = mapped_column(String(50), index=True)
    service: Mapped[str] = mapped_column(String(255), index=True)
    raw_text: Mapped[str] = mapped_column(Text)
    
    # tsvector column for native FTS, can be updated via trigger or application logic
    # Using Text for simplicity in this implementation but will be cast to tsvector in queries
    
    embeddings: Mapped[list["LogEmbedding"]] = relationship("LogEmbedding", back_populates="log_entry", cascade="all, delete-orphan")

class LogEmbedding(Base):
    __tablename__ = "log_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    log_id: Mapped[int] = mapped_column(ForeignKey("log_entries.id", ondelete="CASCADE"), index=True)
    chunk_text: Mapped[str] = mapped_column(Text)
    
    # Vector column using pgvector
    embedding = mapped_column(Vector(settings.embedding_dimensions))

    log_entry: Mapped["LogEntry"] = relationship("LogEntry", back_populates="embeddings")

async def init_db() -> None:
    """Initialize database schema, ensuring pgvector extension exists."""
    try:
        async with engine.begin() as conn:
            # Ensure pgvector extension exists
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create tables
            await conn.run_sync(Base.metadata.create_all)
            
            # Optional: Create an index on the vector column for faster search (ivfflat or hnsw)
            # await conn.execute(func.text("CREATE INDEX ON log_embeddings USING hnsw (embedding vector_cosine_ops);"))
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
