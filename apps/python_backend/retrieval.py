import asyncio
import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from config import settings
from database import AsyncSessionLocal
from worker import model  # reuse the loaded model

logger = logging.getLogger(__name__)

# Severity weights for heuristic scoring
SEVERITY_WEIGHTS = {
    "CRITICAL": 1.5,
    "ERROR": 1.2,
    "WARNING": 1.0,
    "INFO": 0.8,
    "DEBUG": 0.5
}

def calculate_time_decay(log_timestamp: datetime, current_time: datetime = None) -> float:
    """Calculates a time decay factor based on hours elapsed."""
    if not current_time:
        current_time = datetime.now(timezone.utc)
    
    # Ensure log_timestamp is aware
    if log_timestamp.tzinfo is None:
        log_timestamp = log_timestamp.replace(tzinfo=timezone.utc)
        
    delta_hours = (current_time - log_timestamp).total_seconds() / 3600.0
    
    # Simple decay: 1 / (1 + decay_rate * hours)
    # This means a log from 24h ago will have a lower score multiplier
    decay_rate = 0.05
    decay_factor = 1.0 / (1.0 + decay_rate * max(0, delta_hours))
    return decay_factor

def apply_heuristic_reranking(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Applies programmatic re-ranking using time-decay and severity heuristics."""
    reranked = []
    current_time = datetime.now(timezone.utc)
    
    for res in results:
        base_score = float(res.get("rrf_score", 0.0) or 0.0)
        severity = res.get("severity", "INFO").upper()
        timestamp = res.get("timestamp")
        
        # Apply severity multiplier
        sev_multiplier = SEVERITY_WEIGHTS.get(severity, 1.0)
        
        # Apply time decay
        time_multiplier = 1.0
        if timestamp:
            time_multiplier = calculate_time_decay(timestamp, current_time)
            
        final_score = base_score * sev_multiplier * time_multiplier
        
        # Create a new dict with the final score
        new_res = res.copy()
        new_res["final_score"] = final_score
        reranked.append(new_res)
        
    # Sort by final score descending
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked

async def execute_hybrid_search(
    query_text: str, 
    limit: int = 10,
    k: int = 60  # RRF constant
) -> List[Dict[str, Any]]:
    """
    Executes a hybrid search combining native Postgres FTS (tsvector) with pgvector cosine distance,
    merged via Reciprocal Rank Fusion (RRF) at the database or application level.
    For this implementation, we merge them using an advanced CTE query in Postgres.
    """
    # 1. Generate query embedding
    query_embedding = await asyncio.to_thread(model.encode, query_text)
    embedding_list = query_embedding.tolist()
    
    async with AsyncSessionLocal() as session:
        # 2. Execute Hybrid Search Query with RRF directly in SQL
        # This is highly optimized as it avoids moving large amounts of data to Python
        sql_query = text("""
        WITH semantic_search AS (
            SELECT 
                le.id AS embed_id,
                le.log_id,
                le.chunk_text,
                le.embedding <=> CAST(:query_embedding AS vector) AS vector_distance,
                RANK() OVER (ORDER BY le.embedding <=> CAST(:query_embedding AS vector)) AS vector_rank
            FROM log_embeddings le
            ORDER BY vector_distance
            LIMIT :sub_limit
        ),
        keyword_search AS (
            SELECT 
                le.id AS embed_id,
                le.log_id,
                le.chunk_text,
                ts_rank(to_tsvector('english', le.chunk_text), plainto_tsquery('english', :query_text)) AS fts_score,
                RANK() OVER (ORDER BY ts_rank(to_tsvector('english', le.chunk_text), plainto_tsquery('english', :query_text)) DESC) AS fts_rank
            FROM log_embeddings le
            WHERE to_tsvector('english', le.chunk_text) @@ plainto_tsquery('english', :query_text)
            ORDER BY fts_score DESC
            LIMIT :sub_limit
        ),
        rrf_results AS (
            SELECT
                COALESCE(ss.embed_id, ks.embed_id) AS embed_id,
                COALESCE(ss.log_id, ks.log_id) AS log_id,
                COALESCE(ss.chunk_text, ks.chunk_text) AS chunk_text,
                COALESCE(1.0 / (:rrf_k + ss.vector_rank), 0.0) + 
                COALESCE(1.0 / (:rrf_k + ks.fts_rank), 0.0) AS rrf_score
            FROM semantic_search ss
            FULL OUTER JOIN keyword_search ks ON ss.embed_id = ks.embed_id
        )
        SELECT 
            r.embed_id,
            r.log_id,
            r.chunk_text,
            r.rrf_score,
            l.timestamp,
            l.severity,
            l.service,
            l.raw_text
        FROM rrf_results r
        JOIN log_entries l ON r.log_id = l.id
        ORDER BY r.rrf_score DESC
        LIMIT :final_limit;
        """)
        
        result = await session.execute(
            sql_query, 
            {
                "query_embedding": str(embedding_list), 
                "query_text": query_text,
                "sub_limit": limit * 2,  # Fetch more for RRF
                "rrf_k": k,
                "final_limit": limit
            }
        )
        
        rows = result.mappings().all()
        
        # Convert to list of dicts
        raw_results = [dict(row) for row in rows]
        
        # 3. Apply programmatic re-ranking
        final_results = apply_heuristic_reranking(raw_results)
        
        return final_results
