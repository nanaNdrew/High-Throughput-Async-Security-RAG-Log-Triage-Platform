import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone

from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import AsyncSessionLocal, LogEntry, LogEmbedding

logger = logging.getLogger(__name__)

# Load embedding model once per process
try:
    model = SentenceTransformer(settings.embedding_model_name)
    logger.info(f"Loaded SentenceTransformer model: {settings.embedding_model_name}")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Splits text into overlapping chunks of fixed size."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - chunk_overlap
    return chunks

def format_context_header(log_data: Dict[str, Any]) -> str:
    """Formats a contextual header for the chunk."""
    timestamp = log_data.get("timestamp", datetime.now(timezone.utc).isoformat())
    service = log_data.get("service", "unknown")
    severity = log_data.get("severity", "INFO")
    return f"[Timestamp: {timestamp} | Service: {service} | Severity: {severity}] "

async def process_log_batch(logs: List[Dict[str, Any]]):
    """Processes a batch of logs: chunks them, embeds them, and batch inserts to PostgreSQL."""
    async with AsyncSessionLocal() as session:
        try:
            # 1. Create all LogEntry objects
            entries_and_data = []
            for log_data in logs:
                timestamp_str = log_data.get("timestamp")
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.now(timezone.utc)
                    
                log_entry = LogEntry(
                    timestamp=timestamp,
                    severity=log_data.get("severity", "INFO"),
                    service=log_data.get("service", "unknown"),
                    raw_text=log_data.get("raw_text", "")
                )
                session.add(log_entry)
                entries_and_data.append((log_entry, log_data))
            
            # Flush once to populate generated IDs for all entries in the batch
            await session.flush()
            
            # 2. Chunking & preparing all texts to embed
            all_chunks_metadata = []
            all_chunk_texts = []
            
            for log_entry, log_data in entries_and_data:
                header = format_context_header(log_data)
                raw_text = log_data.get("raw_text", "")
                chunks = chunk_text(raw_text, settings.chunk_size, settings.chunk_overlap)
                
                for chunk in chunks:
                    chunk_text_val = f"{header}{chunk}"
                    all_chunks_metadata.append((log_entry.id, chunk_text_val))
                    all_chunk_texts.append(chunk_text_val)
            
            # 3. Calculate embeddings in one batch
            if all_chunk_texts:
                # Encode all chunks in a single call to leverage parallelized tensor/matrix computations
                embeddings = await asyncio.to_thread(model.encode, all_chunk_texts)
                
                # 4. Create all LogEmbedding entries
                for (log_id, chunk_text_val), embedding in zip(all_chunks_metadata, embeddings):
                    log_embedding = LogEmbedding(
                        log_id=log_id,
                        chunk_text=chunk_text_val,
                        embedding=embedding.tolist()  # pgvector expects a list
                    )
                    session.add(log_embedding)
            
            # Commit the batch transaction
            await session.commit()
            logger.debug(f"Successfully processed and inserted batch of {len(logs)} logs.")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Error processing log batch: {e}", exc_info=True)

async def worker_loop(queue: asyncio.Queue, worker_id: int):
    """Background worker loop pulling logs from the queue and processing them in batches."""
    logger.info(f"Worker {worker_id} started.")
    
    batch_size = settings.worker_batch_size
    batch_timeout = settings.worker_batch_timeout
    
    while True:
        try:
            # 1. Wait for at least one log entry to arrive
            log_item = await queue.get()
            batch = [log_item]
            
            # 2. Try to drain the queue up to batch_size, or wait until batch_timeout expires
            start_time = asyncio.get_running_loop().time()
            while len(batch) < batch_size:
                # First try to get any log entries that are immediately available
                try:
                    item = queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    # If the queue is empty, wait for the next item up to the remaining timeout
                    elapsed = asyncio.get_running_loop().time() - start_time
                    remaining = batch_timeout - elapsed
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=remaining)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
            
            # 3. Process the accumulated batch
            try:
                await process_log_batch(batch)
            finally:
                # Mark all processed tasks as done
                for _ in range(len(batch)):
                    queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} shutting down.")
            break
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered an error: {e}", exc_info=True)

class WorkerPool:
    def __init__(self, num_workers: int = settings.worker_concurrency):
        self.queue = asyncio.Queue()
        self.num_workers = num_workers
        self.workers = []
        
    async def start(self):
        """Starts the worker pool."""
        self.workers = [
            asyncio.create_task(worker_loop(self.queue, i))
            for i in range(self.num_workers)
        ]
        logger.info(f"Started {self.num_workers} background workers.")
        
    async def stop(self):
        """Stops the worker pool gracefully."""
        # Wait until the queue is fully processed
        await self.queue.join()
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait until all workers finish cancellation
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Worker pool stopped.")
        
    async def enqueue(self, log_data: Dict[str, Any]):
        """Non-blockingly enqueues a log payload."""
        self.queue.put_nowait(log_data)
        logger.debug(f"Log enqueued. Queue size: {self.queue.qsize()}")

# Global worker pool instance
worker_pool = WorkerPool()
