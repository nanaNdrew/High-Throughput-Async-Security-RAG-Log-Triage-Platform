import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import uuid

from fastapi import FastAPI, BackgroundTasks, HTTPException, status, Depends
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import os

from database import init_db, get_db, LogEntry
from worker import worker_pool
from retrieval import execute_hybrid_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Log Triage API...")
    
    # Initialize DB (creates tables and pgvector extension)
    await init_db()
    
    # Start worker pool
    await worker_pool.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Log Triage API...")
    await worker_pool.stop()

app = FastAPI(
    title="Log Triage & Security RAG Analyst API",
    description="High-throughput event-driven API for log ingestion and intelligent RAG queries.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Models ---

class LogPayload(BaseModel):
    timestamp: Optional[str] = Field(None, description="ISO8601 timestamp")
    severity: str = Field(default="INFO", description="Log severity (e.g., INFO, WARNING, ERROR, CRITICAL)")
    service: str = Field(default="unknown", description="Source service of the log")
    raw_text: str = Field(..., description="The actual raw log content")

class LogStreamRequest(BaseModel):
    logs: List[LogPayload]

class AnalystQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language security query")
    limit: int = Field(10, description="Max results to retrieve")

class AnalystQueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    root_cause_analysis: str

# --- Endpoints ---

@app.post("/api/v1/logs/stream", status_code=status.HTTP_202_ACCEPTED)
async def ingest_logs_stream(request: LogStreamRequest):
    """
    High-throughput endpoint that drops incoming JSON log streams into the asyncio.Queue
    non-blockingly and returns an ingestion task ID.
    """
    task_id = str(uuid.uuid4())
    
    # Enqueue each log payload
    for log in request.logs:
        await worker_pool.enqueue(log.model_dump())
        
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": f"Queued {len(request.logs)} logs for asynchronous processing."
    }

def mock_llm_root_cause_analysis(query: str, contexts: List[Dict[str, Any]]) -> str:
    """Mocks an LLM deterministic root-cause analysis generation."""
    if not contexts:
        return "No relevant context found to determine root cause."
        
    # Analyze the top context which has the highest final score
    top_context = contexts[0]
    severity = top_context.get("severity", "UNKNOWN")
    service = top_context.get("service", "UNKNOWN")
    text_snippet = top_context.get("raw_text", "")[:100]
    
    return (
        f"Based on the query '{query}', the primary root cause appears to be in the '{service}' service "
        f"with a severity of '{severity}'. The most critical evidence points to: '{text_snippet}...'. "
        f"Recommendation: Investigate the '{service}' logs around the reported timestamp."
    )

@app.post("/api/v1/analyst/query", response_model=AnalystQueryResponse)
async def analyst_query(request: AnalystQueryRequest):
    """
    Evaluates natural language security queries using the hybrid retrieval pipeline 
    and feeds the dense context to a mocked LLM completion call to generate a deterministic 
    root-cause analysis report.
    """
    try:
        # Execute the RAG retrieval step
        contexts = await execute_hybrid_search(
            query_text=request.query,
            limit=request.limit
        )
        
        # Generate the analysis
        analysis = mock_llm_root_cause_analysis(request.query, contexts)
        
        return AnalystQueryResponse(
            query=request.query,
            results=contexts,
            root_cause_analysis=analysis
        )
        
    except Exception as e:
        logger.error(f"Error processing analyst query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during query processing")

@app.get("/health")
async def health_check():
    return {"status": "ok", "queue_size": worker_pool.queue.qsize()}

@app.get("/api/v1/logs")
async def get_latest_logs(
    limit: int = 50,
    severity: Optional[str] = None,
    service: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves the latest logs from the database, filterable by severity and service.
    """
    try:
        stmt = select(LogEntry).order_by(LogEntry.timestamp.desc())
        if severity:
            stmt = stmt.where(LogEntry.severity == severity)
        if service:
            stmt = stmt.where(LogEntry.service == service)
        stmt = stmt.limit(limit)
        
        result = await db.execute(stmt)
        entries = result.scalars().all()
        
        return [
            {
                "id": entry.id,
                "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
                "severity": entry.severity,
                "service": entry.service,
                "raw_text": entry.raw_text
            }
            for entry in entries
        ]
    except Exception as e:
        logger.error(f"Error fetching logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching logs from database")

# Mount the static directory for the frontend dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    index_path = os.path.join(static_dir, "index.html")
    if not os.path.exists(index_path):
        from fastapi.responses import HTMLResponse
        return HTMLResponse("<html><body>Dashboard frontend is loading... Refresh in a second.</body></html>")
    return FileResponse(index_path)
