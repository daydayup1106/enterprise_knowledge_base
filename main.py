"""
main.py
=======
FastAPI application entry point for the Enterprise Knowledge Base Expert Collaborative System.

Startup (lifespan):
  1. Loads local Embedding + Reranker models into memory.
  2. Builds the LlamaIndex RAG pipeline (VectorIndex + BM25 + Reranker).
  3. Compiles the LangGraph multi-agent state machine.
  4. Verifies Redis connectivity.
  All of the above happens BEFORE the server accepts any requests.

Endpoints:
  GET  /                    → Serves the static HTML chat UI
  GET  /api/v1/health       → Returns system readiness status
  POST /api/v1/chat         → Main conversational endpoint

Note on route path:
  The requirements spec mentions /chat; we use /api/v1/chat with a versioned
  prefix for maintainability. The frontend index.html calls this same path.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from core.config import get_settings
from core.rag_engine import build_rag_engine
from core.graph import build_graph

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup preload (all heavy work done here, not per-request)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    STARTUP:  Load models, build index, compile graph → store in app.state.
    SHUTDOWN: Mark index as not ready (graceful signal).
    """
    logger.info("=" * 60)
    logger.info("Enterprise Knowledge Base System — STARTING UP")
    logger.info("=" * 60)

    app.state.index_ready = False
    app.state.startup_failed = False
    app.state.rag_engine = None
    app.state.langgraph_app = None

    try:
        # Step 1: Build RAG engine (loads embedding model, reranker, builds index)
        logger.info("[1/2] Building RAG engine (model loading + indexing)...")
        rag_engine = build_rag_engine()
        app.state.rag_engine = rag_engine

        # Step 2: Build LangGraph compiled app (connects to Redis)
        logger.info("[2/2] Building LangGraph state machine...")
        lg_app = build_graph(rag_engine)
        app.state.langgraph_app = lg_app

        app.state.index_ready = True
        logger.info("=" * 60)
        logger.info(f"System READY — {rag_engine.chunk_count} knowledge chunks indexed.")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        app.state.startup_failed = True
        logger.error(f"STARTUP FAILED — knowledge base document missing: {e}")
        # Do not raise — let server start so /health can report the error
    except ConnectionError as e:
        app.state.startup_failed = True
        logger.error(f"STARTUP FAILED — Redis unavailable: {e}")
    except Exception as e:
        app.state.startup_failed = True
        logger.error(f"STARTUP FAILED — unexpected error: {e}", exc_info=True)

    yield  # ← server is alive here, handling requests

    # Shutdown
    app.state.index_ready = False
    logger.info("Enterprise Knowledge Base System — SHUT DOWN")


# ─────────────────────────────────────────────────────────────────────────────
# App & Templates
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="企业知识库专家协同系统",
    description="Multi-agent RAG system powered by DeepSeek + LlamaIndex + LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    thread_id: str = Field(
        ...,
        description="Session UUID. Frontend must generate and persist this per-session.",
        min_length=1,
        max_length=128,
    )
    message: str = Field(
        ...,
        description="User message text. 1–2000 characters.",
        min_length=1,
        max_length=2000,
    )

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be blank or whitespace only.")
        return v.strip()

    @field_validator("thread_id")
    @classmethod
    def thread_id_strip(cls, v: str) -> str:
        return v.strip()


class ChatMetadata(BaseModel):
    used_rag: bool
    error_msg: Optional[str] = None


class ChatResponse(BaseModel):
    status: str          # "success" | "error"
    thread_id: str
    reply: str
    metadata: ChatMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, summary="Chat UI")
async def serve_ui(request: Request):
    """
    Serves the static HTML/JS chat interface.
    Boundary: Returns 500 with error detail if template is missing.
    """
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Failed to render index.html: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="UI template unavailable.")


@app.get("/api/v1/health", summary="System Health Check")
async def health_check(request: Request):
    """
    Returns the current readiness of the system.
    Frontend should poll this on load and only enable the chat input when
    status == 'ready'.

    Returns:
        200 always (so frontend can distinguish partial failures).
        status: "ready" | "indexing" | "error"
    """
    rag_engine = request.app.state.rag_engine
    is_ready = request.app.state.index_ready

    if is_ready and rag_engine is not None:
        return {
            "status": "ready",
            "index_loaded": True,
            "doc_chunk_count": rag_engine.chunk_count,
        }
    elif getattr(request.app.state, "startup_failed", False):
        return {
            "status": "error",
            "index_loaded": False,
            "doc_chunk_count": 0,
        }
    elif rag_engine is None:
        return {
            "status": "indexing",
            "index_loaded": False,
            "doc_chunk_count": 0,
        }
    else:
        return {
            "status": "error",
            "index_loaded": False,
            "doc_chunk_count": 0,
        }


@app.post("/api/v1/chat", response_model=ChatResponse, summary="Chat with Agents")
async def chat(request: Request, body: ChatRequest):
    """
    Main conversational endpoint.

    Calls the LangGraph multi-agent pipeline with the given thread_id (session key).
    LangGraph automatically restores conversation history from Redis for that thread_id,
    then appends the new HumanMessage and runs the agent flow.

    Boundary Conditions:
      - 503 if system is still initializing (index not ready).
      - Returns status="error" in body (not HTTP 500) for recoverable LLM errors.
      - Detects whether Agent B (RAG) was invoked from message history.
    """
    # Guard: system not ready
    if not request.app.state.index_ready or request.app.state.langgraph_app is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is still initializing. Please retry in a few seconds.",
        )

    lg_app = request.app.state.langgraph_app
    cfg = get_settings()

    # LangGraph config — thread_id isolates each user's conversation in Redis
    graph_config = {"configurable": {"thread_id": body.thread_id}}

    # Build the input state for this invocation
    graph_input = {"messages": [HumanMessage(content=body.message)]}

    logger.info(
        f"Chat request | thread_id={body.thread_id} | "
        f"message_preview={body.message[:60]!r}"
    )

    try:
        result_state = lg_app.invoke(graph_input, config=graph_config)
    except Exception as e:
        logger.error(
            f"LangGraph invocation failed | thread_id={body.thread_id} | error={e}",
            exc_info=True,
        )
        return ChatResponse(
            status="error",
            thread_id=body.thread_id,
            reply="专家系统连接出现问题，请稍后重试。",
            metadata=ChatMetadata(used_rag=False, error_msg=str(e)),
        )

    # Extract the final reply from the last AIMessage in state
    messages = result_state.get("messages", [])
    reply_text = "抱歉，系统未能生成有效回复，请重新提问。"
    used_rag = False

    from langchain_core.messages import AIMessage, ToolMessage
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            reply_text = msg.content
            break

    # Detect if RAG was used: check if any ToolMessage with '【知识库检索结果】' exists
    for msg in messages:
        if isinstance(msg, ToolMessage) and "【知识库检索结果】" in msg.content:
            used_rag = True
            break

    logger.info(
        f"Chat response | thread_id={body.thread_id} | "
        f"used_rag={used_rag} | reply_preview={reply_text[:80]!r}"
    )

    return ChatResponse(
        status="success",
        thread_id=body.thread_id,
        reply=reply_text,
        metadata=ChatMetadata(used_rag=used_rag),
    )
