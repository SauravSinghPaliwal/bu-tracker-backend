"""
BU Tracker — Python LangChain chat backend.
Deploy on Vercel (as a standalone project) or any Python host.
"""
import json
import os
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI as RawOpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.3-70b-instruct"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "bu_projects")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="BU Tracker Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LangChain LLM (NVIDIA NIM — OpenAI-compatible)
llm = ChatOpenAI(
    model=CHAT_MODEL,
    api_key=NVIDIA_API_KEY,
    base_url=NVIDIA_BASE_URL,
    streaming=True,
    max_tokens=1024,
    temperature=0.7,
)

_embed_client: RawOpenAI | None = None


def get_embed_client() -> RawOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = RawOpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    return _embed_client


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class Project(BaseModel):
    id: str
    projectName: str
    customer: str
    value: float
    quarter: str
    status: str
    expectedClose: Optional[str] = None
    notes: Optional[str] = None


class ChatRequest(BaseModel):
    question: str
    allProjects: List[Project]
    annualTarget: float = 500.0


class SyncRequest(BaseModel):
    projects: List[Project]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VECTOR_SIZE = 1024  # nvidia/nv-embedqa-e5-v5 output dimension


def project_to_text(p: Project) -> str:
    return "\n".join([
        f"Project: {p.projectName}",
        f"Customer: {p.customer}",
        f"Value: \u20b9{p.value} Lakhs",
        f"Quarter: {p.quarter}",
        f"Status: {p.status}",
        f"Expected Close: {p.expectedClose or 'Not set'}",
        f"Notes: {p.notes or 'None'}",
    ])


def id_from_string(s: str) -> int:
    """Stable numeric ID from a string (djb2 hash → positive uint32)."""
    h = 5381
    for c in s:
        h = ((h << 5) + h) ^ ord(c)
    return h & 0xFFFFFFFF


def get_qdrant_client() -> QdrantClient:
    kwargs: dict = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def embed_query(text: str) -> list[float]:
    res = get_embed_client().embeddings.create(
        model=EMBED_MODEL,
        input=text,
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return res.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts as passages (for indexing)."""
    if not texts:
        return []
    res = get_embed_client().embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    return [d.embedding for d in sorted(res.data, key=lambda x: x.index)]


def get_qdrant_context(question: str) -> str:
    try:
        client = get_qdrant_client()

        info = client.get_collection(COLLECTION)
        if (info.points_count or 0) == 0:
            return ""

        vector = embed_query(question)
        results = client.search(
            COLLECTION,
            query_vector=vector,
            limit=5,
            score_threshold=0.3,
            with_payload=True,
        )
        if not results:
            return ""

        lines: list[str] = []
        for i, r in enumerate(results):
            p = r.payload or {}
            lines += [
                f"--- Project {i + 1} (score: {r.score * 100:.0f}%) ---",
                f"Name: {p.get('projectName', '')}",
                f"Customer: {p.get('customer', '')}",
                f"Value: \u20b9{p.get('value', 0)} Lakhs",
                f"Quarter: {p.get('quarter', '')}",
                f"Status: {p.get('status', '')}",
                f"Expected Close: {p.get('expectedClose') or 'Not set'}",
                f"Notes: {p.get('notes') or 'None'}",
                "",
            ]
        return "\n".join(lines)
    except Exception as exc:
        print(f"[Qdrant] Skipping vector search: {exc}")
        return ""


async def stream_chat(req: ChatRequest) -> AsyncGenerator[str, None]:
    context_text = get_qdrant_context(req.question)

    won = sum(p.value for p in req.allProjects if p.status == "Won")
    pipeline = sum(
        p.value for p in req.allProjects if p.status in ("Pipeline", "Negotiation")
    )
    gap = max(0.0, req.annualTarget - won)
    pct = (won / req.annualTarget * 100) if req.annualTarget > 0 else 0.0
    t = req.annualTarget
    target_display = (
        f"\u20b9{t / 100:.2f} Crore ({t:.0f} Lakhs)" if t >= 100 else f"\u20b9{t:.0f} Lakhs"
    )

    system_prompt = f"""You are an intelligent business analyst assistant for a Business Unit Head tracking FY 2026-27 performance against a {target_display} target.

BU SNAPSHOT:
- Annual Target: {target_display}
- Total Projects: {len(req.allProjects)}
- Won/Closed: \u20b9{won:.2f} L ({pct:.1f}%)
- In Pipeline: \u20b9{pipeline:.2f} L
- Gap to Target: \u20b9{gap:.2f} L

SEMANTICALLY RELEVANT PROJECTS (from Qdrant vector search):
{context_text or 'No closely matching projects found for this query.'}

INSTRUCTIONS:
- Be concise and data-driven
- Use \u20b9 Lakhs / Crores appropriately
- Highlight risks, wins, or actions needed
- If insufficient data, say so clearly"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=req.question),
    ]

    try:
        async for chunk in llm.astream(messages):
            text = chunk.content
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_chat(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/sync")
async def sync(req: SyncRequest) -> Dict:
    """
    Embed all projects with nv-embedqa-e5-v5 (passage mode) and upsert into Qdrant.
    Deletes orphaned points for projects no longer in the list.
    """
    client = get_qdrant_client()
    ensure_collection(client)

    if not req.projects:
        # Drop and recreate the collection to clear all points
        client.delete_collection(COLLECTION)
        ensure_collection(client)
        return {"synced": 0, "deleted": 0}

    # Batch-embed all projects as passages
    texts = [project_to_text(p) for p in req.projects]
    vectors = embed_batch(texts)

    points = [
        PointStruct(
            id=id_from_string(p.id),
            vector=vectors[i],
            payload=p.model_dump(),
        )
        for i, p in enumerate(req.projects)
    ]
    client.upsert(COLLECTION, points=points, wait=True)

    # Remove points for deleted projects
    current_ids = {id_from_string(p.id) for p in req.projects}
    scroll = client.scroll(COLLECTION, limit=1000, with_payload=False, with_vectors=False)
    orphans = [pt.id for pt in scroll[0] if pt.id not in current_ids]
    if orphans:
        client.delete(COLLECTION, points_selector=orphans)

    return {"synced": len(points), "deleted": len(orphans)}


@app.get("/status")
async def status() -> Dict:
    """Return Qdrant connection health and collection point count."""
    try:
        client = get_qdrant_client()
        collections = {c.name for c in client.get_collections().collections}
        exists = COLLECTION in collections
        point_count = 0
        if exists:
            info = client.get_collection(COLLECTION)
            point_count = info.points_count or 0
        return {"connected": True, "collection": COLLECTION, "exists": exists, "pointCount": point_count}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}
