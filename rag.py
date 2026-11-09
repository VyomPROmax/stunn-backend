import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

DATA_DIR = Path(os.getenv("STUNN_DATA_DIR", os.getenv("DATA_DIR", str(Path(__file__).resolve().parent / "data"))))
DB_PATH = Path(os.getenv("STUNN_LIBRARY_DB", str(DATA_DIR / "library.db")))

EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", os.getenv("OPENAI_COMPAT_BASE_URL", "")).rstrip("/")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY", os.getenv("OPENAI_COMPAT_API_KEY", ""))
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").strip().lower() not in ("0", "false", "no")

_db_lock = threading.Lock()
_db_ready = False


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    global _db_ready
    with _db_lock:
        if _db_ready:
            return
        conn = _connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    name TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_documents_name ON documents(user_id, name);

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    user_id TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    embedding_json TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id, chunk_index);
                CREATE INDEX IF NOT EXISTS idx_chunks_user ON chunks(user_id);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    document_id UNINDEXED,
                    user_id UNINDEXED,
                    name,
                    text,
                    tokenize='porter unicode61'
                );
                """
            )
            conn.commit()
            _db_ready = True
        finally:
            conn.close()


def _doc_id(user_id: str, name: str) -> str:
    base = f"{user_id or 'anon'}::{(name or '').strip().lower()}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]


def _chunk_text(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(p) > max_chars:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            start = 0
            while start < len(p):
                part = p[start:start + max_chars].strip()
                if part:
                    chunks.append(part)
                if start + max_chars >= len(p):
                    break
                start = max(0, start + max_chars - overlap)
            continue
        candidate = (cur + "\n\n" + p).strip() if cur else p
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                chunks.append(cur.strip())
            cur = p
    if cur:
        chunks.append(cur.strip())
    # de-dup tiny/near-empty
    return [c for c in chunks if len(c) >= 20]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


async def embed_texts(texts: list[str]) -> list[list[float] | None]:
    if not texts or not EMBEDDINGS_ENABLED or not EMBEDDINGS_BASE_URL or not EMBEDDINGS_API_KEY or httpx is None:
        return [None] * len(texts)
    url = EMBEDDINGS_BASE_URL.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {EMBEDDINGS_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDINGS_MODEL, "input": texts[:64]}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("data") or []
        by_index = {int(it.get("index", i)): it.get("embedding") for i, it in enumerate(items)}
        out: list[list[float] | None] = []
        for i in range(min(len(texts), 64)):
            emb = by_index.get(i)
            out.append([float(x) for x in emb] if isinstance(emb, list) else None)
        out.extend([None] * max(0, len(texts) - len(out)))
        return out
    except Exception as exc:
        print(f"[EMBEDDINGS ERROR] {exc}")
        return [None] * len(texts)


async def ingest_document(user_id: str, name: str, content: str, source: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    init_db()
    user_id = (user_id or "").strip()
    name = (name or "").strip() or "Untitled"
    content = (content or "").strip()
    if not content:
        raise ValueError("No content to store.")
    doc_id = _doc_id(user_id, name)
    now = time.time()
    chunks = _chunk_text(content)
    if not chunks:
        raise ValueError("Content was too short to index after processing.")
    embeddings = await embed_texts(chunks)

    conn = _connect()
    try:
        conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.execute(
            "INSERT INTO documents(id, user_id, name, source, content, metadata_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
            (doc_id, user_id, name, source or "", content, json.dumps(metadata or {}, ensure_ascii=False), now, now),
        )
        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            cid = f"{doc_id}:{i}"
            conn.execute(
                "INSERT INTO chunks(id, document_id, user_id, chunk_index, text, embedding_json, created_at) VALUES (?,?,?,?,?,?,?)",
                (cid, doc_id, user_id, i, text, json.dumps(emb) if emb is not None else None, now),
            )
            conn.execute(
                "INSERT INTO chunks_fts(chunk_id, document_id, user_id, name, text) VALUES (?,?,?,?,?)",
                (cid, doc_id, user_id, name, text),
            )
        conn.commit()
        return {"id": doc_id, "name": name, "chunks": len(chunks), "embedded": sum(1 for e in embeddings if e is not None), "db_path": str(DB_PATH)}
    finally:
        conn.close()


def list_documents(user_id: str) -> list[dict[str, Any]]:
    init_db()
    user_id = (user_id or "").strip()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT d.id, d.user_id, d.name, d.source, d.created_at, d.updated_at, (SELECT COUNT(*) FROM chunks c WHERE c.document_id=d.id) AS chunks "
            "FROM documents d WHERE d.user_id=? ORDER BY d.updated_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_document(user_id: str, doc_id: str) -> bool:
    init_db()
    conn = _connect()
    try:
        row = conn.execute("SELECT id FROM documents WHERE id=? AND user_id=?", (doc_id, (user_id or "").strip())).fetchone()
        if not row:
            return False
        conn.execute("DELETE FROM chunks_fts WHERE document_id=?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, (user_id or "").strip()))
        conn.commit()
        return True
    finally:
        conn.close()


async def search_library(user_id: str, query: str, doc_name: str | None = None, doc_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    init_db()
    user_id = (user_id or "").strip()
    query = (query or "").strip()
    limit = max(1, min(int(limit or 5), 12))
    conn = _connect()
    try:
        # Resolve target doc by id or name hint.
        target_id = (doc_id or "").strip() or None
        if not target_id and doc_name:
            row = conn.execute(
                "SELECT id FROM documents WHERE user_id=? AND lower(name)=lower(?) ORDER BY updated_at DESC LIMIT 1",
                (user_id, doc_name.strip()),
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT id FROM documents WHERE user_id=? AND lower(name) LIKE ? ORDER BY updated_at DESC LIMIT 1",
                    (user_id, f"%{doc_name.strip().lower()}%"),
                ).fetchone()
            if row:
                target_id = row["id"]

        scores: dict[str, float] = {}
        payload: dict[str, dict[str, Any]] = {}

        # FTS / keyword pass (good for named chapter recall).
        if query:
            fts_q = " OR ".join(re.findall(r"[A-Za-z0-9_]+", query)[:12]) or query
            try:
                sql = (
                    "SELECT f.chunk_id, f.document_id, f.name, c.text, bm25(chunks_fts) AS rank "
                    "FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id WHERE f.user_id=? AND chunks_fts MATCH ?"
                )
                params: list[Any] = [user_id, fts_q]
                if target_id:
                    sql += " AND f.document_id=?"
                    params.append(target_id)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit * 3)
                for r in conn.execute(sql, params).fetchall():
                    cid = r["chunk_id"]
                    # bm25: lower is better (often negative)
                    s = 1.0 / (1.0 + max(0.0, float(r["rank"]) * -1.0 if r["rank"] < 0 else float(r["rank"])))
                    scores[cid] = scores.get(cid, 0.0) + s
                    payload[cid] = {"chunk_id": cid, "document_id": r["document_id"], "name": r["name"], "text": r["text"], "match": "fts"}
            except Exception as exc:
                print(f"[FTS SEARCH ERROR] {exc}")

        # Semantic pass.
        qembs = await embed_texts([query]) if query else [None]
        qemb = qembs[0] if qembs else None
        if qemb is not None:
            sql = "SELECT c.id, c.document_id, c.text, c.embedding_json, d.name FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.user_id=?"
            params = [user_id]
            if target_id:
                sql += " AND c.document_id=?"
                params.append(target_id)
            for r in conn.execute(sql, params).fetchall():
                if not r["embedding_json"]:
                    continue
                try:
                    emb = json.loads(r["embedding_json"])
                    sim = _cosine(qemb, emb)
                except Exception:
                    sim = -1.0
                if sim <= 0.05:
                    continue
                cid = r["id"]
                scores[cid] = scores.get(cid, 0.0) + max(0.0, sim) * 1.25
                if cid not in payload:
                    payload[cid] = {"chunk_id": cid, "document_id": r["document_id"], "name": r["name"], "text": r["text"], "match": "semantic"}
                else:
                    payload[cid]["match"] = payload[cid]["match"] + "+semantic"

        # If a specific doc was requested but query had weak matches, return leading chunks.
        if target_id and not payload:
            for r in conn.execute(
                "SELECT c.id, c.document_id, c.text, d.name FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.user_id=? AND c.document_id=? ORDER BY c.chunk_index LIMIT ?",
                (user_id, target_id, limit),
            ).fetchall():
                payload[r["id"]] = {"chunk_id": r["id"], "document_id": r["document_id"], "name": r["name"], "text": r["text"], "match": "document"}
                scores[r["id"]] = 1.0

        ranked = sorted(payload.values(), key=lambda x: scores.get(x["chunk_id"], 0.0), reverse=True)[:limit]
        for item in ranked:
            item["score"] = round(scores.get(item["chunk_id"], 0.0), 4)
        return ranked
    finally:
        conn.close()


def format_library_context(results: list[dict[str, Any]], max_chars: int = 9000) -> str:
    if not results:
        return ""
    parts = ["[STORED LIBRARY CONTEXT — retrieved from previously saved material. Use this as the source of truth for questions about it.]"]
    total = 0
    for i, r in enumerate(results, 1):
        block = f"\nSource {i} ({r.get('name','Untitled')}):\n{(r.get('text') or '').strip()}\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining] + "\n[...truncated]")
            break
        parts.append(block)
        total += len(block)
    parts.append("\n[END STORED LIBRARY CONTEXT]")
    return "".join(parts)
