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

# ===== COMMON / SQLITE CONFIG =====
DATA_DIR = Path(os.getenv("STUNN_DATA_DIR", os.getenv("DATA_DIR", str(Path(__file__).resolve().parent / "data"))))
DB_PATH = Path(os.getenv("STUNN_LIBRARY_DB", str(DATA_DIR / "library.db")))
LIBRARY_STORE = (os.getenv("LIBRARY_STORE", "").strip().lower())

EMBEDDINGS_BASE_URL = os.getenv("EMBEDDINGS_BASE_URL", os.getenv("OPENAI_COMPAT_BASE_URL", "")).rstrip("/")
EMBEDDINGS_API_KEY = os.getenv("EMBEDDINGS_API_KEY", os.getenv("OPENAI_COMPAT_API_KEY", ""))
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
EMBEDDINGS_ENABLED = os.getenv("EMBEDDINGS_ENABLED", "true").strip().lower() not in ("0", "false", "no")

# ===== NOTION CONFIG =====
NOTION_API_KEY = (os.getenv("NOTION_API_KEY", "") or "").strip()
NOTION_LIBRARY_DATABASE_ID = (os.getenv("NOTION_LIBRARY_DATABASE_ID", "") or os.getenv("NOTION_DATABASE_ID", "")).strip()
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28").strip()


def using_notion() -> bool:
    if LIBRARY_STORE == "sqlite":
        return False
    if LIBRARY_STORE == "notion":
        return True
    return bool(NOTION_API_KEY and NOTION_LIBRARY_DATABASE_ID)


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
    if using_notion():
        # Notion is external; no local DB init required.
        return
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
    if not texts or using_notion() or not EMBEDDINGS_ENABLED or not EMBEDDINGS_BASE_URL or not EMBEDDINGS_API_KEY or httpx is None:
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


# ===== NOTION IMPLEMENTATION =====

def _notion_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


async def _notion(method: str, path: str, payload: dict | None = None) -> dict:
    if httpx is None:
        raise RuntimeError("httpx is required for Notion library storage.")
    if not NOTION_API_KEY or not NOTION_LIBRARY_DATABASE_ID:
        raise RuntimeError("Set NOTION_API_KEY and NOTION_LIBRARY_DATABASE_ID to use Notion library storage.")
    url = "https://api.notion.com/v1" + path
    async with httpx.AsyncClient(timeout=45.0) as client:
        resp = await client.request(method, url, headers=_notion_headers(), json=payload)
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = resp.json().get("message") or resp.text
            except Exception:
                detail = resp.text
            raise RuntimeError(f"Notion API {resp.status_code}: {detail}")
        return resp.json() if resp.content else {}


def _rt(text: str, limit: int = 1900) -> list[dict]:
    text = (text or "")
    parts = []
    for i in range(0, len(text), limit):
        chunk = text[i:i + limit]
        if chunk:
            parts.append({"type": "text", "text": {"content": chunk}})
    return parts or [{"type": "text", "text": {"content": ""}}]


def _plain_title(prop: dict | None) -> str:
    for t in (prop or {}).get("title") or []:
        if t.get("plain_text"):
            return t["plain_text"]
    return ""


def _plain_rich(prop: dict | None) -> str:
    return "".join(t.get("plain_text", "") for t in ((prop or {}).get("rich_text") or []))


def _blocks_from_text(content: str) -> list[dict]:
    blocks: list[dict] = []
    text = (content or "").replace("\r\n", "\n").strip()
    for para in re.split(r"\n\s*\n+", text):
        para = para.strip()
        if not para:
            continue
        for i in range(0, len(para), 1900):
            part = para[i:i + 1900]
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(part)}})
            if len(blocks) >= 90:
                return blocks
    return blocks or [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": _rt(text[:1900])}}]


async def _notion_page_text(page_id: str) -> str:
    texts: list[str] = []
    cursor = None
    for _ in range(8):  # paginate up to ~800 blocks
        path = f"/blocks/{page_id}/children?page_size=100"
        if cursor:
            path += f"&start_cursor={cursor}"
        data = await _notion("GET", path)
        for b in data.get("results") or []:
            btype = b.get("type")
            obj = b.get(btype) or {}
            rt = obj.get("rich_text") or []
            line = "".join(t.get("plain_text", "") for t in rt).strip()
            if line:
                texts.append(line)
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not cursor:
            break
    return "\n\n".join(texts).strip()


async def _notion_find_by_name(user_id: str, name: str) -> dict | None:
    # Query database and match by name (+ optional user). Simple and reliable for named chapters.
    payload = {"page_size": 100, "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}]}
    data = await _notion("POST", f"/databases/{NOTION_LIBRARY_DATABASE_ID}/query", payload)
    want = (name or "").strip().lower()
    uid = (user_id or "").strip()
    for page in data.get("results") or []:
        props = page.get("properties") or {}
        title = _plain_title(props.get("Name") or props.get("name") or props.get("Title") or props.get("title")).strip()
        page_uid = _plain_rich(props.get("UserId") or props.get("userId") or props.get("User")).strip()
        if uid and page_uid and page_uid != uid:
            continue
        if title.lower() == want or (want and want in title.lower()):
            return page
    return None


async def _ingest_notion(user_id: str, name: str, content: str, source: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    existing = await _notion_find_by_name(user_id, name)
    if existing and existing.get("id"):
        try:
            await _notion("PATCH", f"/pages/{existing['id']}", {"archived": True})
        except Exception:
            pass
    props = {
        "Name": {"title": _rt(name, 2000)},
    }
    # Optional properties if they exist in the user's DB; Notion ignores? No, unknown props error.
    # Keep payload minimal/compatible: only Name. Store user/source in page content header.
    header = f"Stunn Library Document\nName: {name}\nUserId: {(user_id or '').strip()}\nSource: {source or ''}\nSaved: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
    body = header + "\n" + (content or "").strip()
    payload = {
        "parent": {"database_id": NOTION_LIBRARY_DATABASE_ID},
        "properties": props,
        "children": _blocks_from_text(body),
    }
    page = await _notion("POST", "/pages", payload)
    return {
        "id": page.get("id"),
        "name": name,
        "chunks": 1,
        "embedded": 0,
        "store": "notion",
        "db_path": f"notion:database:{NOTION_LIBRARY_DATABASE_ID}",
    }


def _list_notion(user_id: str) -> list[dict[str, Any]]:
    # sync wrapper not ideal, but our FastAPI endpoints can handle; use anyio? simpler: keep sync signature and run via asyncio in caller if needed.
    # We'll provide async-capable public functions below.
    raise RuntimeError("Use async Notion list.")


async def _list_notion_async(user_id: str) -> list[dict[str, Any]]:
    payload = {"page_size": 100, "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}]}
    data = await _notion("POST", f"/databases/{NOTION_LIBRARY_DATABASE_ID}/query", payload)
    uid = (user_id or "").strip()
    out = []
    for page in data.get("results") or []:
        props = page.get("properties") or {}
        title = _plain_title(props.get("Name") or props.get("name") or props.get("Title") or props.get("title")).strip() or "Untitled"
        # If user stored UserId in page text, we can't filter cheaply here without fetching blocks; allow all if no property.
        page_uid = _plain_rich(props.get("UserId") or props.get("userId") or props.get("User")).strip()
        if uid and page_uid and page_uid != uid:
            continue
        ts = page.get("last_edited_time") or page.get("created_time") or ""
        # Convert ISO-ish roughly
        try:
            from datetime import datetime
            updated_at = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            updated_at = time.time()
        out.append({"id": page.get("id"), "user_id": page_uid or uid, "name": title, "source": "notion", "created_at": updated_at, "updated_at": updated_at, "chunks": 1})
    # If DB has no UserId property and uid was requested, we returned all (best-effort MVP).
    return out


async def _search_notion(user_id: str, query: str, doc_name: str | None = None, doc_id: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
    if doc_id:
        try:
            page = await _notion("GET", f"/pages/{doc_id}")
            props = page.get("properties") or {}
            title = _plain_title(props.get("Name") or props.get("name") or props.get("Title") or props.get("title")).strip() or "Untitled"
            text = await _notion_page_text(doc_id)
            return [{"chunk_id": doc_id, "document_id": doc_id, "name": title, "text": text[:12000], "match": "document", "score": 1.0}] if text else []
        except Exception:
            pass
    name = (doc_name or query or "").strip()
    page = await _notion_find_by_name(user_id, name)
    if not page:
        # fallback: list and fuzzy pick
        docs = await _list_notion_async(user_id)
        q = (query or "").lower()
        for d in docs:
            if q and (q in d["name"].lower() or d["name"].lower() in q):
                page = {"id": d["id"], "properties": {"Name": {"title": [{"plain_text": d["name"]}]}}}
                break
        if not page and docs:
            page = {"id": docs[0]["id"], "properties": {"Name": {"title": [{"plain_text": docs[0]["name"]}]}}} if not (doc_name or query) else None
    if not page:
        return []
    pid = page.get("id")
    props = page.get("properties") or {}
    title = _plain_title(props.get("Name") or props.get("name") or props.get("Title") or props.get("title")).strip() or (doc_name or "Untitled")
    text = await _notion_page_text(pid)
    if not text:
        return []
    # Simple chunking for context size
    chunks = _chunk_text(text, max_chars=1800, overlap=120) or [text[:4000]]
    ql = (query or "").lower()
    scored = []
    for i, ch in enumerate(chunks):
        s = 1.0 + (0.5 if (ql and any(w in ch.lower() for w in re.findall(r"[a-z0-9]+", ql)[:8])) else 0.0) - i * 0.01
        scored.append((s, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"chunk_id": f"{pid}:{i}", "document_id": pid, "name": title, "text": ch, "match": "notion", "score": round(s, 4)} for i, (s, ch) in enumerate(scored[:max(1, min(int(limit or 5), 8))])]


async def _delete_notion(user_id: str, doc_id: str) -> bool:
    try:
        # best-effort: verify accessible by listing? MVP archives directly.
        await _notion("PATCH", f"/pages/{doc_id}", {"archived": True})
        return True
    except Exception:
        return False


# ===== SQLITE IMPLEMENTATION (existing behavior) =====

async def _ingest_sqlite(user_id: str, name: str, content: str, source: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
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
        return {"id": doc_id, "name": name, "chunks": len(chunks), "embedded": sum(1 for e in embeddings if e is not None), "db_path": str(DB_PATH), "store": "sqlite"}
    finally:
        conn.close()


def list_documents(user_id: str) -> list[dict[str, Any]]:
    if using_notion():
        # Should be called via async wrapper in FastAPI; provide safe fallback.
        try:
            import asyncio
            return asyncio.get_event_loop().run_until_complete(_list_notion_async(user_id))  # may fail in running loop
        except Exception:
            return []
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
    if using_notion():
        try:
            import asyncio
            return asyncio.get_event_loop().run_until_complete(_delete_notion(user_id, doc_id))
        except Exception:
            return False
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
    if using_notion():
        return await _search_notion(user_id, query, doc_name=doc_name, doc_id=doc_id, limit=limit)
    init_db()
    user_id = (user_id or "").strip()
    query = (query or "").strip()
    limit = max(1, min(int(limit or 5), 12))
    conn = _connect()
    try:
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
                    s = 1.0 / (1.0 + max(0.0, float(r["rank"]) * -1.0 if r["rank"] < 0 else float(r["rank"])))
                    scores[cid] = scores.get(cid, 0.0) + s
                    payload[cid] = {"chunk_id": cid, "document_id": r["document_id"], "name": r["name"], "text": r["text"], "match": "fts"}
            except Exception as exc:
                print(f"[FTS SEARCH ERROR] {exc}")

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


# ===== PUBLIC API =====

async def ingest_document(user_id: str, name: str, content: str, source: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    if using_notion():
        return await _ingest_notion(user_id, name, content, source, metadata)
    return await _ingest_sqlite(user_id, name, content, source, metadata)


async def list_documents_async(user_id: str) -> list[dict[str, Any]]:
    if using_notion():
        return await _list_notion_async(user_id)
    return list_documents(user_id)


async def delete_document_async(user_id: str, doc_id: str) -> bool:
    if using_notion():
        return await _delete_notion(user_id, doc_id)
    return delete_document(user_id, doc_id)


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
