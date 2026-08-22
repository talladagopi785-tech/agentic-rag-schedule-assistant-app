# Agentic RAG Schedule Assistant

A lightweight FastAPI + ChromaDB application that manages a user's schedule for the next 30 days.

## Features

- Sample meetings, workshops, tasks, and appointments.
- ChromaDB vector database.
- Basic RAG retrieval using deterministic lightweight embeddings (no large model download).
- Agentic routing between exactly two tools:
  - `get_schedule`: retrieves relevant schedule information.
  - `update_schedule`: adds, updates, or removes events.
- Natural-language examples:
  - `What do I have scheduled tomorrow?`
  - `Am I free Friday afternoon?`
  - `Add a meeting called Team Planning tomorrow at 3 PM`
  - `Move my meeting from 2 PM to 4 PM`
- FastAPI Swagger docs at `/docs`.
- Health check at `/health`.
- Browser UI at `/`.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## Render deployment

Render's official FastAPI deployment uses `pip install -r requirements.txt` as the build command and an Uvicorn command binding to `0.0.0.0:$PORT`.

1. Push this folder to a GitHub repository.
2. In Render, create a **New Web Service** and connect the GitHub repository.
3. Select branch `main`.
4. Render can read `render.yaml`, or enter:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy.
6. Open the generated `https://<service-name>.onrender.com/`.
7. API documentation: `https://<service-name>.onrender.com/docs`.

## Important storage note

The included ChromaDB and JSON files are local filesystem storage. On an ephemeral/free Render instance, filesystem changes are not guaranteed to survive service replacement/redeploy. For permanent multi-user production storage, replace the local schedule store with a managed database (or attach appropriate persistent storage).

## Memory target

The project intentionally avoids `sentence-transformers` and large LLM downloads. This keeps runtime memory substantially lower than an `all-mpnet-base-v2` deployment and is more appropriate for a <=512 MB target. The vector retrieval is still performed by ChromaDB.

## API

- `GET /health`
- `GET /api/schedule`
- `POST /api/chat` with `{"message":"..."}`
- `POST /tools/get_schedule` with `{"message":"..."}`
- `POST /tools/update_schedule`
- `POST /api/reindex`
