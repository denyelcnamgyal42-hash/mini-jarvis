Jarvis Agent
============

FastAPI app for a voice-first personal AI assistant served with a static HUD frontend.

Environment variables
---------------------

- `GROQ_API_KEY`: required for chat completions and Whisper transcription.
- `TAVILY_API_KEY`: optional but required for online web research.
- `JARVIS_DB_PATH`: optional SQLite path. Defaults to `jarvis.db`.

Render start command
--------------------

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Deployment note
---------------

The default SQLite database is fine for local development, but Render's normal filesystem is not persistent across redeploys. For production memory/tasks, move the database layer to PostgreSQL or attach persistent storage.
