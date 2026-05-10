from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import init_db

from agent import ask_jarvis 

load_dotenv()

app = FastAPI(title="Mini Jarvis Agent")
init_db()

app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    session_id: str  

class ChatResponse(BaseModel):
    reply: str 

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        print("User message:", request.message)
        print("Session ID:", request.session_id)

        reply = ask_jarvis(request.message, request.session_id)

        print("Jarvis reply:", reply)

        return {"reply": reply}

    except Exception as e:
        print("Backend error:", e)
        return {"reply": f"Backend error: {str(e)}"}