import os 
import tempfile
from groq import Groq
from fastapi import UploadFile, File, Form 

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import init_db

from agent import ask_jarvis 

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

@app.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = Form(...)
):
    try:
        print("Voice request received")
        print("Session ID:", session_id)
        print("Audio file:", audio.filename)

        suffix = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(await audio.read())
            temp_audio_path = temp_audio.name

        with open(temp_audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3-turbo",
                response_format="text"
            )

        os.remove(temp_audio_path)

        print("Transcribed text:", transcription)

        reply = ask_jarvis(transcription, session_id)

        return {
            "transcript": transcription,
            "reply": reply
        }

    except Exception as e:
        print("Voice backend error:", e)
        return {
            "transcript": "",
            "reply": f"I heard you, but Jarvis had an agent error: {str(e)}"
        }