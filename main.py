import os 
import tempfile
from groq import Groq
from fastapi import UploadFile, File, Form 

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import init_db, get_dashboard_data
from io import BytesIO
from pypdf import PdfReader


from agent import ask_jarvis, analyze_uploaded_file

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

def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """
    Extract text from supported uploaded files.
    """

    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        text_parts = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n\n".join(text_parts)

    text_extensions = (
    ".txt", ".md", ".py", ".js", ".html", ".css",
    ".json", ".csv", ".xml", ".yaml", ".yml", ".java",
    ".cpp", ".c", ".ts", ".tsx", ".jsx", ".vue",
    ".go", ".rs", ".php", ".rb", ".cs"
)
    

    if lower_name.endswith(text_extensions):
        return file_bytes.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, code, text, markdown, JSON, CSV, HTML, CSS, or JS files.")

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
                model="whisper-large-v3",
                response_format="text",
                language="en",
                prompt=(
        "The speaker is talking to an AI assistant called Mini Jarvis. "
        "Common topics include adding tasks, getting current time, Live Updates on Football Teams "
    )
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
    
@app.get("/dashboard")
def dashboard(session_id: str):
    try:
        data = get_dashboard_data(session_id)
        return data

    except Exception as e:
        print("Dashboard error:", e)
        return {
            "pending_tasks": 0,
            "completed_tasks": 0,
            "notes_count": 0,
            "today_focus": None,
            "error": str(e)
        }
    
@app.post("/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    instruction: str = Form("")
):
    try:
        print("File upload received:", file.filename)
        print("Session ID:", session_id)
        print("Instruction:", instruction)

        file_bytes = await file.read()

        max_size_mb = 5
        if len(file_bytes) > max_size_mb * 1024 * 1024:
            return {
                "reply": f"File is too large. Please upload a file smaller than {max_size_mb} MB."
            }

        file_text = extract_text_from_file(file.filename, file_bytes)

        if not file_text.strip():
            return {
                "reply": "I could not extract readable text from this file."
            }

        reply = analyze_uploaded_file(
            filename=file.filename,
            file_content=file_text,
            user_instruction=instruction,
            session_id=session_id
        )

        return {
            "filename": file.filename,
            "reply": reply
        }

    except Exception as e:
        print("File upload error:", e)
        return {
            "reply": f"File upload error: {str(e)}"
        }