import os
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from sarvamai import SarvamAI


# ============================================================
# Project environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# Configuration
# ============================================================

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if SARVAM_API_KEY:
    SARVAM_API_KEY = (
        SARVAM_API_KEY
        .strip()
        .strip('"')
        .strip("'")
    )

if not SARVAM_API_KEY:
    raise RuntimeError(
        "SARVAM_API_KEY is not set. "
        f"Expected it in: {ENV_FILE}"
    )


RAG_URL = "http://127.0.0.1:8002/ask"

MAX_AUDIO_SIZE = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".webm",
    ".m4a",
    ".aac",
    ".flac",
}


# ============================================================
# Constants
# ============================================================

NO_CONTEXT_ANSWER = (
    "दिए गए संदर्भ में इस प्रश्न का "
    "पर्याप्त उत्तर उपलब्ध नहीं है।"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="RAG-Goa Voice API",
    description=(
        "Voice-enabled Hindi RAG pipeline "
        "using Sarvam Saaras v3"
    ),
    version="1.1.0",
)


# ============================================================
# Sarvam client
# ============================================================

client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "RAG-Goa Voice API",
        "speech_to_text": "Sarvam Saaras v3",
        "language": "hi-IN",
        "rag_service": RAG_URL,
        "max_audio_size_mb": 10,
    }


# ============================================================
# Voice → STT → RAG
# ============================================================

@app.post("/voice-ask")
async def voice_ask(
    audio: UploadFile = File(...)
):

    # ========================================================
    # Validate upload
    # ========================================================

    if audio is None:
        raise HTTPException(
            status_code=400,
            detail="Audio file is required."
        )


    filename = (
        audio.filename
        or "audio.wav"
    )

    extension = Path(
        filename
    ).suffix.lower()


    # ========================================================
    # Validate extension
    # ========================================================

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                f"Allowed formats: "
                f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        )


    # ========================================================
    # Read uploaded audio
    # ========================================================

    try:

        audio_bytes = await audio.read()

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Unable to read audio file: {exc}"
        )


    # ========================================================
    # Validate audio
    # ========================================================

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail="Audio file is empty."
        )


    if len(audio_bytes) > MAX_AUDIO_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Audio file is too large. "
                "Maximum allowed size is 10 MB."
            )
        )


    # ========================================================
    # Temporary file
    # ========================================================

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_file.write(
                audio_bytes
            )

            temp_path = temp_file.name


        # ====================================================
        # Speech-to-text
        # ====================================================

        try:

            with open(
                temp_path,
                "rb"
            ) as audio_file:

                transcript_response = (
                    client.speech_to_text.transcribe(
                        file=audio_file,
                        model="saaras:v3",
                        language_code="hi-IN",
                    )
                )

        except Exception as exc:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Sarvam speech-to-text failed: "
                    f"{exc}"
                )
            )


        # ====================================================
        # Extract transcript
        # ====================================================

        transcript = getattr(
            transcript_response,
            "transcript",
            None
        )


        # Handle dictionary response
        if not transcript:

            if isinstance(
                transcript_response,
                dict
            ):

                transcript = (
                    transcript_response.get(
                        "transcript"
                    )
                )


        # ====================================================
        # Validate transcript
        # ====================================================

        if not transcript:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Sarvam returned an empty transcript."
                )
            )


        transcript = str(
            transcript
        ).strip()


        if not transcript:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Speech could not be transcribed."
                )
            )


        # ====================================================
        # Send transcript to RAG API
        # ====================================================

        try:

            rag_response = requests.post(
                RAG_URL,
                json={
                    "query": transcript
                },
                timeout=180
            )

        except requests.Timeout:

            raise HTTPException(
                status_code=504,
                detail="RAG service timed out."
            )

        except requests.RequestException as exc:

            raise HTTPException(
                status_code=503,
                detail=(
                    "RAG service unavailable: "
                    f"{exc}"
                )
            )


        # ====================================================
        # Handle RAG HTTP errors
        # ====================================================

        if rag_response.status_code != 200:

            raise HTTPException(
                status_code=502,
                detail=(
                    "RAG service returned HTTP "
                    f"{rag_response.status_code}: "
                    f"{rag_response.text[:1000]}"
                )
            )


        # ====================================================
        # Parse RAG response
        # ====================================================

        try:

            rag_data = rag_response.json()

        except ValueError:

            raise HTTPException(
                status_code=502,
                detail=(
                    "RAG service returned invalid JSON."
                )
            )


        # ====================================================
        # Extract answer
        # ====================================================

        answer = rag_data.get(
            "answer"
        )


        if not answer:

            answer = NO_CONTEXT_ANSWER


        # ====================================================
        # Final response
        # ====================================================

        return {

            "transcript":
                transcript,

            "answer":
                answer,

            "model":
                rag_data.get(
                    "model"
                ),

            "grounded":
                rag_data.get(
                    "grounded",
                    False
                ),

            "sources":
                rag_data.get(
                    "sources",
                    []
                ),

            "retrieval":
                rag_data.get(
                    "retrieval",
                    {}
                ),

            "speech_to_text": {

                "provider":
                    "Sarvam",

                "model":
                    "saaras:v3",

                "language":
                    "hi-IN"

            }

        }


    finally:

        # ====================================================
        # Cleanup temporary file
        # ====================================================

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

            except OSError:

                pass