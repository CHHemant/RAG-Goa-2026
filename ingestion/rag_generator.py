import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv


# ============================================================
# Project environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


# ============================================================
# Configuration
# ============================================================

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if NVIDIA_API_KEY:
    NVIDIA_API_KEY = (
        NVIDIA_API_KEY
        .strip()
        .strip('"')
        .strip("'")
    )

if not NVIDIA_API_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. "
        f"Expected it in: {ENV_FILE}"
    )


NVIDIA_URL = (
    "https://integrate.api.nvidia.com/v1/chat/completions"
)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

NVIDIA_TIMEOUT = 60

MAX_CONTEXTS = 5
MAX_CONTEXT_CHARS = 10000

MAX_OUTPUT_CHARS = 1500

NO_CONTEXT_ANSWER = (
    "दिए गए संदर्भ में इस प्रश्न का "
    "पर्याप्त उत्तर उपलब्ध नहीं है।"
)


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Model answer cleaning
# ============================================================

def clean_model_answer(answer: str) -> str:

    if not isinstance(answer, str):
        return ""

    answer = answer.strip()

    if not answer:
        return ""


    # --------------------------------------------------------
    # Remove code fences
    # --------------------------------------------------------

    answer = re.sub(
        r"^```(?:text|plain|markdown)?\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(
        r"\s*```$",
        "",
        answer
    )

    answer = answer.strip()


    # --------------------------------------------------------
    # Remove common answer prefixes
    # --------------------------------------------------------

    answer = re.sub(
        r"^(उत्तर|Answer)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE
    )


    # --------------------------------------------------------
    # Remove model meta-commentary
    # --------------------------------------------------------

    answer = re.sub(
        r"\s*\(\s*\d+\s*(?:शब्द|words?|word)\s*\)\s*$",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(
        r"\s*\(\s*(?:संक्षेप में|in short)\s*\)\s*$",
        "",
        answer,
        flags=re.IGNORECASE
    )


    # --------------------------------------------------------
    # Remove repeated identical tokens
    # --------------------------------------------------------

    tokens = answer.split()

    if len(tokens) >= 8:

        cleaned_tokens = []

        previous = None
        repeat_count = 0

        for token in tokens:

            normalized = token.lower()

            if normalized == previous:
                repeat_count += 1
            else:
                repeat_count = 1
                previous = normalized

            if repeat_count <= 2:
                cleaned_tokens.append(token)

        answer = " ".join(
            cleaned_tokens
        ).strip()


    # --------------------------------------------------------
    # Detect pathological character repetition
    # --------------------------------------------------------

    compact = re.sub(
        r"\s+",
        "",
        answer
    )

    if len(compact) >= 40:

        unique_chars = set(compact)

        if len(unique_chars) <= 4:
            return ""


    # --------------------------------------------------------
    # Remove obvious "thinking" leakage
    # --------------------------------------------------------

    forbidden_patterns = [
        r"<think>.*?</think>",
        r"<analysis>.*?</analysis>",
    ]

    for pattern in forbidden_patterns:

        answer = re.sub(
            pattern,
            "",
            answer,
            flags=re.IGNORECASE | re.DOTALL
        )


    answer = answer.strip()


    # --------------------------------------------------------
    # Hard safety limit
    # --------------------------------------------------------

    if len(answer) > MAX_OUTPUT_CHARS:

        answer = answer[
            :MAX_OUTPUT_CHARS
        ].rstrip()


    return answer


# ============================================================
# Generate grounded answer
# ============================================================

def generate_answer(
    query: str,
    contexts: list[dict]
) -> dict:

    # ========================================================
    # Validate query
    # ========================================================

    if not isinstance(query, str):

        raise ValueError(
            "Query must be a string."
        )

    query = clean_text(query)

    if not query:

        raise ValueError(
            "Query cannot be empty."
        )


    # ========================================================
    # Validate contexts
    # ========================================================

    if not isinstance(contexts, list):

        raise ValueError(
            "Contexts must be a list."
        )


    # ========================================================
    # No context
    # ========================================================

    if not contexts:

        return {
            "answer": NO_CONTEXT_ANSWER,
            "model": None,
            "grounded": False,
            "sources": []
        }


    # ========================================================
    # Build bounded context
    # ========================================================

    context_blocks = []
    valid_contexts = []

    total_chars = 0


    for index, item in enumerate(
        contexts[:MAX_CONTEXTS],
        start=1
    ):

        if not isinstance(item, dict):
            continue


        text = clean_text(
            str(
                item.get(
                    "text",
                    ""
                )
            )
        )


        if not text:
            continue


        remaining = (
            MAX_CONTEXT_CHARS
            - total_chars
        )


        if remaining <= 0:
            break


        if len(text) > remaining:

            text = text[
                :remaining
            ].rstrip()


        if not text:
            continue


        valid_contexts.append(item)

        context_blocks.append(
            f"[संदर्भ {index}]\n{text}"
        )

        total_chars += len(text)


    # ========================================================
    # No usable context
    # ========================================================

    if not context_blocks:

        return {
            "answer": NO_CONTEXT_ANSWER,
            "model": None,
            "grounded": False,
            "sources": []
        }


    context = "\n\n".join(
        context_blocks
    )


    # ========================================================
    # System prompt
    # ========================================================

    system_prompt = """
आप RAG-Goa के लिए एक सटीक और विश्वसनीय Hindi RAG
प्रश्न-उत्तर सहायक हैं।

आपका उत्तर केवल दिए गए संदर्भों पर आधारित होना चाहिए।

अनिवार्य नियम:

1. केवल दिए गए संदर्भों की जानकारी का उपयोग करें।
2. बाहरी ज्ञान, इंटरनेट या अपनी सामान्य जानकारी का उपयोग न करें।
3. संदर्भ में मौजूद नहीं तथ्य न जोड़ें।
4. अनुमान या कल्पना न करें।
5. यदि संदर्भ प्रश्न का पर्याप्त उत्तर नहीं देता है, तो केवल यह लिखें:
   "दिए गए संदर्भ में इस प्रश्न का पर्याप्त उत्तर उपलब्ध नहीं है।"
6. उपयोगकर्ता की भाषा में उत्तर दें।
7. सीधे प्रश्न का उत्तर दें।
8. उत्तर संक्षिप्त और तथ्यात्मक रखें।
9. सामान्यतः उत्तर 1 से 3 वाक्यों में रखें।
10. केवल अंतिम उत्तर दें।
11. reasoning, analysis या thinking process न दिखाएं।
12. कोई शब्द-गणना, मेटा-कमेंट्री या अतिरिक्त टिप्पणी न दें।
13. दोहराव, filler text या token artifacts न लिखें।
14. संदर्भ से बाहर की जानकारी का उपयोग न करें।
15. यदि उत्तर संदर्भ से स्पष्ट रूप से समर्थित नहीं है, तो उत्तर न गढ़ें।
""".strip()


    # ========================================================
    # User prompt
    # ========================================================

    user_prompt = f"""
प्रश्न:

{query}

उपलब्ध संदर्भ:

{context}

कार्य:

केवल ऊपर दिए गए संदर्भों के आधार पर प्रश्न का उत्तर दें।

उत्तर 1 से 3 वाक्यों में रखें।
केवल अंतिम उत्तर दें।
कोई reasoning नहीं।
कोई analysis नहीं।
कोई thinking process नहीं।
कोई शब्द-गणना नहीं।
कोई अतिरिक्त टिप्पणी नहीं।
संदर्भ से बाहर की जानकारी का उपयोग न करें।
""".strip()


    # ========================================================
    # NVIDIA payload
    # ========================================================

    payload = {

        "model": MODEL,

        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        "temperature": 0.0,

        "top_p": 0.8,

        "max_tokens": 128,

        "stream": False,

        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }


    # ========================================================
    # Headers
    # ========================================================

    headers = {

        "Authorization":
            f"Bearer {NVIDIA_API_KEY}",

        "Accept":
            "application/json",

        "Content-Type":
            "application/json"
    }


    # ========================================================
    # NVIDIA request
    # ========================================================

    try:

        response = requests.post(

            NVIDIA_URL,

            headers=headers,

            json=payload,

            timeout=NVIDIA_TIMEOUT
        )

    except requests.Timeout as exc:

        raise RuntimeError(
            "NVIDIA generation timed out."
        ) from exc

    except requests.RequestException as exc:

        raise RuntimeError(
            f"NVIDIA connection failed: {exc}"
        ) from exc


    # ========================================================
    # HTTP error
    # ========================================================

    if response.status_code != 200:

        raise RuntimeError(
            "NVIDIA generation failed: "
            f"HTTP {response.status_code} - "
            f"{response.text[:2000]}"
        )


    # ========================================================
    # Parse JSON
    # ========================================================

    try:

        data = response.json()

    except ValueError as exc:

        raise RuntimeError(
            "NVIDIA returned invalid JSON."
        ) from exc


    # ========================================================
    # Validate choices
    # ========================================================

    choices = data.get(
        "choices",
        []
    )


    if (
        not isinstance(choices, list)
        or not choices
    ):

        raise RuntimeError(
            "NVIDIA returned no choices."
        )


    choice = choices[0]


    if not isinstance(
        choice,
        dict
    ):

        raise RuntimeError(
            "NVIDIA returned an invalid choice."
        )


    # ========================================================
    # Extract message
    # ========================================================

    message = choice.get(
        "message",
        {}
    )


    if not isinstance(
        message,
        dict
    ):

        raise RuntimeError(
            "NVIDIA returned an invalid message."
        )


    # ========================================================
    # Extract answer
    # ========================================================

    answer = message.get(
        "content",
        ""
    )


    if not isinstance(
        answer,
        str
    ):

        answer = ""


    answer = clean_model_answer(
        answer
    )


    # ========================================================
    # Empty / corrupted answer protection
    # ========================================================

    if not answer:

        raise RuntimeError(
            "NVIDIA returned an empty or "
            "invalid generated answer. "
            f"Finish reason: "
            f"{choice.get('finish_reason')}"
        )


    # ========================================================
    # Build source metadata
    # ========================================================

    sources = []


    for index, item in enumerate(
        valid_contexts,
        start=1
    ):

        sources.append({

            "context_id":
                index,

            "source":
                item.get(
                    "source"
                ),

            "language":
                item.get(
                    "language"
                ),

            "passage_index":
                item.get(
                    "passage_index"
                ),

            "score":
                item.get(
                    "score"
                ),

            "rerank_score":
                item.get(
                    "rerank_score"
                ),

            "is_selected":
                item.get(
                    "is_selected"
                )

        })


    # ========================================================
    # Final response
    # ========================================================

    return {

        "answer":
            answer,

        "model":
            data.get(
                "model",
                MODEL
            ),

        "grounded":
            True,

        "sources":
            sources
    }