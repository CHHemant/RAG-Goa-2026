import requests
import json
import time


RAG_URL = "http://127.0.0.1:8002/ask"


TEST_QUERIES = [

    "मैनहट्टन परियोजना क्या थी?",

    "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",

    "कंप्यूटर की परिभाषा क्या है?",

    "लंदन का समय भारतीय समय से कैसे संबंधित है?",

    "सैन फ्रांसिस्को में जुलाई का औसत तापमान क्या है?",

    "लाच शब्द का कानूनी अर्थ क्या है?",

    "क्यू.वी.एल.सी. सॉफ्टवेयर क्या है?",

    "लाइमिंग इंजन की कीमत क्या है?",

    "विजुअल स्टूडियो में समाधान फ़ाइल का नाम कैसे बदलते हैं?",

    "भारत का वर्तमान राष्ट्रपति कौन है?"
]


print("=" * 80)
print("RAG-Goa FINAL EVALUATION")
print("=" * 80)


for number, query in enumerate(
    TEST_QUERIES,
    start=1
):

    print()
    print("=" * 80)
    print(f"TEST {number}")
    print("=" * 80)

    print("QUERY:")
    print(query)

    start = time.perf_counter()

    try:

        response = requests.post(
            RAG_URL,
            json={
                "query": query
            },
            timeout=180
        )

        elapsed = time.perf_counter() - start

        response.raise_for_status()

        data = response.json()

        print()
        print("ANSWER:")
        print(data.get("answer"))

        print()
        print("MODEL:")
        print(data.get("model"))

        print()
        print("GROUNDED:")
        print(data.get("grounded"))

        print()
        print("LATENCY:")
        print(f"{elapsed:.2f} seconds")

        retrieval = data.get(
            "retrieval",
            {}
        )

        print()
        print("RETRIEVAL:")
        print(json.dumps(
            retrieval,
            ensure_ascii=False,
            indent=2
        ))

    except Exception as exc:

        print()
        print("ERROR:")
        print(exc)


print()
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)