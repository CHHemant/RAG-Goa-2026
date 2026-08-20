import statistics
import time
import requests


RAG_URL = "http://127.0.0.1:8002/ask"

QUERIES = [
    "मैनहट्टन परियोजना क्या थी?",
    "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
    "मैनहट्टन परियोजना कब शुरू हुई?",
    "मैनहट्टन परियोजना का नेतृत्व किसने किया?",
    "मैनहट्टन परियोजना में किन देशों का समर्थन था?",
    "मैनहट्टन परियोजना का उद्देश्य क्या था?",
    "परमाणु हथियारों के निर्माण से संबंधित परियोजना कौन सी थी?",
    "मैनहट्टन परियोजना कितने वर्षों तक चली?",
    "मैनहट्टन परियोजना के दौरान लॉस अलामोस प्रयोगशाला का क्या महत्व था?",
    "मैनहट्टन परियोजना किस युद्ध के दौरान हुई थी?",
]


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * (p / 100)

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def main():

    print("=" * 70)
    print("RAG-Goa Latency Benchmark")
    print("=" * 70)

    # --------------------------------------------------------
    # Health check
    # --------------------------------------------------------

    try:

        health = requests.get(
            "http://127.0.0.1:8002/health",
            timeout=5
        )

        health.raise_for_status()

        print("RAG API: ONLINE")

    except Exception as exc:

        print("RAG API unavailable:")
        print(exc)
        return


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print()
    print("Running warm-up request...")

    try:

        requests.post(
            RAG_URL,
            json={
                "query": QUERIES[0]
            },
            timeout=180
        )

    except Exception as exc:

        print("Warm-up failed:")
        print(exc)
        return


    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    latencies = []

    successful = 0
    failed = 0

    print()
    print("Running benchmark...")
    print()

    for i, query in enumerate(
        QUERIES,
        start=1
    ):

        start = time.perf_counter()

        try:

            response = requests.post(
                RAG_URL,
                json={
                    "query": query
                },
                timeout=180
            )

            elapsed = (
                time.perf_counter()
                - start
            ) * 1000

            if response.ok:

                successful += 1
                latencies.append(elapsed)

                print(
                    f"{i:02d}. "
                    f"{elapsed:8.2f} ms | "
                    f"{query}"
                )

            else:

                failed += 1

                print(
                    f"{i:02d}. FAILED | "
                    f"HTTP {response.status_code}"
                )

        except Exception as exc:

            failed += 1

            print(
                f"{i:02d}. FAILED | "
                f"{exc}"
            )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    if not latencies:

        print()
        print("No successful benchmark requests.")
        return


    p50 = percentile(
        latencies,
        50
    )

    p70 = percentile(
        latencies,
        70
    )

    p100 = max(
        latencies
    )

    average = statistics.mean(
        latencies
    )


    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Queries tested : {len(QUERIES)}"
    )

    print(
        f"Successful     : {successful}"
    )

    print(
        f"Failed         : {failed}"
    )

    print(
        f"Average        : {average:.2f} ms"
    )

    print(
        f"P50            : {p50:.2f} ms"
    )

    print(
        f"P70            : {p70:.2f} ms"
    )

    print(
        f"P100           : {p100:.2f} ms"
    )

    print("=" * 70)

    if p100 < 200:

        print(
            "STATUS: UNDER 200 ms TARGET"
        )

    else:

        print(
            "STATUS: ABOVE 200 ms TARGET"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()