from data_loader import load_bigpatent_sample
from chunker import create_chunks
from manager import decide_route


# =========================================================
# CONFIG
# =========================================================

NUM_DOCUMENTS = 10
START_INDEX = 0


# =========================================================
# LOAD DATASET
# =========================================================

print("=" * 70)
print("LOADING BIGPATENT")
print("=" * 70)

dataset = load_bigpatent_sample(
    n=NUM_DOCUMENTS,
    start=START_INDEX
)

print(
    f"Documents loaded: {len(dataset)}"
)


# =========================================================
# COUNTERS
# =========================================================

total_chunks = 0
slm_chunks = 0
llm_chunks = 0


# =========================================================
# TEST
# =========================================================

for i, item in enumerate(dataset):

    article = item["article"]

    print("\n")
    print("=" * 70)
    print(
        f"DOCUMENT {i + 1}/{len(dataset)}"
    )
    print("=" * 70)

    print(
        f"Original words : "
        f"{len(article.split())}"
    )

    # =====================================================
    # CHUNKING
    # =====================================================

    chunks = create_chunks(
        article
    )

    print(
        f"Total chunks   : "
        f"{len(chunks)}"
    )

    print()
    print("CHUNKS:")
    print("-" * 70)

    # =====================================================
    # ROUTING EACH CHUNK
    # =====================================================

    for chunk_index, chunk in enumerate(
        chunks,
        start=1
    ):

        routing = decide_route(
            chunk
        )

        route = routing["Route"]

        words = len(
            chunk.split()
        )

        if route == "SLM":
            slm_chunks += 1

        elif route == "LLM":
            llm_chunks += 1

        total_chunks += 1

        print(
            f"Chunk "
            f"{chunk_index:>2} | "
            f"{words:>4} words | "
            f"{route:<3} | "
            f"ASL="
            f"{routing['ASL']:.3f} | "
            f"Clause="
            f"{routing['ClauseDensity']:.3f} | "
            f"Rarity="
            f"{routing['WordRarity']:.3f} | "
            f"{routing['Reason']}"
        )


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("CHUNKING + ROUTING SUMMARY")
print("=" * 70)

print(
    f"Documents tested : "
    f"{len(dataset)}"
)

print(
    f"Total chunks     : "
    f"{total_chunks}"
)

print(
    f"SLM chunks       : "
    f"{slm_chunks}"
)

print(
    f"LLM chunks       : "
    f"{llm_chunks}"
)

if total_chunks > 0:

    print(
        f"SLM percentage   : "
        f"{slm_chunks / total_chunks:.2%}"
    )

    print(
        f"LLM percentage   : "
        f"{llm_chunks / total_chunks:.2%}"
    )