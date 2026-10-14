from data_loader_bigpatent import load_bigpatent_sample
from manager import decide_route


# ============================================================
# CONFIG
# ============================================================

NUM_DOCUMENTS = 20
START_INDEX = 0


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_bigpatent_sample(
    n=NUM_DOCUMENTS,
    start=START_INDEX
)


# ============================================================
# COUNTERS
# ============================================================

slm_count = 0
llm_count = 0


# ============================================================
# TEST
# ============================================================

for i, item in enumerate(dataset):

    article = item["article"]

    result = decide_route(article)

    route = result["Route"]

    if route == "SLM":
        slm_count += 1

    elif route == "LLM":
        llm_count += 1


    print("=" * 70)
    print(f"DOCUMENT {i + 1}")
    print("=" * 70)

    print(f"Article words   : {len(article.split())}")
    print(f"ASL             : {result['ASL']}")
    print(f"Clause Density  : {result['ClauseDensity']}")
    print(f"Word Rarity     : {result['WordRarity']}")
    print(f"Route           : {result['Route']}")
    print(f"Reason          : {result['Reason']}")


# ============================================================
# SUMMARY
# ============================================================

total = len(dataset)

print()
print("=" * 70)
print("BIGPATENT ROUTING SUMMARY")
print("=" * 70)

print(f"Total documents : {total}")
print(f"SLM documents   : {slm_count}")
print(f"LLM documents   : {llm_count}")

if total > 0:

    print(
        f"SLM percentage  : "
        f"{slm_count / total * 100:.2f}%"
    )

    print(
        f"LLM percentage  : "
        f"{llm_count / total * 100:.2f}%"
    )