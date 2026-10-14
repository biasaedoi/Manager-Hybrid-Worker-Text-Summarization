from data_loader import load_bigpatent_sample


# =========================================================
# CONFIG
# =========================================================

START_INDEX = 0
END_INDEX = 53

SKIP_DOCUMENTS = {
    41,
    46,
    51
}

OUTPUT_FILE = "bigpatent_documents.txt"


# =========================================================
# LOAD DATASET
# =========================================================

samples = load_bigpatent_sample(
    n=END_INDEX,
    start=START_INDEX
)


# =========================================================
# WRITE TXT
# =========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    saved_count = 0

    for index, sample in enumerate(
        samples,
        start=1
    ):

        if index in SKIP_DOCUMENTS:
            continue

        saved_count += 1

        file.write("=" * 70)
        file.write("\n")
        file.write(
            f"Document = {index}\n"
        )
        file.write("=" * 70)
        file.write("\n\n")

        file.write("ARTICLE:\n")
        file.write(
            sample["article"]
        )
        file.write("\n\n")

        file.write(
            "REFERENCE / ABSTRACT:\n"
        )
        file.write(
            sample["reference"]
        )
        file.write("\n\n\n")


# =========================================================
# RESULT
# =========================================================

print("=" * 70)
print("DATA EXPORT COMPLETE")
print("=" * 70)

print(
    f"Documents saved : {saved_count}"
)

print(
    f"Skipped         : {sorted(SKIP_DOCUMENTS)}"
)

print(
    f"Output file     : {OUTPUT_FILE}"
)