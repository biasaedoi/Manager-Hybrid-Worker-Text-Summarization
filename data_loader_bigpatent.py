from datasets import load_dataset


def load_bigpatent_sample(n=20, start=0):
    """
    Load n samples from BIGPATENT test split
    using streaming to avoid downloading the full dataset.
    """

    dataset = load_dataset(
        "NortheasternUniversity/big_patent",
        "all",
        split="test",
        streaming=True
    )

    samples = []

    for index, item in enumerate(dataset):

        if index < start:
            continue

        samples.append({
            "id": str(index),
            "article": item["description"],
            "reference": item["abstract"]
        })

        if len(samples) >= n:
            break

    return samples