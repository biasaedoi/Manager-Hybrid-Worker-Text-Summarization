from datasets import load_dataset


def load_bigpatent_sample(n=53, start=0):
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
            "id": item.get("id", index),
            "article": item["description"],
            "reference": item["abstract"]
        })

        if len(samples) >= n:
            break

    return samples