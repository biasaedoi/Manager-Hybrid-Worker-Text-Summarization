from data_loader import load_bigpatent_sample

samples = load_bigpatent_sample(
    n=7,
    start=40
)

for i, sample in enumerate(samples, start=41):
    article = sample["article"]

    print(
        f"Document {i}: "
        f"{len(article.split())} words | "
        f"{len(article)} characters"
    )