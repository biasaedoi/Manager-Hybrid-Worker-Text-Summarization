from data_loader import load_arxiv_sample

dataset = load_arxiv_sample(
    start=0,
    n=10
)

for i, item in enumerate(dataset):
    with open(f"document_{i}.txt", "w", encoding="utf-8") as f:
        f.write(item["article"])