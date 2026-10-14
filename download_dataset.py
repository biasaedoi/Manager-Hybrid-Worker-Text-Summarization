from data_loader import load_bigpatent_sample

samples = load_bigpatent_sample(
    n=50,
    start=0
)

print(f"\nBerhasil mendapatkan {len(samples)} dokumen.\n")

for i, sample in enumerate(samples, start=1):
    print("=" * 80)
    print(f"DOKUMEN {i}")
    print(f"ID        : {sample['id']}")
    print(f"Article   : {len(sample['article'].split())} words")
    print(f"Reference : {len(sample['reference'].split())} words")
    print("\nARTICLE:")
    print(sample["article"])
    print("\nREFERENCE / ABSTRACT:")
    print(sample["reference"])
    print()