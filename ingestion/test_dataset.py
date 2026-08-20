from datasets import load_dataset

print("Connecting to MSMARCO-XI...")

ds = load_dataset(
    "ai4bharat/MSMARCO-XI",
    split="train",
    streaming=True
)

print("Dataset connected.")
print("Features:", ds.features)

print("Reading one sample...")

for sample in ds:
    print("\nSample received:")
    print(sample)
    break

print("\nDone.")