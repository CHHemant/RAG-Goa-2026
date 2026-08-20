from datasets import load_dataset

URL = "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/train/hintrain.parquet"

ds = load_dataset(
    "parquet",
    data_files=URL,
    split="train",
    streaming=True
)

for i, row in enumerate(ds):
    print(f"\n--- Record {i + 1} ---")
    print("Query:", row["query"])
    print("Answer:", row["Answer"])
    print("Translated passages:", len(row["passages"]["Translated_passages"]))

    if i >= 99:
        break

print("\nSuccessfully inspected 100 Hindi records.")