from datasets import load_dataset
import pandas as pd

AI_COLUMNS = [
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract",
]

print("Downloading dataset...")
ds_all = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

rows = []
for split_name, split_data in ds_all.items():
    df = split_data.to_pandas()
    df["generation_method"] = split_name

    for _, row in df.iterrows():
        if pd.notna(row.get("original_abstract")):
            rows.append({
                "text": str(row["original_abstract"]),
                "label": "human",
                "generation_method": split_name,
                "source_model": "human",
            })

    for col in AI_COLUMNS:
        if col not in df.columns:
            continue
        model_name = col.replace("_generated_abstract", "")
        for _, row in df.iterrows():
            if pd.notna(row.get(col)):
                rows.append({
                    "text": str(row[col]),
                    "label": "ai",
                    "generation_method": split_name,
                    "source_model": model_name,
                })

final_df = pd.DataFrame(rows)
print("Total rows:", final_df.shape[0])
print(final_df["label"].value_counts())
print(final_df["source_model"].value_counts())

final_df.to_parquet("data/raw/raw_data.parquet", index=False)
print("Saved to data/raw/raw_data.parquet")
