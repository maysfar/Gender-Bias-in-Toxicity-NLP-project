from transformers import pipeline
import pandas as pd
import sys
from tqdm import tqdm  

# Input
input_file = sys.argv[1]

# Load dataset
data = pd.read_csv(input_file)

# Load classifier
classifier = pipeline(
    "text-classification",
    model="moatazhamza194/gender_classification-deberta",
    tokenizer="moatazhamza194/gender_classification-deberta",
)

# Neutral mask
neutral_mask = (data["male"] < 0.2) & (data["female"] < 0.2)
data["gender"] = "neutral"  # default

# Process only non-neutral rows
texts_to_classify = data.loc[~neutral_mask, "comment"]

# Batch processing
batch_size = 64
results = []

for i in tqdm(range(0, len(texts_to_classify), batch_size)):
    batch_texts = texts_to_classify.iloc[i:i+batch_size].tolist()
    preds = classifier(batch_texts, truncation=True, max_length=512, batch_size=batch_size)

    for pred in preds:
        pred_id = int(pred["label"].split("_")[-1])
        results.append("male" if pred_id == 1 else "female")

# Assign results back
data.loc[~neutral_mask, "gender"] = results

# Drop old columns
data = data.drop(columns=["male", "female"])

# Save output
data.to_csv("gendered_test_data", index=False)