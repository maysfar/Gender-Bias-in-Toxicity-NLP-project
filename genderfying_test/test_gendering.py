from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Input
dataset = load_dataset("moatazhamza194/gb_test")
data = dataset["train"].to_pandas()

# Load classifier
classifier = pipeline(
    "text-classification",
    model="moatazhamza194/gender_classification-deberta",
    tokenizer="moatazhamza194/gender_classification-deberta",
)

# Initialize gender column as neutral
data["gender"] = "neutral"

# --- Rule-based logic ---
def assign_gender(row):
    male_val = row["male"]
    female_val = row["female"]

    # Case 1: Female strong
    if female_val >= 0.5 and male_val < 0.5:
        return "female"

    # Case 2: Male strong
    elif male_val >= 0.5 and female_val < 0.5:
        return "male"

    # Case 3: Neutral
    elif male_val == 0 and female_val == 0:
        return "neutral"

    # Case 4 & 5: Ambiguous cases -> return None for model classification
    elif (0 < female_val < 0.5 and 0 < male_val < 0.5) or (female_val > 0.5 and male_val > 0.5):
        return None

    # Fallback (just in case)
    return "neutral"

# Apply rule-based logic
data["gender"] = data.apply(assign_gender, axis=1)

# --- Model for ambiguous cases ---
ambiguous_mask = data["gender"].isna()
texts_to_classify = data.loc[ambiguous_mask, "comment"]

batch_size = 64
results = []

for i in tqdm(range(0, len(texts_to_classify), batch_size)):
    batch_texts = texts_to_classify.iloc[i:i+batch_size].tolist()
    preds = classifier(batch_texts, truncation=True, max_length=512, batch_size=batch_size)

    for pred in preds:
        pred_id = int(pred["label"].split("_")[-1])
        results.append("male" if pred_id == 1 else "female")

# Fill in ambiguous cases with model results
data.loc[ambiguous_mask, "gender"] = results

# Drop old columns
data = data.drop(columns=["male", "female"])
