import pandas as pd
import json
from sklearn.model_selection import train_test_split

# import the augmentation functions
from gender_swap import augment_with_gender_swap
from gender_mask import augment_with_gender_mask

# import the dataset
data = pd.read_csv("../subdataset/subdataset.csv")

# splitting
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data = train_data.drop(columns=["gender"]) # we dont need it for training

# import the dictionary
with open("../gendered_word_pairs/gendered_word_pairs.json", "r", encoding="utf-8") as f:
    gendered_word_pairs = json.load(f)
    
# add reverse pairs IN PLACE
for k, v in list(gendered_word_pairs.items()):
    gendered_word_pairs[v] = k
    
# build terms from your existing dict (both keys and values)
genderTerms = sorted({str(k).lower() for k in gendered_word_pairs.keys()} |
               {str(v).lower() for v in gendered_word_pairs.values()})

gender_swap_data = augment_with_gender_swap(train_data, gendered_word_pairs)
gender_mask_data = augment_with_gender_mask(train_data, gendered_word_pairs)

#print(train_data.head())
#print(gender_swap_data.head())
#print(gender_mask_data.head())

# add the gender swapped rows to the original data
gender_swap_data = pd.concat([train_data, gender_swap_data], ignore_index=True)

gender_swap_data.to_csv("gender_swap_data.csv", index=False)
gender_mask_data.to_csv("gender_mask_data.csv", index=False)
