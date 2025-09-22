import pandas as pd
import json
import sys


# import the augmentation functions
from gender_swap import augment_with_gender_swap
from gender_mask import augment_with_gender_mask

# input training dataset
input_file = sys.argv[1]
data = pd.read_csv(input_file)


# import the dictionary
with open("../gendered_word_pairs/gendered_word_pairs.json", "r", encoding="utf-8") as f:
    gendered_word_pairs = json.load(f)
    
# add reverse pairs IN PLACE
for k, v in list(gendered_word_pairs.items()):
    gendered_word_pairs[v] = k
    
# build terms from your existing dict (both keys and values)
genderTerms = sorted({str(k).lower() for k in gendered_word_pairs.keys()} |
               {str(v).lower() for v in gendered_word_pairs.values()})

gender_swap_data = augment_with_gender_swap(data, gendered_word_pairs)
gender_mask_data = augment_with_gender_mask(data, gendered_word_pairs)

#print(data.head())
#print(gender_swap_data.head())
#print(gender_mask_data.head())



# add the gender swapped rows to the original data
gender_swap_data = pd.concat([data, gender_swap_data], ignore_index=True)


gender_swap_data.to_csv("gender_swap_data.csv", index=False)
gender_mask_data.to_csv("gender_mask_data.csv", index=False)


