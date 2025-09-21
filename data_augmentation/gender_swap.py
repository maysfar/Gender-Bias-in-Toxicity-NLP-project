import pandas as pd
import re

def comment_gender_swap(comment, gendered_word_pairs):
    # Sort keys by length (longest first) to avoid partial matches
    keys = sorted(gendered_word_pairs.keys(), key=len, reverse=True)

    # Build one regex that matches any of the gendered words
    pattern = re.compile(
        r'\b(?:' + '|'.join(map(re.escape, keys)) + r')\b',
        flags=re.IGNORECASE
    )

    def replace(m):
        word = m.group(0)
        base = gendered_word_pairs[word.lower()]

        if word.isupper():        # e.g., "HE" → "SHE"
            return base.upper()
        elif word[0].isupper():  # e.g., "He" → "She"
            return base.capitalize()
        else:                    # e.g., "he" → "she"
            return base.lower()

    return pattern.sub(replace, comment)

def augment_with_gender_swap(df,pairs):

    # 1) produce swapped texts for all rows
    swapped_texts = df["comment"].astype(str).apply(lambda t: comment_gender_swap(t, pairs))

    # 2) find rows where text actually changed
    changed = ~swapped_texts.eq(df["comment"].astype(str))
    if not changed.any():
        return df.copy()

    # 3) take only changed rows, set new text
    aug = df.loc[changed].copy()
    aug["comment"] = swapped_texts.loc[changed].values

    # 5) return augmented rows
    return aug

