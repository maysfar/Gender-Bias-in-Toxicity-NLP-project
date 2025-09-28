import pandas as pd
import re

def comment_gender_swap(comment, gendered_word_pairs):
    # Sort keys by length (longest first) to avoid partial matches
    keys = sorted(gendered_word_pairs.keys(), key=len, reverse=True)

    pattern = re.compile(
        r'(?<!\w)(?:' + '|'.join(map(re.escape, keys)) + r')(?=(?:\'s|’s)?(?![\w’\']))',
        flags=re.IGNORECASE
    )

    def replace(m):
        word = m.group(0)

        # Strip possessive for dictionary lookup
        base_word = re.sub(r"(\'s|’s)$", "", word, flags=re.IGNORECASE)

        if base_word.lower() not in gendered_word_pairs:
            return word  # no replacement

        base = gendered_word_pairs[base_word.lower()]

        # Preserve casing
        if word.isupper():
            replacement = base.upper()
        elif word[0].isupper():
            replacement = base.capitalize()
        else:
            replacement = base.lower()

        # Reattach possessive if needed
        if word.lower().endswith(("'s", "’s")):
            replacement += "'s"

        return replacement

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


