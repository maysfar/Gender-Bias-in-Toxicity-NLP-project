import pandas as pd
import re


def _compile_gender_mask_regex_from_terms(terms):
    """
    Build a single case-insensitive regex that matches any term
    (plus simple plural/possessive tails like s/'s/’s),
    but avoids false positives inside contractions or larger words.
    """
    if not terms:
        return re.compile(r"(?!x)x", flags=re.IGNORECASE)  # never matches

    # longest-first to avoid partial overlaps (e.g., 'herself' before 'her')
    vocab = sorted({t.lower() for t in terms}, key=len, reverse=True)

    pattern = r'(?<!\w)(?:%s)(?=(?:\'s|’s|s|es)?(?![\w’\']))' % "|".join(map(re.escape, vocab))

    return re.compile(pattern, flags=re.IGNORECASE)

def comment_gender_mask(text, genderTerms, token="[GENDER]"):
    """
    String-level masking (like comment_gender_swap but replaces with a fixed token).
    """
    rx = _compile_gender_mask_regex_from_terms(genderTerms)
    return rx.sub(token, str(text))

def augment_with_gender_mask(df, genderTerms, text_col="comment", token="[GENDER]"):
    """
    Returns original df + masked duplicates for rows whose text changed.
    - Uses the provided `terms` list (derived from your gendered pairs).
    - Zeros subgroup cols (male/female) on augmented rows.
    - No new columns are added.
    """
    #assert text_col in df.columns, f"Missing column: {text_col}"
    rx = _compile_gender_mask_regex_from_terms(genderTerms)

    # 1) produce masked texts for all rows
    masked_texts = df[text_col].astype(str).apply(lambda t: rx.sub(token, t))

    # 2) find rows where text actually changed
    changed = ~masked_texts.eq(df[text_col].astype(str))
    if not changed.any():
        return df.copy()

    # 3) take only changed rows, set new text
    aug = df.loc[changed].copy()
    aug[text_col] = masked_texts.loc[changed].values


    # 5) return original + augmented rows
    return aug # return only the masked data

