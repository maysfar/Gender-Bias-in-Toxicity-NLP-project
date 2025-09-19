#data cleaning functions

"""Strip leading/trailing whitespace and normalize Unicode (NFC)"""

import unicodedata
import pandas as pd
import re


def strip_and_normalize_nfc(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    return unicodedata.normalize("NFC", text)

"""Replace URLs with <URL> and user handles with <USER>"""

URL_RE  = re.compile(r'((?:https?://|http?://|www\.)\S+)', flags=re.IGNORECASE)
USER_RE = re.compile(r'(?<!\w)@\w+')

def replace_urls_and_users(text: str) -> str:
    text = URL_RE.sub("<URL>", text)
    text = USER_RE.sub("<USER>", text)
    return text

"""Remove \n and similar, then collapse excess whitespace to single spaces"""

WS_RE = re.compile(r"\s+")

def remove_newlines_and_collapse_ws(text: str) -> str:
    # replace any whitespace run (including \n, \t) with a single space
    return WS_RE.sub(" ", text).strip()

#data augmentaion helpers and functions
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

def comment_gender_mask(text, genderTerms, token="[GENDER]"):
    """
    String-level masking (like comment_gender_swap but replaces with a fixed token).
    """
    rx = _compile_gender_mask_regex_from_terms(genderTerms)
    return rx.sub(token, str(text))

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

    # 4) swap numeric male/female values on those rows
    tmp = aug["male"].copy()
    aug["male"]   = aug["female"].values
    aug["female"] = tmp.values

    # 5) return original + augmented rows
    return pd.concat([df, aug], ignore_index=True)

def _compile_gender_mask_regex_from_terms(terms):
    """
    Build a single case-insensitive regex that matches any term
    (plus simple plural/possessive tails like s/es/'s).
    """
    if not terms:
        return re.compile(r"(?!x)x", flags=re.IGNORECASE)  # never matches
    # longest-first to avoid partial overlaps (e.g., 'herself' before 'her')
    vocab = sorted({t.lower() for t in terms}, key=len, reverse=True)
    pattern = r"\b(?:%s)(?:['’]s|s|es)?\b" % "|".join(map(re.escape, vocab))
    return re.compile(pattern, flags=re.IGNORECASE)

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

    # 4) zero numeric male/female values on those rows (if present)
    for col in ("male", "female"):
        if col in aug.columns:
            aug[col] = 0

    # 5) return original + augmented rows
    return pd.concat([df, aug], ignore_index=True) #Not sure IF we want to have only masked or both


gendered_word_pairs = {
    # Pronouns
    "he": "she",
    "him": "her",
    "his": "hers",
    "himself": "herself",

    # Common people words
    "man": "woman",
    "men": "women",
    "boy": "girl",
    "boys": "girls",
    "guy": "girl",
    "guys": "girls",
    "dude": "chick",   # casual/slang
    "bro": "sis",
    "gentleman": "lady",

    # Family terms
    "dad": "mom",
    "father": "mother",
    "son": "daughter",
    "brother": "sister",
    "uncle": "aunt",
    "husband": "wife",
    "boyfriend": "girlfriend",
    "bf": "gf",

    # Roles / references (popular in slang)
    "king": "queen",
    "prince": "princess"
}

# add reverse pairs IN PLACE
for k, v in list(gendered_word_pairs.items()):
    gendered_word_pairs[v] = k

# Build terms from your existing dict (both keys and values)
genderTerms = sorted({str(k).lower() for k in gendered_word_pairs.keys()} |
               {str(v).lower() for v in gendered_word_pairs.values()})
