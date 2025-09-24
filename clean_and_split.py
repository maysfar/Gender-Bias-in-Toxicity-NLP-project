#!/usr/bin/env python3
import argparse, os, re, unicodedata
import pandas as pd
from sklearn.model_selection import train_test_split

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
WS_RE  = re.compile(r'\s+')

def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = URL_RE.sub(" <URL> ", s)
    s = s.replace('\r', '\n')
    s = WS_RE.sub(' ', s).strip()
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to CSV (e.g., all_data.csv)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--text_col", default="comment_text")
    ap.add_argument("--label_col", default="target")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    if args.text_col not in df.columns:
        raise SystemExit(f"Text column '{args.text_col}' not found. Available: {list(df.columns)[:20]}")

    # Basic cleaning
    df[args.text_col] = df[args.text_col].astype(str).map(clean_text)
    df = df[df[args.text_col].str.len() > 0]
    df = df.drop_duplicates(subset=[args.text_col])

    df = df[["comment_text", "target", "male", "female"]]

    # Keep label if present; otherwise split without strat
    stratify = None
    if args.label_col in df.columns:
        # Jigsaw target is continuous; bin for stratified split
        try:
            bins = pd.qcut(df[args.label_col], q=10, duplicates="drop")
            stratify = bins
        except Exception:
            stratify = None
    # Save cleaned full
    cleaned_path = os.path.join(args.outdir, "cleaned_full.csv")
    df.to_csv(cleaned_path, index=False)

    # Split
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, stratify=stratify
    )

    train_path = os.path.join(args.outdir, "train_80.csv")
    test_path  = os.path.join(args.outdir, "test_20.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Tiny sanity print
    print(f"Saved:\n  {cleaned_path} ({len(df)})\n  {train_path} ({len(train_df)})\n  {test_path} ({len(test_df)})")

if __name__ == "__main__":
    main()
