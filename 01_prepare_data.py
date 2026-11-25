"""
Step 1 - Build a manageable subset of Amazon Reviews 2023 (Beauty_and_Personal_Care).

Inputs (put next to this script, .jsonl or .jsonl.gz both work):
    Beauty_and_Personal_Care.jsonl        <- REVIEW file (user ratings)
    meta_Beauty_and_Personal_Care.jsonl   <- META file   (product content)

Outputs:
    interactions.csv   user_id, parent_asin, rating, timestamp
    items.csv          parent_asin, title, main_category, store, text

Why a subset? The full category has ~24M ratings and ~1M items. We keep a
deterministic random sample of USERS (all of their reviews), then apply a
k-core filter so every user/item has enough interactions to evaluate on.
"""
import gzip
import json
import zlib

import pandas as pd

# ----------------------------- settings -------------------------------------
REVIEW_FILE = "Beauty_and_Personal_Care.jsonl.gz"
META_FILE = "meta_Beauty_and_Personal_Care.jsonl"

USER_SAMPLE_PERCENT = 5      # keep ~5% of users (hash based, reproducible)
MIN_INTERACTIONS = 5         # k-core: min reviews per user AND per item
MAX_USERS = 5000             # cap the number of users after filtering
MAX_LINES = None             # e.g. 500_000 for a quick test run, None = all
SEED = 42
# -----------------------------------------------------------------------------


def open_any(path):
    """Open plain or gzip-compressed text files."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def keep_user(user_id):
    """Deterministic sampling: same user is always kept or always dropped."""
    return zlib.crc32(user_id.encode("utf-8")) % 100 < USER_SAMPLE_PERCENT


def load_reviews():
    rows = []
    with open_any(REVIEW_FILE) as fp:
        for i, line in enumerate(fp):
            if MAX_LINES and i >= MAX_LINES:
                break
            if i % 1_000_000 == 0:
                print(f"  reviews read: {i:,}")
            r = json.loads(line)
            uid = r.get("user_id")
            if not uid or not keep_user(uid):
                continue
            rows.append((
                uid,
                r.get("parent_asin"),
                r.get("rating"),
                r.get("timestamp") or r.get("sort_timestamp"),
            ))
    df = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    print(f"Sampled interactions: {len(df):,}")
    return df


def k_core(df, k):
    """Repeat until every user and every item has at least k interactions."""
    while True:
        before = len(df)
        df = df[df.groupby("user_id")["parent_asin"].transform("size") >= k]
        df = df[df.groupby("parent_asin")["user_id"].transform("size") >= k]
        if len(df) == before:
            return df


def to_text(value):
    """meta fields can be str, list or None -> always return one string."""
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def load_meta(item_ids):
    items = {}
    with open_any(META_FILE) as fp:
        for i, line in enumerate(fp):
            if i % 200_000 == 0:
                print(f"  meta read: {i:,}")
            m = json.loads(line)
            pid = m.get("parent_asin")
            if pid not in item_ids:
                continue
            text = " ".join([
                to_text(m.get("title")),
                to_text(m.get("features")),
                to_text(m.get("description")),
                to_text(m.get("categories")),
                to_text(m.get("store")),
            ])
            items[pid] = {
                "parent_asin": pid,
                "title": to_text(m.get("title")),
                "main_category": to_text(m.get("main_category")),
                "store": to_text(m.get("store")),
                "text": text,
            }
    return pd.DataFrame(items.values())


def main():
    print("1) Reading reviews ...")
    df = load_reviews()

    print("2) Removing duplicate (user, item) pairs and applying k-core ...")
    df = df.dropna(subset=["user_id", "parent_asin", "rating"])
    df = df.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last")
    df = k_core(df, MIN_INTERACTIONS)
    print(f"  after k-core: {len(df):,} rows, {df.user_id.nunique():,} users, "
          f"{df.parent_asin.nunique():,} items")

    if df.user_id.nunique() > MAX_USERS:
        users = pd.Series(df.user_id.unique()).sample(MAX_USERS, random_state=SEED)
        df = df[df.user_id.isin(set(users))]

    print("3) Reading item metadata ...")
    items = load_meta(set(df.parent_asin))

    # Some items have no metadata -> drop them (report this in the paper!)
    missing = df.parent_asin.nunique() - len(items)
    df = df[df.parent_asin.isin(set(items.parent_asin))]
    items = items[items.text.str.strip() != ""]
    df = df[df.parent_asin.isin(set(items.parent_asin))]
    print(f"  items without usable metadata dropped: {missing:,}")

    df.to_csv("interactions.csv", index=False)
    items.to_csv("items.csv", index=False)

    print("\nFINAL DATASET SUMMARY (use in the report's dataset table)")
    print(f"  users:        {df.user_id.nunique():,}")
    print(f"  items:        {df.parent_asin.nunique():,}")
    print(f"  interactions: {len(df):,}")
    print(f"  rating mean:  {df.rating.mean():.2f}")
    print(f"  sparsity:     {1 - len(df) / (df.user_id.nunique() * df.parent_asin.nunique()):.4%}")


if __name__ == "__main__":
    main()
