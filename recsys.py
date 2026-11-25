"""
recsys.py - Content-based recommender (TF-IDF + cosine similarity).

This module is the ONE shared implementation used by:
    - 02_evaluate.py  (metrics for the report)
    - the GUI         (so the GUI uses exactly the method described in the paper)

Pipeline:
    interactions.csv + items.csv
      -> per-user 80/20 split (by time)
      -> text cleaning
      -> TF-IDF vectors (fitted on TRAIN items only)
      -> user profile = rating-weighted average of the TF-IDF vectors of liked items
      -> cosine similarity(profile, every item)
      -> drop items the user already interacted with -> top-5
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

LIKE_THRESHOLD = 4.0   # rating >= 4  ->  "positive interaction"
TEST_RATIO = 0.2       # last 20% of each user's interactions go to the test set
TOP_K = 5


# ------------------------------ data & split ---------------------------------
def clean_text(text):
    """Lowercase, strip HTML tags and punctuation, collapse whitespace."""
    text = re.sub(r"<[^>]+>", " ", str(text).lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(interactions_path="interactions.csv", items_path="items.csv"):
    inter = pd.read_csv(interactions_path)
    items = pd.read_csv(items_path).fillna("")
    items = items.drop_duplicates("parent_asin").reset_index(drop=True)
    inter = inter[inter.parent_asin.isin(set(items.parent_asin))]
    return inter, items


def split_by_user(inter, test_ratio=TEST_RATIO):
    """Per-user temporal split: oldest ~80% -> train, newest ~20% -> test."""
    inter = inter.sort_values(["user_id", "timestamp"])
    rank = inter.groupby("user_id").cumcount()
    size = inter.groupby("user_id")["parent_asin"].transform("size")
    n_test = np.maximum(1, (size * test_ratio).round().astype(int))
    is_test = rank >= (size - n_test)
    return inter[~is_test].copy(), inter[is_test].copy()


# --------------------------------- model -------------------------------------
class ContentRecommender:
    def __init__(self, items, train):
        self.items = items.reset_index(drop=True)
        self.index_of = {pid: i for i, pid in enumerate(self.items.parent_asin)}
        texts = self.items.text.map(clean_text)

        # Fit the vocabulary / IDF on TRAIN items only, then vectorize every item.
        train_idx = sorted({self.index_of[p] for p in train.parent_asin})
        self.vectorizer = TfidfVectorizer(
            stop_words="english", min_df=2, max_df=0.5,
            ngram_range=(1, 2), max_features=50000, sublinear_tf=True,
        )
        self.vectorizer.fit(texts.iloc[train_idx])
        self.tfidf = self.vectorizer.transform(texts)      # rows are L2-normalised

        self.seen = train.groupby("user_id")["parent_asin"].apply(set).to_dict()
        liked = train[train.rating >= LIKE_THRESHOLD]
        self.liked = {
            u: (np.array([self.index_of[p] for p in g.parent_asin]),
                g.rating.to_numpy(dtype=float))
            for u, g in liked.groupby("user_id")
        }

    # ---- similarity helpers ----
    def _scores_from_vector(self, vec):
        """Cosine similarity between one (dense) vector and every item."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.zeros(self.tfidf.shape[0])
        return np.asarray(self.tfidf @ (vec / norm)).ravel()

    def _top_k(self, scores, exclude_ids, k):
        scores = scores.copy()
        for pid in exclude_ids:
            if pid in self.index_of:
                scores[self.index_of[pid]] = -np.inf
        top = np.argsort(-scores)[:k]
        return [(self.items.parent_asin[i], float(scores[i])) for i in top]

    # ---- user-based recommendation ----
    def recommend_ids(self, user_id, k=TOP_K):
        if user_id not in self.seen:
            raise KeyError(f"Unknown user: {user_id}")
        if user_id not in self.liked:
            raise ValueError("This user has no positively rated items in the training data.")
        idx, ratings = self.liked[user_id]
        profile = np.asarray(
            self.tfidf[idx].multiply(ratings[:, None]).sum(axis=0)
        ).ravel() / ratings.sum()
        scores = self._scores_from_vector(profile)
        return self._top_k(scores, self.seen[user_id], k)

    def recommend(self, user_id, k=TOP_K):
        """Top-k recommendations as a DataFrame with the main content features."""
        return self._to_frame(self.recommend_ids(user_id, k))

    # ---- item-to-item recommendation ----
    def similar_items(self, parent_asin, k=TOP_K):
        if parent_asin not in self.index_of:
            raise KeyError(f"Unknown item: {parent_asin}")
        vec = self.tfidf[self.index_of[parent_asin]].toarray().ravel()
        scores = self._scores_from_vector(vec)
        return self._to_frame(self._top_k(scores, [parent_asin], k))

    def _to_frame(self, recs):
        rows = []
        for rank, (pid, score) in enumerate(recs, start=1):
            row = self.items.iloc[self.index_of[pid]]
            rows.append({"rank": rank, "parent_asin": pid, "title": row.title,
                         "main_category": row.main_category, "store": row.store,
                         "similarity": round(score, 4)})
        return pd.DataFrame(rows)

    def random_ids(self, user_id, k=TOP_K, rng=None):
        """Baseline: k random unseen items (for comparison in the report)."""
        rng = rng or np.random.default_rng()
        pool = [p for p in self.items.parent_asin if p not in self.seen[user_id]]
        return [(p, 0.0) for p in rng.choice(pool, size=k, replace=False)]


# -------------------------------- evaluation ---------------------------------
def evaluate(model, test, k=TOP_K, recommender=None):
    """Precision@k, Recall@k and MAP@k over users with >=1 relevant test item.

    relevant = test items the user rated >= LIKE_THRESHOLD
    AP@k     = sum_i( precision@i * hit_i ) / min(#relevant, k)
    """
    recommender = recommender or model.recommend_ids
    relevant = (test[test.rating >= LIKE_THRESHOLD]
                .groupby("user_id")["parent_asin"].apply(set))
    precisions, recalls, aps = [], [], []
    for user, rel in relevant.items():
        if user not in model.liked:
            continue
        recs = [pid for pid, _ in recommender(user, k)]
        hits = np.array([1 if pid in rel else 0 for pid in recs])
        n_hits = hits.sum()
        precisions.append(n_hits / k)
        recalls.append(n_hits / len(rel))
        prec_at_i = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        aps.append((prec_at_i * hits).sum() / min(len(rel), k))
    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"MAP@{k}": float(np.mean(aps)),
        "users_evaluated": len(precisions),
    }
