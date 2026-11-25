"""
Step 2 - Train the content-based model and evaluate it.

Needs: interactions.csv and items.csv (created by 01_prepare_data.py) and recsys.py
Run:   python 02_evaluate.py
Output: console table + results.csv (copy the numbers into the report)
"""
import numpy as np
import pandas as pd

import recsys

K = 5
SEED = 42


def main():
    inter, items = recsys.load_data()
    train, test = recsys.split_by_user(inter)
    print(f"train interactions: {len(train):,} | test interactions: {len(test):,}")

    print("Building TF-IDF model ...")
    model = recsys.ContentRecommender(items, train)
    print(f"TF-IDF matrix: {model.tfidf.shape[0]:,} items x {model.tfidf.shape[1]:,} terms")

    print("Evaluating content-based model ...")
    content = recsys.evaluate(model, test, k=K)

    print("Evaluating random baseline ...")
    rng = np.random.default_rng(SEED)
    baseline = recsys.evaluate(
        model, test, k=K, recommender=lambda u, k: model.random_ids(u, k, rng)
    )

    results = pd.DataFrame(
        {"Content-based (TF-IDF + cosine)": content, "Random baseline": baseline}
    ).T
    print("\n", results.round(4).to_string())
    results.round(4).to_csv("results.csv")

    # Example output for a screenshot / sanity check
    sample_user = next(iter(model.liked))
    print(f"\nSample recommendations for user {sample_user}:")
    print(model.recommend(sample_user, K)[["rank", "title", "main_category", "similarity"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
