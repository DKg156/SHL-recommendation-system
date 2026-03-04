import requests
import pandas as pd
from collections import defaultdict

API_URL = "http://localhost:8000/recommend"

from urllib.parse import urlsplit, urlunsplit

def canon_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u

    # normalize SHL base path differences
    u = u.replace(
        "https://www.shl.com/solutions/products/product-catalog/view/",
        "https://www.shl.com/products/product-catalog/view/",
    )

    # remove querystring + fragment, normalize trailing slash
    parts = urlsplit(u)
    clean = urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))
    return clean


def recall_at_k(predicted, ground_truth, k=10):
    predicted_k = predicted[:k]
    hits = len(set(predicted_k) & set(ground_truth))
    if len(ground_truth) == 0:
        return 0.0
    return hits / len(ground_truth)


def evaluate(train_path: str):
    df = pd.read_excel(train_path, sheet_name="Train-Set")

    # Group ground truth URLs per query
    gt_map = defaultdict(list)
    for _, row in df.iterrows():
        query = row["Query"].strip()
        url = canon_url(row["Assessment_url"])
        gt_map[query].append(url)

    recalls = []

    for idx, (query, ground_truth_urls) in enumerate(gt_map.items(), 1):

        response = requests.post(
            API_URL,
            json={"query": query},
            timeout=240
        )

        if response.status_code != 200:
            print(f"API error for query {idx}")
            continue

        recs = response.json()["recommended_assessments"]
        predicted_urls = [canon_url(r["url"]) for r in recs]

        r10 = recall_at_k(predicted_urls, ground_truth_urls, k=10)
        recalls.append(r10)

        print(f"\n=== Query {idx} ===")
        print("Recall@10:", round(r10, 2))

    mean_recall = sum(recalls) / len(recalls)
    print("\nMean Recall@10:", round(mean_recall, 4))


if __name__ == "__main__":
    evaluate("Gen_AI_Dataset.xlsx")
