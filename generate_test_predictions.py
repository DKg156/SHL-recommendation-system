import requests
import pandas as pd

API_URL = "https://dkg156-shl-assessment-recommender.hf.space/recommend"

def generate_predictions(path: str, out_csv: str = "test_predictions.csv", k: int = 10):
    df = pd.read_excel(path, sheet_name="Test-Set")

    rows = []

    for idx, row in df.iterrows():
        query = str(row.get("Query", "")).strip()
        if not query:
            continue

        try:
            resp = requests.post(
                API_URL,
                json={"query": query},
                timeout=120
            )
        except Exception as e:
            print(f"[{idx}] request failed: {e}")
            continue

        if resp.status_code != 200:
            print(f"[{idx}] API error {resp.status_code}: {resp.text[:200]}")
            continue

        data = resp.json()
        recs = data.get("recommended_assessments", []) or []
        predicted_urls = [r.get("url", "") for r in recs if r.get("url")]

        # Ensure we output at most k rows per query (spec says max 10)
        predicted_urls = predicted_urls[:k]

        for url in predicted_urls:
            rows.append({
                "Query": query,
                "Assessment_url": url
            })

    out_df = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    out_df.to_csv(out_csv, index=False)

    print(f"Saved predictions to {out_csv} (rows={len(out_df)})")


if __name__ == "__main__":
    generate_predictions("Gen_AI_Dataset.xlsx")
