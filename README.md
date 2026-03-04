# SHL-recommendation-system
Web-based RAG tool recommending assessments using SHL product catalog 

SHL Assessment Recommendation System

A semantic recommendation engine that suggests the most relevant SHL assessments based on a job description, hiring query, or job posting URL.

The system combines vector retrieval and LLM reranking to produce accurate and diverse recommendations. Candidate assessments are retrieved using embedding similarity and then reranked using an LLM to improve relevance.

---

Technologies Used

- Python
- FastAPI
- Streamlit
- FAISS
- Sentence Transformers
- Google Gemini Flash

System Architecture

The system follows a two-stage retrieval architecture, commonly used in modern search and recommendation systems.

User Query / Job Description / JD URL
                │
                ▼
        Query Processing
                │
                ▼
      Sentence Embedding Model
                │
                ▼
        FAISS Vector Search
       (Candidate Retrieval)
                │
                ▼
        LLM Reranking
        (Gemini Flash)
                │
                ▼
      Balanced Selection Logic
                │
                ▼
       Top-K Assessments Returned

This architecture allows fast candidate retrieval while leveraging LLM reasoning to refine the final ranking.

---

System Components

1. Query Processing

The system accepts three input types:

- Hiring query
- Job description text
- Job description URL

For URLs, the job description is extracted before generating embeddings.

---

2. Embedding Generation

Queries and assessment descriptions are embedded using a Sentence Transformer model.

Embeddings capture semantic meaning rather than relying on keyword overlap.

Example:

"software developer logical reasoning"

can match assessments related to problem solving or analytical thinking, even if the exact words differ.

---

3. FAISS Candidate Retrieval

All assessment embeddings are indexed using FAISS (Facebook AI Similarity Search).

This allows fast nearest-neighbor search to retrieve the most similar assessments.

Advantages:

- Efficient similarity search over many embeddings
- Scales well for large catalogs
- Low latency retrieval

The FAISS index is built using:

python build_index.py

---

4. LLM Reranking

The retrieved candidates are reranked using Gemini Flash.

The LLM evaluates the relevance between:

- job description
- assessment description

and produces a refined ranking.

This step improves recommendation quality because LLMs can reason about:

- skill requirements
- job responsibilities
- candidate attributes

beyond simple embedding similarity.

---

5. Balanced Recommendation Logic

For mixed queries, the recommender ensures diversity across assessment types.

Example categories:

- Knowledge-based tests
- Personality tests

The system allocates quotas to ensure both categories appear in the final recommendations.

This avoids returning results dominated by a single test type.

---

Project Structure

SHL-recommendation-system
│
├── data/                      # SHL catalog and FAISS index
│
├── docker/
│   ├── backend/
│   │   └── Dockerfile         # FastAPI container
│   └── frontend/
│       └── Dockerfile         # Streamlit container
│
├── src/
│   ├── api.py                 # FastAPI endpoints
│   ├── recommender.py         # Core recommendation logic
│   ├── llm_rerank.py          # LLM reranking module
│   └── jd_extract.py          # Job description extraction
│
├── build_index.py             # Builds FAISS index
├── scrape_catalog.py          # SHL catalog scraping
├── evaluate.py                # Training set evaluation
├── generate_test_predictions.py
│
├── app.py                     # Streamlit frontend
├── requirements.txt
└── README.md

---

API Endpoint

The backend exposes a FastAPI endpoint for generating recommendations.

Endpoint:

POST /recommend

Example request:

{
  "query": "Looking for software engineers with strong analytical and logical reasoning skills"
}

---

Deployment

The system is deployed as two services.

Streamlit app:

https://dkg156-recommender-frontend.hf.space/

API endpoints:

https://dkg156-shl-assessment-recommender.hf.space/health  (GET)
https://dkg156-shl-assessment-recommender.hf.space/recommend (POST)

---

Evaluation

Performance was evaluated and fine-tuned locally using Mean Recall@10 on the provided dataset.

Evaluation is performed using:

python evaluate.py

<img width="2528" height="1041" alt="Screenshot 2026-03-05 023250" src="https://github.com/user-attachments/assets/913def4b-1772-446d-8a7a-9d3cf374c2ac" />

<img width="1074" height="1170" alt="1" src="https://github.com/user-attachments/assets/cbfc57bd-b06a-4008-a129-345b42dbd5bb" />

<img width="1055" height="1172" alt="2" src="https://github.com/user-attachments/assets/18284ba1-2628-46f9-b0e6-f518be5dcd4f" />

<img width="1057" height="1177" alt="3" src="https://github.com/user-attachments/assets/f5c3301b-bd1e-4c3a-b393-2572382f40a6" />

<img width="1057" height="1169" alt="4" src="https://github.com/user-attachments/assets/e3e7c772-e37c-48f0-93d5-096cb3719310" />



Predictions for the final test set are generated using:

python generate_test_predictions.py

Output:

test_predictions.csv

---

Running the System Locally

Start the backend:

uvicorn src.api:app --host 0.0.0.0 --port 8000

Start the frontend:

streamlit run app.py

---


---


