FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build FAISS index from catalog.json
RUN python build_index.py

CMD uvicorn src.api:app --host 0.0.0.0 --port 7860
