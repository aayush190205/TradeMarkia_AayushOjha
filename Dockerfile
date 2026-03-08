FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --default-timeout=2000 torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --default-timeout=2000 --no-deps sentence-transformers==2.2.2

COPY requirements.txt .
RUN pip install --default-timeout=2000 -r requirements.txt

# THE HACK: Install Pillow completely separate so we don't break the cache above!
RUN pip install --default-timeout=2000 Pillow==10.0.1

COPY . .

EXPOSE 8000
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]