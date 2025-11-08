# indexing.py
import faiss
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import os
import time

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "job_recommender"
COLLECTION = "jobs"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
jobs_col = db[COLLECTION]



# Load same embedding model used in your app
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index global holder (in memory)
_faiss_index = None
_id_to_mongo_id = []

def get_index():
    global _faiss_index
    return _faiss_index

def get_id_map():
    global _id_to_mongo_id
    return _id_to_mongo_id

def build_faiss_index(batch_size=512):
    global _faiss_index, _id_to_mongo_id # Use the new names
    docs = list(jobs_col.find({}, {"_id": 1, "description_clean": 1, "description": 1}))
    print(f"📊 Found {len(docs)} docs in MongoDB")

    # fallback to 'description' if 'description_clean' missing
    texts = [d.get("description_clean") or d.get("description", "") for d in docs]
    ids = [str(d["_id"]) for d in docs]

    # Check sample
    if docs:
        print("🧩 Sample job:", docs[0])

    if not texts or all(t.strip() == "" for t in texts):
        print("⚠️ No valid text fields found for FAISS embedding.")
        _faiss_index = None
        _id_to_mongo_id = []
        return

    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    embeddings = np.array(embeddings).astype('float32')

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    _faiss_index = index
    _id_to_mongo_id = ids
    print(f"✅ Built FAISS index with {len(ids)} vectors.")

def add_job_to_index(job_doc):
    """
    job_doc: dict with fields: _id, description_clean, embedding(optional)
    """
    global _faiss_index, _id_to_mongo_id
    text = job_doc.get("description_clean","")
    emb = job_doc.get("embedding")
    if emb is None:
        emb = model.encode([text])[0]
    emb = np.array(emb).astype('float32')
    norm = np.linalg.norm(emb)
    if norm == 0:
        norm = 1.0
    emb = emb / norm

    if _faiss_index is None:
        dim = emb.shape[0]
        _faiss_index = faiss.IndexFlatIP(dim)
        _id_to_mongo_id = []

    _faiss_index.add(emb.reshape(1, -1))
    _id_to_mongo_id.append(str(job_doc["_id"]))
