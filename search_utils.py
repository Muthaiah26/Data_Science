# search_utils.py
import numpy as np
from indexing import (
    get_index, get_id_map, build_faiss_index, 
    model, jobs_col
)

def find_similar_jobs_by_embedding(text, top_k=20):
    faiss_index = get_index()
    id_to_mongo_id = get_id_map()

    # --- Step 1: Ensure FAISS is loaded ---
    if faiss_index is None or id_to_mongo_id is None or len(id_to_mongo_id) == 0:
        print("⚙️ FAISS index missing in memory — rebuilding now...")
        build_faiss_index()
        faiss_index = get_index()
        id_to_mongo_id = get_id_map()
        if faiss_index is None or faiss_index.ntotal == 0:
            print("❌ FAISS rebuild failed — no vectors available.")
            return []

    # --- Step 2: Compute embedding ---
    emb = model.encode([text])[0].astype('float32')
    emb = emb / (np.linalg.norm(emb) + 1e-12)

    # --- Step 3: Search FAISS ---
    D, I = faiss_index.search(emb.reshape(1, -1), top_k)
    indices = I[0].tolist()
    scores = D[0].tolist()

    results = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(id_to_mongo_id):
            continue
        mongo_id = id_to_mongo_id[idx]
        job = jobs_col.find_one({"_id": mongo_id})
        if job:
            job_copy = job.copy()
            job_copy["vector_score"] = float(score)
            results.append(job_copy)

    print(f"✅ Found {len(results)} similar jobs via FAISS.")
    return results
