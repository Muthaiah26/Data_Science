from indexing import build_faiss_index, faiss_index
build_faiss_index()
print("Index size:", faiss_index.ntotal if faiss_index else 0)