from db import VectorDB
from bm25 import BM25Index

vector_db = VectorDB()
bm25_index = None

def init_bm25(documents):
    global bm25_index
    bm25_index = BM25Index(documents)

# def reciprocal_rank_fusion(vector_hits, bm25_hits, k=60, top_n=5):
#     fused_scores = {}
#     doc_store = {}

#     # Vector results
#     for rank, (doc, _) in enumerate(vector_hits):
#         doc_id = id(doc)
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (k + rank)
#         doc_store[doc_id] = doc

#     # BM25 results
#     for rank, (doc, _) in enumerate(bm25_hits):
#         doc_id = id(doc)
#         fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (k + rank)
#         doc_store[doc_id] = doc

#     ranked = sorted(
#         fused_scores.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return [doc_store[doc_id] for doc_id, _ in ranked[:top_n]]

def reciprocal_rank_fusion(vector_hits, bm25_hits, k=60, top_n=5):
    fused_scores = {}
    doc_store = {}

    for rank, (doc, score) in enumerate(vector_hits):
        doc_id = id(doc)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (k + rank)
        doc_store[doc_id] = (doc, score)

    for rank, (doc, score) in enumerate(bm25_hits):
        doc_id = id(doc)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (k + rank)
        doc_store[doc_id] = (doc, score)

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {
            "doc": doc_store[doc_id][0],
            "retrieval_score": doc_store[doc_id][1],
            "rrf_score": fused
        }
        for doc_id, fused in ranked[:top_n]
    ]

def hybrid_search(query: str, k: int = 5):
    vector_hits = vector_db.search(query, k=k)
    bm25_hits = bm25_index.search(query, k=k)

    return reciprocal_rank_fusion(vector_hits, bm25_hits, top_n=k)
