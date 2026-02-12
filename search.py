from db import VectorDB
from bm25 import BM25Index


vector_db = VectorDB()
bm25_index = None

def init_bm25(documents):
    global bm25_index
    bm25_index = BM25Index(documents)

def reciprocal_rank_fusion(vector_hits, bm25_hits, k=60, top_n=5):
    seen = set()
    results = []

    fused_scores = {}
    id_to_doc = {}

    for rank, (doc, _) in enumerate(vector_hits):
        doc_id = getattr(doc, "id", None) or id(doc)
        id_to_doc[doc_id] = doc

        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, (doc, _) in enumerate(bm25_hits):
        doc_id = getattr(doc, "id", None) or id(doc)
        id_to_doc[doc_id] = doc

        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    sorted_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for doc_id, score in sorted_results:
        doc = id_to_doc[doc_id]

        citation_key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("chunk_id"),
        )

        if citation_key not in seen:
            seen.add(citation_key)
            results.append({"doc": doc, "rrf_score": score})

    return results[:top_n]


def hybrid_search(query: str, k: int = 5):
    vector_hits = vector_db.search(query, k=k)
    bm25_hits = bm25_index.search(query, k=k)

    return reciprocal_rank_fusion(vector_hits, bm25_hits, top_n=k)
