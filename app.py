import sys
from db import VectorDB
from search import init_bm25
from workflow import run_query
from ingestion import load_documents, semantic_chunk

vector_db = VectorDB()

def ingest(path):
    docs = semantic_chunk(load_documents(path))
    vector_db.add_documents(docs)
    init_bm25(docs)
    print(f"Ingested {len(docs)} chunks")


if __name__ == "__main__":
    ingest(sys.argv[1])
    history = []
    while True:
        q = input("\nQuestion: ")
        if q.lower() == "exit":
            break

        result = run_query(q, history)
        print("\nAnswer:", result["answer"])

        if 'documents' in result and 'not found in the document' not in result["answer"].lower():
            print("\nRetrieved Chunks:")
            print(f"{'Rank':<5} {'Citation':<12} {'RRF':<8} Snippet")
            print("-" * 125)
            for i, hit in enumerate(result["documents"], 1):
                doc = hit["doc"]
                snippet = doc.page_content[:100].replace("\n", " ") + "..."
                print(
                    f"{i:<5} "
                    f"p{doc.metadata.get('page')}:c{doc.metadata.get('chunk_id'):<8} "
                    f"{hit['rrf_score']:<8.4f} "
                    f"{snippet}"
                )
        history.append(f"User: {q}")
        history.append(f"Assistant: {result["answer"]}")
        history = history[-5:]
