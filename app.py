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
        q = input("\nQuestion (exit): ")
        if q.lower() == "exit":
            break

        answer = run_query(q, history)
        print("\nAnswer:\n", answer)

        history.append(f"User: {q}")
        history.append(f"Assistant: {answer}")
