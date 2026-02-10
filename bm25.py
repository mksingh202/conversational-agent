import re
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [self.tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def search(self, query, k=5):
        scores = self.bm25.get_scores(self.tokenize(query))
        ranked = sorted(
            zip(self.documents, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:k]
