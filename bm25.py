import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


nltk.download("punkt")

class BM25Index:
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [word_tokenize(d.page_content.lower()) for d in documents]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, k=5):
        scores = self.bm25.get_scores(word_tokenize(query.lower()))
        ranked = sorted(
            zip(self.documents, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:k]
