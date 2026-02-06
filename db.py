import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector


load_dotenv()

CONNECTION_STRING = f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB_NAME')}"
COLLECTION_NAME = "adani_releases"

class VectorDB:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        self.store = PGVector(
            connection_string=CONNECTION_STRING,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            collection_metadata={
                "domain": "finance",
                "version": "FY26"
            }
        )

    def add_documents(self, documents):
        self.store.add_documents(documents)

    def search(self, query, k=5):
        return self.store.similarity_search_with_score(query, k=k)
