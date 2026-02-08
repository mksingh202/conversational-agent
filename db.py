import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings


load_dotenv()


class VectorDB:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        engine = create_engine(
            f"postgresql+psycopg://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@"
            f"{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB_NAME')}"
        )
        self.store = PGVector(
            embeddings=self.embeddings,
            connection=engine,
            collection_name="adani_releases",
        )

    def add_documents(self, documents):
        self.store.add_documents(documents)

    def search(self, query, k=5):
        return self.store.similarity_search_with_score(query, k=k)
