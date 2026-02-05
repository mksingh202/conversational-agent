import os
import pdfplumber
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_csv(path: str) -> List[Document]:
    df = pd.read_csv(path)
    docs = []

    for idx, row in df.iterrows():
        content = " | ".join(f"{k}: {v}" for k, v in row.items())
        docs.append(
            Document(
                page_content=content,
                metadata={"source": os.path.basename(path), "row": idx},
            )
        )
    return docs

def load_pdf(path: str) -> List[Document]:
    docs = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": os.path.basename(path), "page": page_no},
                    )
                )
    return docs

def load_documents(path: str) -> List[Document]:
    if path.endswith(".csv"):
        return load_csv(path)
    if path.endswith(".pdf"):
        return load_pdf(path)
    raise ValueError("Unsupported file type")

def semantic_chunk(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for doc in documents:
        for i, text in enumerate(splitter.split_text(doc.page_content)):
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        # "chunk_id": f"{doc.metadata.get('page', doc.metadata.get('row'))}:{i}",
                        "chunk_id": i,
                    },
                )
            )
    return chunks
