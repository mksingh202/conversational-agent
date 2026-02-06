import re
import os
import camelot
import pdfplumber
import pandas as pd
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SECTION_PATTERN = re.compile(
    r"\n(?=[A-Z][A-Za-z &]{3,}\n)"
)

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

# def load_pdf(path: str) -> List[Document]:
#     docs = []
#     with pdfplumber.open(path) as pdf:
#         for page_no, page in enumerate(pdf.pages, start=1):
#             text = page.extract_text()
#             if text:
#                 docs.append(
#                     Document(
#                         page_content=text,
#                         metadata={"source": os.path.basename(path), "page": page_no},
#                     )
#                 )
#     return docs

def load_pdf(path: str) -> List[Document]:
    docs: List[Document] = []
    source = os.path.basename(path)

    # ---- 1. Extract normal text using pdfplumber ----
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": page_no,
                            "type": "text",
                        },
                    )
                )

    # ---- 2. Extract tables using Camelot ----
    try:
        tables = camelot.read_pdf(
            path,
            pages="all",
            flavor="lattice",  # use "stream" if lattice fails
        )

        for i, table in enumerate(tables):
            df = table.df
            table_text = df.to_csv(index=False)

            docs.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "source": source,
                        "page": table.page,
                        "type": "table",
                        "table_index": i,
                    },
                )
            )

    except Exception as e:
        # Fail gracefully (important for pipelines)
        print(f"[WARN] Camelot failed on {path}: {e}")

    return docs

def load_documents(path: str) -> List[Document]:
    if path.endswith(".csv"):
        return load_csv(path)
    if path.endswith(".pdf"):
        return load_pdf(path)
    raise ValueError("Unsupported file type")


def semantic_chunk(documents: List[Document]) -> List[Document]:
    chunks = []
    for doc in documents:
        text = doc.page_content
        
        # 1. Split by semantic sections
        sections = SECTION_PATTERN.split(text)
        for sec_id, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # 2. Keep tables intact
            # if "TABLE DATA:" in section:
            if doc.metadata.get("type", '') == "table":
                chunks.append(
                    Document(
                        page_content=section,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "table",
                            "chunk_id": sec_id
                        },
                    )
                )
                continue

            # 3. Light size control inside semantic blocks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". "],
            )

            for i, sub_chunk in enumerate(splitter.split_text(section)):
                chunks.append(
                    Document(
                        page_content=sub_chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_type": "text",
                            "chunk_id": f"{sec_id}.{i}"
                        },
                    )
                )
    return chunks
