import re
import os
import fitz
from langchain_core.documents import Document


PERIOD_PATTERN = r"(H\d-\d{2}|Q\d-\d{2}|FY\d{2})"

def load_documents(path):
    docs = []
    pdf = fitz.open(path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text("text")

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_num + 1,
                    "source": path
                }
            )
        )

    return docs

def extract_entity(source_path: str) -> str:
    filename = os.path.basename(source_path)
    return filename.split("_")[0]

def detect_scope(text: str) -> str:
    text_lower = text.lower()

    if "consolidated" in text_lower:
        return "consolidated"

    if "standalone" in text_lower:
        return "standalone"

    if "segment" in text_lower:
        return "segment"

    return "general"

def clean_text(text: str) -> str:
    if not text:
        return ""

    patterns_to_remove = [
        r"STRICTLY\s+CONFIDENTIAL",
        r"Home\s+outline\s+Hamburger\s+Menu\s+Icon\s+with\s+solid\s+fill\s*\d*",
        r"Hamburger\s+Menu\s+Icon\s+with\s+solid\s+fill\s*\d*",
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\b\d+\b(?=\s*$)", "", text)
    text = re.sub(r"[•●▪■]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def semantic_chunk(documents):
    chunks = []
    page_chunk_counter = {}

    for doc in documents:
        page = doc.metadata["page"]
        source = doc.metadata["source"]

        if page not in page_chunk_counter:
            page_chunk_counter[page] = 0

        entity = extract_entity(source)

        raw_text = clean_text(doc.page_content)
        scope = detect_scope(raw_text)
        blocks = re.split(r"\n{2,}", raw_text)
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            periods = re.findall(PERIOD_PATTERN, block)

            if periods:
                chunk_type = "financial"
            else:
                chunk_type = "commentary"

            chunks.append(
                Document(
                    page_content=raw_text,
                    metadata={
                        "page": page,
                        "source": source,
                        "entity": entity,
                        "scope": scope,
                        "chunk_type": chunk_type,
                        "chunk_id": page_chunk_counter[page]
                    }
                )
            )
            page_chunk_counter[page] += 1

    return chunks
