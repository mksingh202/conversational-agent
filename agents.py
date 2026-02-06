import re
import nltk
from nltk.tokenize import sent_tokenize
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

nltk.download("punkt")

SYSTEM_PROMPT = """
    Answer the question using the provided context.
    DO NOT include citations, page numbers, or bracketed references in your answer.
    Citations will be added separately.

    If the answer is not present, say:
    "Not found in the document."
"""

CITATION_PATTERN = re.compile(r"\[p\d+:[^\]]+\]")

def generate_answer(question, documents):
    context = "\n".join(
        f"[p{d.metadata.get('page')}:c{d.metadata.get('chunk_id')}] {d.page_content}"
        for d in documents
    )

    prompt = f"""
        {SYSTEM_PROMPT}

        Context:
        {context}

        Question:
        {question}
    """

    response = llm.invoke(prompt)
    raw_text = response.content

    if not isinstance(raw_text, str) or not raw_text.strip():
        return "Not found in the document."

    clean_text = re.sub(CITATION_PATTERN, "", raw_text).strip()
    sentences = sent_tokenize(clean_text)

    citations = list(dict.fromkeys(
        f"[p{d.metadata.get('page')}:c{d.metadata.get('chunk_id')}]"
        for d in documents
        if d.metadata.get("page") is not None
    ))[:1]

    final_answer = " ".join(sentence.strip() for sentence in sentences)
    if final_answer.lower().startswith("not found"):
        return final_answer
    
    if citations:
        final_answer = f"{final_answer} {citations[0]}"
    
    return final_answer