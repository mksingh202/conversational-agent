from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


SYSTEM_PROMPT = """
    Answer using ONLY the provided context.

    Rules:
    - Every factual sentence MUST end with a citation.
    - Citation format: [p<page>:c<chunk>]
    - If multiple chunks support a sentence, cite all of them.
    - Do NOT combine multiple facts under one citation.
    - Keep the answer short (5 sentences max).

    If the answer is not present in the context, reply exactly:
    "Not found in the document."

    If you cannot provide citations, you MUST refuse.
"""

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

    final_answer = raw_text.strip()
    if "[p" not in final_answer or final_answer.lower().startswith("not found"):
        return "Not found in the document."
    
    return final_answer
