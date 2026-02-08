from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

REWRITE_PROMPT = """
    Given the conversation history and the follow-up question,
    rewrite the question to be a standalone question.

    Conversation History:
    {history}

    Follow-up Question:
    {question}

    Standalone Question:
"""

def rewrite_question(chat_history, question):
    history_lines = []
    for h in chat_history:
        if isinstance(h, dict):
            user = h.get("user")
            assistant = h.get("assistant")
            if user:
                history_lines.append(f"User: {user}")
            if assistant:
                history_lines.append(f"Assistant: {assistant}")

        elif isinstance(h, (tuple, list)) and len(h) == 2:
            history_lines.append(f"User: {h[0]}")
            history_lines.append(f"Assistant: {h[1]}")

        elif isinstance(h, str):
            history_lines.append(h)

    history_text = "\n".join(history_lines)

    prompt = REWRITE_PROMPT.format(
        history=history_text,
        question=question
    )

    response = llm.invoke(prompt)
    rewritten = response.content

    if not isinstance(rewritten, str) or not rewritten.strip():
        return question

    return rewritten.strip()
