from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = """
    Classify the query into one label:
    FACTUAL, FOLLOW_UP, OUT_OF_SCOPE

    Query:
    {query}

    Return only the label.
"""

def route_query(query: str) -> str:
    response = llm.invoke(PROMPT.format(query=query))
    label = response.content.strip().upper()
    return label if label in {"FACTUAL", "FOLLOW_UP", "OUT_OF_SCOPE"} else "FACTUAL"
