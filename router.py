from llm_hub import openai_llm


PROMPT = """
    Classify the query into one label:
    FACTUAL, FOLLOW_UP, OUT_OF_SCOPE

    Query:
    {query}

    Return only the label.
"""

def route_query(query: str) -> str:
    response = openai_llm.invoke(PROMPT.format(query=query))
    label = response.content.strip().upper()

    return label if label in {"FACTUAL", "FOLLOW_UP", "OUT_OF_SCOPE"} else "FACTUAL"
