from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = """
    Classify the query into one label:
    FACTUAL, FOLLOW_UP, OUT_OF_SCOPE

    Query:
    {query}

    Return only the label.
"""

FINANCIAL_KEYWORDS = [
    "revenue", "income", "ebitda", "pat", "pbt", "h1", "h2", "q1", "q2", "fy", "quarter", "growth", "margin"
]

def route_query(query: str) -> str:
    q_lower = query.lower()
    if any(word in q_lower for word in FINANCIAL_KEYWORDS):
        return "FACTUAL"

    response = llm.invoke(PROMPT.format(query=query))
    label = response.content.strip().upper()
    return label if label in {"FACTUAL", "FOLLOW_UP", "OUT_OF_SCOPE"} else "FACTUAL"
