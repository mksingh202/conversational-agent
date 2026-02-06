from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from router import route_query
from rewriter import rewrite_question
from search import hybrid_search
from agents import generate_answer


class AgentState(TypedDict):
    question: str
    rewritten: str
    documents: list
    answer: str
    chat_history: List[str]
    route: str

def router_node(state):
    return {"route": route_query(state["question"])}

def retrieve_node(state):
    q = state.get("rewritten") or state["question"]
    return {"documents": hybrid_search(q)}

def answer_node(state):
    if not state["documents"]:
        return {"answer": "Not found in the document."}
    q = state.get("rewritten") or state["question"]
    docs_only = [d["doc"] for d in state["documents"]]
    return {"answer": generate_answer(q, docs_only)}
    # return {"answer": generate_answer(q, state["documents"])}

def rewrite_node(state):
    return {
        "rewritten": rewrite_question(state["chat_history"], state["question"])
    }

def refuse_node(state):
    return {"answer": "Not found in the document."}

graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("rewrite", rewrite_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("refuse", refuse_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda s: s["route"],
    {
        "FACTUAL": "retrieve",
        "FOLLOW_UP": "rewrite",
        "OUT_OF_SCOPE": "refuse",
    },
)

graph.add_edge("retrieve", "answer")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("answer", END)
graph.add_edge("refuse", END)

agent_graph = graph.compile()