from graph import agent_graph


def run_query(question: str, chat_history: list):
    result = agent_graph.invoke(
        {"question": question, "chat_history": chat_history}
    )
    return result["answer"]
