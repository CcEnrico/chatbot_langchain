import requests
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    # Cerca nel database vettoriale utilizzando query risultati ottimi
    response = requests.post('http://localhost:5001/similarity_search', json={"query": [state["question"]]})
    retrieved_docs = response.json()["results"]
    return {"context": [Document(page_content=doc) for doc in retrieved_docs]}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = hub.pull("rlm/rag-prompt")
    
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    
    # Serializza i messaggi
    if isinstance(messages, list):
        serialized_messages = [
            message.to_dict() if hasattr(message, "to_dict") else str(message) for message in messages
        ]
    else:
        serialized_messages = messages.to_dict() if hasattr(messages, "to_dict") else str(messages)

    response = requests.post('http://localhost:5003/invoke', json={"messages": serialized_messages})
    return {"answer": response.json()["response"]}

def main():

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    # Execute the query on the LLM
    question = "You are a helpful assistant. Answer all questions to the best of your ability in Italian. raccontami Nabucodo'nosor"
    response = graph.invoke({"question": question})
    print(response["answer"])

if __name__ == "__main__":
    main()
