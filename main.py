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
from langgraph.checkpoint.memory import MemorySaver

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
    # Combine retrieved document content into a single string
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Create the message structure for the LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specialized in embedded systems. Provide detailed, thorough, and "
                "well-structured answers in Italian, including examples, explanations, and step-by-step guidance."
            )
        },
        {
            "role": "assistant",
            "content": f"Ecco alcune informazioni di contesto utili:\n\n{docs_content}"
        },
        {
            "role": "user",
            "content": state["question"]
        }
    ]

    # Send the messages to the /invoke endpoint of the LLM server
    response = requests.post('http://localhost:5003/invoke', json={"messages": messages})
    
    # Return the assistant's response
    return {"answer": response.json().get("response", "No response from server.")}


def main():

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")

    # Add memory
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "chat1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        response = graph.invoke({"question": user_input}, config)
        print(f"AI: {response['answer']}")

if __name__ == "__main__":
    main()
