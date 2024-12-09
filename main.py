import requests
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Definizione del tipo di dato State utilizzando TypedDict
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Funzione per recuperare documenti simili dal database vettoriale
def retrieve(state: State):
    # Invia una richiesta POST al server per effettuare una ricerca di similarità
    response = requests.post('http://localhost:5001/similarity_search', json={"query": [state["question"]]})
    # Ottieni i documenti recuperati dalla risposta del server
    retrieved_docs = response.json()["results"]
    # Restituisci i documenti come contesto
    return {"context": [Document(page_content=doc) for doc in retrieved_docs]}

# Funzione per generare una risposta utilizzando i documenti recuperati
def generate(state: State):
    # Combina il contenuto dei documenti recuperati in una singola stringa
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # Crea la struttura dei messaggi per il modello di linguaggio
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant specialized in llm systems. Provide detailed, thorough, and "
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

    # Invia i messaggi all'endpoint /invoke del server LLM
    response = requests.post('http://localhost:5001/invoke', json={"messages": messages})
    
    # Restituisci la risposta dell'assistente
    return {"answer": response.json().get("response", "No response from server.")}

# Funzione principale
def main():
    # Crea un grafo di stato e aggiungi le funzioni retrieve e generate come sequenza
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    # Aggiungi un bordo dal nodo di inizio alla funzione retrieve
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
        # Modalita normale output
        response = graph.invoke({"question": user_input}, config=config)
        print(f"AI: {response['answer']}")
       

if __name__ == "__main__":
    main()
