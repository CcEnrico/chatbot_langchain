from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from typing import List, TypedDict
import os
import requests

# Variabili d'ambiente
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Definisci lo Stato con cronologia dei messaggi e contesti separati
class State(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    general_context: str
    document_context: List[Document]

# Inizializza il LLM
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2000, temperature=0.6)

# Configura il trimmer
trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Funzione per recuperare documenti simili dal database vettoriale
def retrieve(state: State):

    # Pulisci il contesto dei documenti
    state["document_context"] = []

    # Assicurati che ci sia un messaggio prima di tentare la ricerca
    if not state["messages"]:
        print("Nessun messaggio disponibile per effettuare la ricerca.")
        return state

    # Invia una richiesta POST al server per effettuare una ricerca di similarità
    try:
        response = requests.post(
            'http://localhost:5001/similarity_search',
            json={"query": [state["messages"][-1].content]}
        )
        response.raise_for_status()  # Assicurati che la richiesta abbia avuto successo
        retrieved_docs = response.json().get("results", [])
        # Aggiorna il contesto dei documenti nello stato
        state["document_context"] = [Document(page_content=doc) for doc in retrieved_docs]
    except requests.exceptions.RequestException as e:
        print(f"Errore durante il recupero dei documenti: {e}")
    
    return state

# Funzione per generare una risposta
def generate(state: State):
    
    # Combina i messaggi esistenti
    try:
        previous_messages = trimmer.invoke(state["messages"])
        # print("Trimmed messages:", previous_messages)
    except Exception as e:
        print(f"Errore durante il trimming dei messaggi: {e}")
        return state

    # Prepara il contesto generale e dei documenti
    document_context = "\n\n".join(doc.page_content for doc in state["document_context"])

    # Crea la risposta dell'assistente basata sui contesti
    new_messages = [    
        #"Sei un assistente utile. Fornisci risposte dettagliate, approfondite e ben strutturate in italiano, in modo conciso e puntuale." 
        {"role": "system", "content": "Sei uno studente di Ingegneria del Software Semplice. Fornisci risposte dettagliate, approfondite e ben strutturate in italiano basandoti sul contesto fornito."},
        *[
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in previous_messages
        ],
        {"role": "system", "content": f"Contesto:\n\n{document_context}"},
    ]

    # Genera la risposta del LLM
    try:
        response = llm.invoke(new_messages)
        state["messages"].append(AIMessage(content=response.content))
    except Exception as e:
        print(f"Errore durante l'invocazione dell'LLM: {e}")
    
    return state

# Funzione principale
def main():
    
    # Inizializza la memoria e il grafo
    memory = MemorySaver()
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile(checkpointer=memory)

    # Stato iniziale
    initial_state = {
        "messages": [],
        "general_context": "",
        "document_context": [],
    }

    # Configurazione specifica per thread
    config = {"configurable": {"thread_id": "chat1"}}

    while True:
        user_input = input("\033[91mTu:\033[0m ")
        if user_input.lower() in ["exit", "quit"]:
            print("Uscita dalla chat.")
            break

        # Aggiungi l'input dell'utente ai messaggi
        initial_state["messages"].append(HumanMessage(content=user_input))

        # Invoca il grafo
        try:
            output = graph.invoke(initial_state, config)
            print("\033[94mDon_Salva:\033[0m", output["messages"][-1].content + "\n")
        except Exception as e:
            print(f"Errore durante l'esecuzione del grafo: {e}")

if __name__ == "__main__":
    main()
