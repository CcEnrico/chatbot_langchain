from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_openai import ChatOpenAI
from typing import List, TypedDict
import os
import requests
from dotenv import load_dotenv

# Variabili d'ambiente
load_dotenv()

# Definisci lo Stato con cronologia dei messaggi e contesti separati
class State(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    general_context: str
    document_context: List[Document]

# Inizializza il LLM
llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=2000, temperature=0.6)

# Configura il trimmer
trimmer = trim_messages(
    max_tokens=200,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Funzione per recuperare documenti simili dal database vettoriale
def retrieve(state: State):
    state["document_context"] = []

    if not state["messages"]:
        print("Nessun messaggio disponibile per effettuare la ricerca.")
        return state

    try:
        response = requests.post(
            'http://localhost:5001/similarity_search',
            json={"query": [state["messages"][-1].content]}
        )
        response.raise_for_status()
        retrieved_docs = response.json().get("results", [])
        state["document_context"] = [Document(page_content=doc) for doc in retrieved_docs]
    except requests.exceptions.RequestException as e:
        print(f"Errore durante il recupero dei documenti: {e}")
    
    return state

# Funzione per generare una risposta
def generate(state: State):
    try:
        previous_messages = trimmer.invoke(state["messages"])
    except Exception as e:
        print(f"Errore durante il trimming dei messaggi: {e}")
        return state

    document_context = "\n\n".join(doc.page_content for doc in state["document_context"])

    new_messages = [
        {"role": "system", "content": "Sei un assistente virtuale esperto, progettato per fornire risposte precise, esaustive e professionali riguardanti l'azienda Valsana. Rispondi alle domande in italiano con un tono professionale, ma accessibile e amichevole, e fai riferimento alle informazioni fornite dall'azienda."},
        *[
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
            for msg in previous_messages
        ],
        {"role": "system", "content": f"Contesto:\n\n{document_context}"},
    ]

    try:
        response = llm.invoke(new_messages)
        state["messages"].append(AIMessage(content=response.content))
    except Exception as e:
        print(f"Errore durante l'invocazione dell'LLM: {e}")
    
    return state

# Configurazione dell'app Flask
app = Flask(__name__)
CORS(app)  # Abilita CORS per tutte le route

@app.route('/process_message', methods=['POST'])
def process_message():
    data = request.json
    new_message = data['newMessage']
    old_messages = data['oldMessages']

    state = {
        "messages": [HumanMessage(content=msg['message']) for msg in old_messages],
        "general_context": "",
        "document_context": [],
    }

    state["messages"].append(HumanMessage(content=new_message))

    state = retrieve(state)
    state = generate(state)

    return jsonify({"response": state["messages"][-1].content})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
