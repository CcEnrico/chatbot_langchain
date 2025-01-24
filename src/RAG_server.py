import logging
from flask import Flask, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

# Variabili d'ambiente
# Carica le variabili d'ambiente dal file .env
load_dotenv(dotenv_path="../.env")

# os.environ["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM")
# os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Inizializza il modello di embedding
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Inizializza un vector store vuoto
vector_store = FAISS.from_texts([""], embedding=embedding_model)

# Percorso del file per salvare il vector store
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "../database/vector_store")

# Configurazione dell'app Flask
app = Flask(__name__)

# Configura il logger
logging.basicConfig(filename='server.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')


@app.route('/add_document', methods=['POST'])
def add_documents():
    documents = request.json['documents']
    
    # Converti i dizionari in istanze di Document
    document_objects = [Document(page_content=doc['page_content']) for doc in documents]
    
    vector_store.add_documents(document_objects)
    
    # Stampa informazioni sull'operazione
    print(f"Added {len(document_objects)} documents to vector store")
    
    return jsonify({"status": "success"})


@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    query_embedding = request.json['query']
    results = vector_store.similarity_search(query_embedding[0], k=4)
    
    # Log informazioni sull'operazione
    logging.info(f"Performed similarity search with query: {query_embedding}")
    logging.info(f"Results: {[doc.page_content for doc in results]}")
    
    return jsonify({"results": [doc.page_content for doc in results]})


@app.route('/save_vector_store', methods=['POST'])
def save_vector_store():
    vector_store.save_local(VECTOR_STORE_PATH)
    
    # Log informazioni sull'operazione
    logging.info(f"Saved vector store to {VECTOR_STORE_PATH}")
    
    return jsonify({"status": "vector store saved"})


@app.route('/load_vector_store', methods=['POST'])
def load_vector_store():
    global vector_store
    # Controlla se il file del vector store esiste
    if os.path.exists(VECTOR_STORE_PATH):
        # Carica il vector store dal file
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH, 
            embeddings=embedding_model, 
            allow_dangerous_deserialization=True # Abilita questo se ti fidi della fonte dei dati
        )
        
        # Log informazioni sul vector store caricato
        file_size = os.path.getsize(VECTOR_STORE_PATH + "/index.faiss")
        logging.info(f"Size of the FAISS file: {file_size / (1024 * 1024):.2f} MB")
        logging.info(f"Number of vectors: {vector_store.index.ntotal}")
        logging.info(f"Dimension of vectors: {vector_store.index.d}")
        logging.info(f"Metric type: {vector_store.index.metric_type}")
        
        return jsonify({"status": "vector store loaded"})
    else:
        # Log errore se il file del vector store non viene trovato
        logging.error(f"Vector store file not found at {VECTOR_STORE_PATH}")
        return jsonify({"status": "vector store file not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)