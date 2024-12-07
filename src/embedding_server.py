from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

app = Flask(__name__)

# Inizializza il modello di embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Inizializza un vector store vuoto
vector_store = FAISS.from_texts([""], embedding=embedding_model)

# Percorso del file per salvare il vector store
VECTOR_STORE_PATH = "../database/vector_store"

@app.route('/add_document', methods=['POST'])
def add_documents():
    documents = request.json['documents']
    
    # Convert dictionaries to Document instances
    document_objects = [Document(page_content=doc['page_content']) for doc in documents]
    
    vector_store.add_documents(document_objects)

    print("Added documents to vector store" + str(len(document_objects)))

    return jsonify({"status": "success"})


@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    query_embedding = request.json['query']
    results = vector_store.similarity_search(query_embedding[0], k=4)
    
    return jsonify({"results": [doc.page_content for doc in results]})


@app.route('/save_vector_store', methods=['POST'])
def save_vector_store():
    # vector_store.save(VECTOR_STORE_PATH)
    vector_store.save_local(VECTOR_STORE_PATH)
    return jsonify({"status": "vector store saved"})


@app.route('/load_vector_store', methods=['POST'])
def load_vector_store():
    global vector_store
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load(VECTOR_STORE_PATH, embedding=embedding_model)
        return jsonify({"status": "vector store loaded"})
    else:
        return jsonify({"status": "vector store file not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)