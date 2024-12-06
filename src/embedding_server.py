from flask import Flask, request, jsonify
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

app = Flask(__name__)

# Inizializza il modello di embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Inizializza un vector store vuoto
vector_store = FAISS.from_texts([""], embedding=embedding_model)

# @app.route('/get_embeddings', methods=['POST'])
# def get_embeddings():
#     texts = request.json['texts']
#     embeddings = embedding_model.embed_texts(texts)
#     return jsonify({"embeddings": embeddings})

from langchain.schema import Document

@app.route('/add_document', methods=['POST'])
def add_documents():
    documents = request.json['documents']
    # print(documents)
    
    # Convert dictionaries to Document instances
    document_objects = [Document(page_content=doc['page_content']) for doc in documents]
    
    vector_store.add_documents(document_objects)

    print("Added documents to vector store" + str(len(document_objects)))

    return jsonify({"status": "success"})


@app.route('/similarity_search', methods=['POST'])
def similarity_search():
    query_embedding = request.json['query']
    results = vector_store.similarity_search(query_embedding[0], k=3)
    
    return jsonify({"results": [doc.page_content for doc in results]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)