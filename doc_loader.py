import os

os.environ["USER_AGENT"] = "myagent"

import requests
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Liste di file di testo e PDF
text_paths = [
    "data/bibbia.txt"
    ]

pdf_paths = [
    "data/lezionario.pdf", 
    "data/messale.pdf",
    "data/ccc.pdf",
    ]

# Caricare i documenti di testo
text_documents = []
for text_path in text_paths:
    text_loader = TextLoader(text_path, encoding="iso-8859-1") # iso-8859-1 Encoding per il file bibbia.txt altrimenti utf-8
    text_documents.extend(text_loader.load())

# Caricare i documenti PDF
pdf_documents = []
for pdf_path in pdf_paths:
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_documents.extend(pdf_loader.load())

# Unire i documenti caricati
docs = text_documents + pdf_documents
print(f"Loaded {len(docs)} chunks of text")

# Configurare il text splitter
chunk_size = 700
chunk_overlap = 70
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Suddividere i documenti
all_splits = text_splitter.split_documents(docs)

# Inviare i documenti suddivisi all'endpoint /add_document
response = requests.post(
    'http://localhost:5001/add_document',
    json={"documents": [{"page_content": doc.page_content} for doc in all_splits]}
)

if response.status_code == 200:
    print("Documents added to vector store")
else:
    print({"status": "failure", "reason": response.text})
