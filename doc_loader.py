import os

os.environ["USER_AGENT"] = "myagent"

import requests
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory contenente i file di ergon
directory = "data/valsana_dataset"

# Liste di file di testo e PDF
text_paths = []
pdf_paths = []

# Smistare i file in base all'estensione
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        text_paths.append(os.path.join(directory, filename))
    elif filename.endswith(".pdf"):
        pdf_paths.append(os.path.join(directory, filename))
# Liste di file di testo e PDF

print(f"Found {len(text_paths)} text files and {len(pdf_paths)} PDF files")

# Caricare i documenti di testo
text_documents = []
for text_path in text_paths:
    print(f"Loading text document: {text_path}")
    text_loader = TextLoader(text_path, encoding="utf-8") # iso-8859-1 Encoding per il file bibbia.txt altrimenti utf-8
    text_documents.extend(text_loader.load())

# Caricare i documenti PDF
pdf_documents = []
for pdf_path in pdf_paths:
    print(f"Loading PDF document: {pdf_path}")
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_documents.extend(pdf_loader.load())

# Unire i documenti caricati
docs = text_documents + pdf_documents
print(f"Loaded {len(docs)} chunks of text")

# Configurare il text splitter
chunk_size = 3000
chunk_overlap = 300
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Suddividere i documenti
all_splits = text_splitter.split_documents(docs)

# Inviare i documenti suddivisi all'endpoint /add_document
print("Sending documents to the endpoint /add_document")
response = requests.post(
    'http://localhost:5001/add_document',
    json={"documents": [{"page_content": doc.page_content} for doc in all_splits]}
)

if response.status_code == 200:
    print("Documents added to vector store")
else:
    print({"status": "failure", "reason": response.text})
