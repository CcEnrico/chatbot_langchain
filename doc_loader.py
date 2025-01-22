import os

os.environ["USER_AGENT"] = "myagent"

import requests
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# Liste di file di testo e PDF
text_paths = [
    # "data/bibbia.txt"
    ]

pdf_paths = [
    # "data/lezionario.pdf", 
    # "data/messale.pdf",
    # "data/ccc.pdf"
    # "data/Incontro 1 - Sabato 23_11_24.pdf",
    # "data/Incontro 2 - Sabato 30-11-24.pdf",
    # "data/Incontro 3 - Sabato 07_12_24.pdf",
    # "data/Incontro 4 - Sabato 14_12_24.pdf",
    # "data/Incontro 5 - Sabato 21_12_24.pdf",
    # "data/Incontro 6 - Sabato 28_12_24.pdf",
    # "data/Incontro 7 - Martedì 07_01_25.pdf",
    # "data/INFORMAZIONI E CONTATTI.pdf",
    # "data/PRESENTAZIONE.pdf",
    "data/Ingegneria del software semplice (per davvero).pdf",
    # "data/Tutti i Quiz (Teoria e Pratica)_al 2023.pdf",
    # "data/Quiz_Teoria_Pratica.pdf"
    "data/Raccolta risposte Teoria (2023).pdf",
    "data/Domande di teoria (2016).pdf",
    "data/SWE-Domande&Risposte (2018).pdf"
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
chunk_size = 3000
chunk_overlap = 300
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
