import os
os.environ["USER_AGENT"] = "myagent"
import requests

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# web_paths = ("https://lilianweng.github.io/posts/2023-06-23-agent/",)
text_path = "../data/bibbia.txt"
# pdf_paths = ("../data/bibbia.pdf",)
document_loader = TextLoader(text_path, encoding="iso-8859-1")

chunk_size = 500
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Load and split documents
docs = document_loader.load()
all_splits = text_splitter.split_documents(docs)

# Send the split documents to the /add_document endpoint
response = requests.post(
    'http://localhost:5001/add_document',
    json={"documents": [{"page_content": doc.page_content} for doc in all_splits]}
)

if response.status_code == 200:
    print("Documents added to vector store")
else:
    print({"status": "failure", "reason": response.text})
