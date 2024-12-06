import os

os.environ["USER_AGENT"] = "LangChainExample/1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import bs4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create FAISS vector store
vector_store = FAISS.from_texts([""], embedding=embeddings)

# Load and chunk contents of a blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Add documents to FAISS
vector_store.add_documents(documents=all_splits)

# Pull the prompt for RAG
prompt = hub.pull("rlm/rag-prompt")

# Define application state
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define retrieval step
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    print(retrieved_docs)
    return {"context": retrieved_docs}

# Define generation step
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Run the application with a test question
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
