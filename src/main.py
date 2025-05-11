from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import os

def load_documents(directory):
    docs = []
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif filename.endswith(".txt"):
            loader = TextLoader(path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def create_faiss_index(chunks, model_name="all-MiniLM-L6-v2", path="config/faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(path)
    return db

def load_faiss_index(model_name="all-MiniLM-L6-v2", path="config/faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(path, embedding_model)

def run_rag_query(db, query):
    llm = OpenAI(temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(), return_source_documents=True)
    return qa.run(query)

if __name__ == "__main__":
    docs = load_documents("documents")
    chunks = split_documents(docs)
    db = create_faiss_index(chunks)

    print("\nRAG System Ready.")
    while True:
        query = input("\nEnter a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        response = run_rag_query(db, query)
        print(f"\nAnswer: {response}")
