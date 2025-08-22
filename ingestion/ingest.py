import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from config import CHROMA_DIR, CHROMA_GLOBAL_COLLECTION_NAME, EMBEDDING_MODEL

def load_documents():
    docs = []
    kb_path = "knowledge_base"
    for file in os.listdir(kb_path):
        file_path = os.path.join(kb_path, file)
        if file.endswith(".json"):
            loader = JSONLoader(file_path, jq_schema=".", text_content=False)
            docs.extend(loader.load())
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
    return docs

def main():
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=CHROMA_GLOBAL_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectorstore.add_documents(splits)
    print(f"✅ Ingested {len(splits)} documents into Chroma DB")

if __name__ == "__main__":
    main()
