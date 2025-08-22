from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import CHROMA_DIR, CHROMA_GLOBAL_COLLECTION_NAME, EMBEDDING_MODEL
from crewai_tools import tool

def _make_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=CHROMA_GLOBAL_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

@tool("kb_retriever")
def retriever_tool(query: str) -> str:
    """Retrieve relevant chunks from knowledge base (JSON, PDFs)."""
    vs = _make_vectorstore()
    docs = vs.similarity_search(query, k=4)
    if not docs:
        return "NO_MATCH"
    return "\n\n".join([f"Source: {d.metadata.get('source','?')}\n{d.page_content}" for d in docs])
