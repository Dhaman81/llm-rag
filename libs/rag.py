import os
import base64
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
import inspect
from sqlalchemy import create_engine, text
import libs.logging_txt as log
import libs.db as db
from dotenv import load_dotenv
load_dotenv()

# EMBEDDING_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

def create_pgvector_table():
    engine = create_engine(db.db_connection_string())
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.page_content = doc.page_content.replace("\n", " ").strip()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def load_and_split_pdf_docling(file_path, chunk_size=1000, chunk_overlap=200):
    loader = DoclingLoader(file_path=file_path)
    # print(inspect.signature(DoclingLoader.__init__))
    raw_documents = loader.load()

    # Filter aman: buang dokumen kosong atau corrupt
    valid_documents = []
    for doc in raw_documents:
        try:
            content = getattr(doc, "page_content", None) or getattr(doc, "text", None)
            page = getattr(doc, "page", None)
            page_no = getattr(doc, "page_no", None)

            # if content and content.strip() and page is not None:
            if content and content.strip() is not None:
                valid_documents.append(doc)
            else:
                print(f"[SKIP] Halaman {page_no} kosong atau tidak valid.")
        except Exception as e:
            print(f"[ERROR] Dokumen halaman {getattr(doc, 'page_no', '?')} rusak: {e}")

    # Split dokumen menjadi chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(valid_documents)
    return chunks

def store_embeddings(collection,docs):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection,
        connection_string=db.db_connection_string(),
    )

def search_similar_docs(collection,query_text, k=os.getenv("TOP_K")):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = PGVector(
        collection_name=collection,
        connection_string=db.db_connection_string(),
        embedding_function=embeddings
    )
    docs_and_scores = vectorstore.similarity_search_with_score(query_text, k=k)
    return docs_and_scores

def load_retriever_from_pgvector(collection,model_name=EMBEDDING_MODEL,top_k=os.getenv("TOP_K")):
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = PGVector(
        collection_name=collection,
        connection_string=db.db_connection_string(),
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",  # mmr, similarity, similarity_score_threshold
        search_kwargs={"k": top_k}
    )
    return retriever

