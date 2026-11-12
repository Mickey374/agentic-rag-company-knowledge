from app.services.ingestion import load_documents_from_file, chunk_documents
from pathlib import Path
from app.rag.vectorstore import add_documents_to_vector_store

docs = load_documents_from_file(Path("data/dpr_emnlp2020.pdf"))
chunks = chunk_documents(docs)

add_documents_to_vector_store(chunks)
