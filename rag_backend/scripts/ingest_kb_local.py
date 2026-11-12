from pathlib import Path

from app.core.config import get_settings
from app.rag.vectorstore import add_documents_to_vector_store
from app.services.ingestion import (
    SUPPORTED_FILE_TYPES,
    chunk_documents,
    load_documents_from_file,
)

settings = get_settings()

data_dir = Path(settings.sample_kb_dir)
files = sorted(
    path
    for path in data_dir.iterdir()
    if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_TYPES
)

all_documents = []
failed_files = []

print(f"Found {len(files)} file(s) in {data_dir}")

for index, file_path in enumerate(files, start=1):
    try:
        print(f"[{index}/{len(files)}] Loading {file_path.name}...")
        documents = load_documents_from_file(file_path)
        all_documents.extend(documents)
        print(f"[{index}/{len(files)}] Loaded {len(documents)} document(s)")
    except Exception as error:
        failed_files.append(file_path)
        print(f"[{index}/{len(files)}] Failed to load {file_path.name}: {error}")

if not all_documents:
    raise RuntimeError(f"No documents were loaded from {data_dir}")

print(f"Chunking {len(all_documents)} loaded document(s)...")
chunked_documents = chunk_documents(all_documents)
print(f"Created {len(chunked_documents)} chunks")

print("Indexing chunks...")
add_documents_to_vector_store(chunked_documents)

print(
    f"Indexed {len(chunked_documents)} chunks from "
    f"{len(files) - len(failed_files)} file(s)."
)

if failed_files:
    print(f"Completed with {len(failed_files)} failed file(s):")
    for file_path in failed_files:
        print(f"  - {file_path.name}")
