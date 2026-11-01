import csv
import json
from pathlib import Path
from typing import Any, List, Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from docx import Document as DocxDocument

SUPPORTED_FILE_TYPES = {".pdf", ".txt", ".csv", ".md", ".json", ".docx"}


def _load_csv_documents(file_path: Path) -> List[Document]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {file_path}")

        documents = []
        for row_number, row in enumerate(reader, start=2):
            content = "\n".join(
                f"{field}: {value}" for field, value in row.items() if value
            )
            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "row": row_number,
                    },
                )
            )

    return documents


def _load_json_documents(file_path: Path) -> List[Document]:
    with file_path.open("r", encoding="utf-8") as json_file:
        payload: Any = json.load(json_file)

    records = payload if isinstance(payload, list) else [payload]
    documents = []
    for record_number, record in enumerate(records, start=1):
        if isinstance(record, (dict, list)):
            content = json.dumps(record, ensure_ascii=False, indent=2)
        else:
            content = str(record)

        if not content.strip():
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "record": record_number,
                },
            )
        )

    return documents


def load_documents_from_file(file_path: Path) -> List[Document]:
    """
    Load documents from a file based on its extension.

    Args:
        file_path (str): The path to the file.
        Returns:
        List[Document]: A list of loaded documents.
    """
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")

    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    elif suffix in [".txt", ".md"]:
        return TextLoader(str(file_path), encoding="utf-8").load()
    elif suffix == ".docx":
        docx = DocxDocument(str(file_path))
        text = "\n".join([para.text for para in docx.paragraphs if para.text.strip()])
        return [Document(page_content=text, metadata={"source": str(file_path)})]
    elif suffix == ".csv":
        return _load_csv_documents(file_path)
    elif suffix == ".json":
        return _load_json_documents(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def chunk_documents(
    documents: Iterable[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunk documents into smaller pieces.

    Args:
        documents (List[Document]): The list of documents to chunk.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping characters between chunks.

    Returns:
        List[Document]: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    return text_splitter.split_documents(list(documents))
