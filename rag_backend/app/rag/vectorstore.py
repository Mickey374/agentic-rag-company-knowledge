import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vector_store = None

EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-babbage-001": 2048,
    "text-embedding-curie-001": 4096,
    "text-embedding-davinci-001": 12288,
    "all-MiniLM-L6-v2": 384,
}


def get_embedding_dimensions(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model).strip()

    if not name:
        raise RuntimeError(
            "Embedding model name is not set. Please set the 'embedding_model' in the configuration."
        )

    normalized_name = name.lower()

    if normalized_name in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized_name]
    if "text-embedding-3-small" in normalized_name:
        return 1536
    if "text-embedding-3-large" in normalized_name:
        return 3072
    if "text-embedding-ada-002" in normalized_name:
        return 1536
    if "text-embedding-babbage-001" in normalized_name:
        return 2048
    if "text-embedding-curie-001" in normalized_name:
        return 4096
    if "text-embedding-davinci-001" in normalized_name:
        return 12288
    if "all-MiniLM-L6-v2" in normalized_name:
        return 384

    raise ValueError(f"Unsupported embedding model: {name}")


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OpenAI API key is not set. Please set the 'openai_api_key' in the configuration."
            )

        _embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings


def ensure_pinecone_index_exists():
    """
    Function to ensure that the Pinecone index exists. If it doesn't, create it.
    """
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "Pinecone API key is not set. Please set the 'pinecone_api_key' in the configuration."
        )

    desired_dimensions = get_embedding_dimensions(settings.embedding_model)
    pc_client = Pinecone(api_key=settings.pinecone_api_key)
    namespaces = [x.name for x in pc_client.list_indexes()]

    if settings.pinecone_index_name in namespaces:
        index_info = pc_client.describe_index(settings.pinecone_index_name)
        current_dimensions = getattr(index_info, "dimension", None)
        if current_dimensions is None and isinstance(index_info, dict):
            current_dimensions = index_info.get("dimension")

        if current_dimensions is not None and current_dimensions != desired_dimensions:
            pc_client.delete_index(settings.pinecone_index_name)
            while settings.pinecone_index_name in [
                x.name for x in pc_client.list_indexes()
            ]:
                time.sleep(1)

    if settings.pinecone_index_name not in [x.name for x in pc_client.list_indexes()]:
        pc_client.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimensions,
            metric="cosine",
            serverless=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        while not pc_client.describe_index(settings.pinecone_index_name).status.ready:
            time.sleep(1)

    return pc_client.Index(settings.pinecone_index_name)
