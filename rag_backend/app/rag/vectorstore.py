import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
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
    name = (model_name or settings.huggingface_embedding_model).strip()

    if not name:
        raise RuntimeError(
            "Embedding model name is not set. Please set the Hugging Face embedding model in the configuration."
        )

    normalized_name = name.lower()

    normalized_dimensions = {
        model.lower(): dimensions
        for model, dimensions in EMBEDDING_DIMENSIONS.items()
    }

    if normalized_name in normalized_dimensions:
        return normalized_dimensions[normalized_name]
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
    if "all-minilm-l6-v2" in normalized_name:
        return 384

    raise ValueError(f"Unsupported embedding model: {name}")


def get_openai_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError(
                "OpenAI API key is not set. Please set the 'openai_api_key' in the configuration."
            )

        _embeddings = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key,
        )
    return _embeddings


def get_huggingface_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
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

    desired_dimensions = get_embedding_dimensions(settings.huggingface_embedding_model)
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
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        while not pc_client.describe_index(settings.pinecone_index_name).status.ready:
            time.sleep(1)

    return pc_client.Index(settings.pinecone_index_name)


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        index = ensure_pinecone_index_exists()
        embeddings = get_huggingface_embeddings()
        _vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=settings.pinecone_namespace,
        )
    return _vector_store


def get_retriever():
    return get_vector_store().as_retriever(search_kwargs={"k": settings.top_k})


def add_documents_to_vector_store(documents):
    vector_store = get_vector_store()
    vector_store.add_documents(documents)
