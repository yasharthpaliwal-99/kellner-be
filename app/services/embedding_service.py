"""
Generates text embeddings using Azure AI Services (text-embedding-ada-002).
Returns a 1536-dimensional float list ready for pgvector storage/search.
"""
from typing import List

from openai import AzureOpenAI

from app.config import config

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        if not config.AZURE_EMBEDDING_ENDPOINT or not config.AZURE_EMBEDDING_API_KEY:
            raise ValueError("Set AZURE_EMBEDDING_ENDPOINT and AZURE_EMBEDDING_API_KEY in .env")
        _client = AzureOpenAI(
            azure_endpoint=config.AZURE_EMBEDDING_ENDPOINT,
            api_key=config.AZURE_EMBEDDING_API_KEY,
            api_version="2023-05-15",
        )
    return _client


def embed(text: str) -> List[float]:
    """Return the embedding vector for a single piece of text."""
    import time
    t = time.perf_counter()
    response = _get_client().embeddings.create(
        model=config.AZURE_EMBEDDING_DEPLOYMENT,
        input=text.replace("\n", " "),
    )
    print(f"  [EMBED] '{text[:40]}': {time.perf_counter() - t:.2f}s")
    return response.data[0].embedding
