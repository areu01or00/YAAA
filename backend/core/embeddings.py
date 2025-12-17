"""Embedding client for OpenRouter."""

import os
import httpx


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
