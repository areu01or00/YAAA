"""Clustering and dimensionality reduction."""

import json
import numpy as np
from pydantic import BaseModel
from core.arxiv_client import Paper
from core import llm


class Category(BaseModel):
    id: int
    name: str
    description: str
    color: str
    count: int


COLORS = [
    "#e63946", "#f4a261", "#2a9d8f", "#264653", "#e9c46a",
    "#9b5de5", "#00bbf9", "#00f5d4", "#f15bb5", "#fee440",
    "#8338ec", "#3a86ff", "#fb5607", "#ff006e", "#8ac926",
]


def find_nearest_neighbors(embeddings: np.ndarray, n_neighbors: int = 3) -> list[list[int]]:
    """Find n nearest neighbors for each paper based on cosine similarity."""
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    # Compute similarity matrix
    similarity = normalized @ normalized.T

    # For each paper, find top n_neighbors (excluding self)
    neighbors = []
    for i in range(len(embeddings)):
        # Set self-similarity to -inf to exclude
        sim = similarity[i].copy()
        sim[i] = -np.inf
        # Get indices of top n neighbors
        top_indices = np.argsort(sim)[-n_neighbors:][::-1]
        neighbors.append(top_indices.tolist())

    return neighbors


def kmeans(embeddings: np.ndarray, n_clusters: int, max_iter: int = 100) -> np.ndarray:
    """K-means clustering."""
    n_samples = embeddings.shape[0]
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = embeddings[indices].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(embeddings[:, np.newaxis] - centroids, axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.all(labels == new_labels):
            break
        labels = new_labels

        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                centroids[k] = embeddings[mask].mean(axis=0)

    return labels


def pca_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce to 2D using PCA."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    projection = centered @ eigenvectors[:, :2]
    projection = projection / (np.abs(projection).max() + 1e-10)
    return projection


async def name_clusters(papers: list[Paper], n_clusters: int, query: str) -> dict[int, tuple[str, str]]:
    """Name clusters using LLM."""
    clusters: dict[int, list[Paper]] = {}
    for paper in papers:
        if paper.cluster is not None:
            clusters.setdefault(paper.cluster, []).append(paper)

    cluster_texts = []
    for cluster_id in sorted(clusters.keys()):
        reps = clusters[cluster_id][:5]
        papers_text = "\n".join([f"  - {p.title}" for p in reps])
        cluster_texts.append(f"Cluster {cluster_id} ({len(clusters[cluster_id])} papers):\n{papers_text}")

    prompt = f"""Papers about "{query}" grouped by similarity. Name each cluster (2-4 words).

{chr(10).join(cluster_texts)}

Return ONLY JSON:
{{"clusters": [{{"id": 0, "name": "Name", "description": "One sentence"}}]}}"""

    response = await llm.complete([{"role": "user", "content": prompt}])

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        data = json.loads(response[json_start:json_end])
        return {c["id"]: (c["name"], c["description"]) for c in data["clusters"]}
    except:
        return {i: (f"Cluster {i}", "") for i in range(n_clusters)}


def build_categories(papers: list[Paper], cluster_names: dict[int, tuple[str, str]]) -> list[Category]:
    """Build category list from papers."""
    counts: dict[int, int] = {}
    for paper in papers:
        if paper.cluster is not None:
            counts[paper.cluster] = counts.get(paper.cluster, 0) + 1

    categories = []
    for cluster_id in sorted(counts.keys()):
        name, desc = cluster_names.get(cluster_id, (f"Cluster {cluster_id}", ""))
        categories.append(Category(
            id=cluster_id,
            name=name,
            description=desc,
            color=COLORS[cluster_id % len(COLORS)],
            count=counts[cluster_id],
        ))

    return categories
