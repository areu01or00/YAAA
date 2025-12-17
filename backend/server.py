"""PaperMap API Server."""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from core import arxiv_client, embeddings, clustering

load_dotenv()

app = FastAPI(title="PaperMap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str
    max_results: int = 200


class SearchResponse(BaseModel):
    papers: list[arxiv_client.Paper]
    categories: list[clustering.Category]
    query: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search arxiv, embed, cluster, return papers with 2D positions."""
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Fetch
    papers = await arxiv_client.fetch_papers(request.query, request.max_results)
    if len(papers) < 2:
        return SearchResponse(papers=papers, categories=[], query=request.query)

    # Embed
    texts = [f"{p.title}. {p.abstract[:500]}" for p in papers]
    emb = await embeddings.get_embeddings(texts)
    emb_np = np.array(emb)

    # Cluster
    n_clusters = min(8, max(3, len(papers) // 20))
    labels = clustering.kmeans(emb_np, n_clusters)
    positions = clustering.pca_2d(emb_np)

    # Find nearest neighbors
    neighbor_indices = clustering.find_nearest_neighbors(emb_np, n_neighbors=3)

    for i, paper in enumerate(papers):
        paper.cluster = int(labels[i])
        paper.x = float(positions[i, 0])
        paper.y = float(positions[i, 1])
        paper.neighbors = [papers[j].arxiv_id for j in neighbor_indices[i]]

    # Name clusters
    cluster_names = await clustering.name_clusters(papers, n_clusters, request.query)
    for paper in papers:
        if paper.cluster in cluster_names:
            paper.cluster_name = cluster_names[paper.cluster][0]

    categories = clustering.build_categories(papers, cluster_names)

    return SearchResponse(papers=papers, categories=categories, query=request.query)


if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
