"""PaperMap API Server."""

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from core import arxiv_client, embeddings, clustering, llm, openalex

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


class CitationLink(BaseModel):
    source: str  # arxiv_id of citing paper
    target: str  # arxiv_id of cited paper


class SearchResponse(BaseModel):
    papers: list[arxiv_client.Paper]
    categories: list[clustering.Category]
    citation_links: list[CitationLink] = []  # Real citation relationships
    query: str
    expanded_queries: list[str] = []
    max_citations: int = 0  # For normalizing pulse intensity


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search arxiv, embed, cluster, return papers with 2D positions."""
    if not request.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    # Generate expanded queries
    queries = await llm.generate_search_queries(request.query, num_queries=6)

    # Fetch papers from all queries
    papers_per_query = max(20, request.max_results // len(queries))
    all_papers: list[arxiv_client.Paper] = []
    seen_ids: set[str] = set()

    for q in queries:
        fetched = await arxiv_client.fetch_papers(q, papers_per_query)
        for p in fetched:
            if p.arxiv_id not in seen_ids:
                seen_ids.add(p.arxiv_id)
                all_papers.append(p)

    papers = all_papers[:request.max_results]

    if len(papers) < 2:
        return SearchResponse(papers=papers, categories=[], query=request.query, expanded_queries=queries)

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

    # Fetch citation data from OpenAlex
    arxiv_ids = [p.arxiv_id for p in papers]
    paper_id_set = set(arxiv_ids)
    citation_data = openalex.get_citation_data_batch(arxiv_ids)

    # Build citation links and update papers
    citation_links: list[CitationLink] = []
    max_citations = 0

    for paper in papers:
        data = citation_data.get(paper.arxiv_id, {})
        paper.citation_count = data.get("citation_count", 0)
        paper.references = data.get("references", [])
        max_citations = max(max_citations, paper.citation_count)

        # Add links for references that are in our paper set
        for ref_id in paper.references or []:
            if ref_id in paper_id_set:
                citation_links.append(CitationLink(source=paper.arxiv_id, target=ref_id))

    return SearchResponse(
        papers=papers,
        categories=categories,
        citation_links=citation_links,
        query=request.query,
        expanded_queries=queries,
        max_citations=max_citations
    )


if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
