"""PaperMap API Server."""

import os
import asyncio
import uuid
from enum import Enum
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from core import arxiv_client, embeddings, clustering, llm, openalex, chat

load_dotenv()


# Job status tracking for async PDF parsing
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ParseJob:
    def __init__(self, arxiv_id: str, pdf_url: str):
        self.job_id = str(uuid.uuid4())
        self.arxiv_id = arxiv_id
        self.pdf_url = pdf_url
        self.status = JobStatus.PENDING
        self.progress = 0  # 0-100
        self.error: str | None = None


# In-memory job tracking (in production, use Redis)
_parse_jobs: dict[str, ParseJob] = {}
_arxiv_to_job: dict[str, str] = {}  # Map arxiv_id to job_id for deduplication

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


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class PaperContext(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str


class ChatRequest(BaseModel):
    message: str
    papers: list[PaperContext]
    history: list[ChatMessage] = []
    parse_pdfs: bool = True  # Whether to parse PDFs with VLM
    use_web_search: bool = True  # Whether to search web for additional context


class ChatResponse(BaseModel):
    response: str
    papers_parsed: list[str] = []  # arxiv_ids of papers that were parsed


class ParsePaperRequest(BaseModel):
    arxiv_id: str
    pdf_url: str


class ParsePaperResponse(BaseModel):
    """Response when starting a parse job."""
    job_id: str
    arxiv_id: str
    status: str  # pending, processing, completed, failed
    cached: bool = False  # True if already in cache


class ParseStatusResponse(BaseModel):
    """Response when checking parse job status."""
    job_id: str
    arxiv_id: str
    status: str
    progress: int  # 0-100
    content: str | None = None  # Only present when completed
    error: str | None = None  # Only present when failed


# In-memory cache for parsed papers (in production, use Redis or similar)
_paper_content_cache: dict[str, str] = {}


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

    # Fetch papers from all queries - PARALLEL with staggered 3s delays
    # Arxiv rate limit is 1 request per 3 seconds, so we stagger starts
    papers_per_query = max(20, request.max_results // len(queries))

    async def fetch_with_delay(q: str, delay: float):
        if delay > 0:
            await asyncio.sleep(delay)
        return await arxiv_client.fetch_papers(q, papers_per_query)

    # Launch all queries with staggered delays (0s, 3s, 6s, 9s, 12s, 15s)
    tasks = [fetch_with_delay(q, i * 3.0) for i, q in enumerate(queries)]
    results = await asyncio.gather(*tasks)

    # Deduplicate results
    all_papers: list[arxiv_client.Paper] = []
    seen_ids: set[str] = set()
    for fetched in results:
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

    # Parallel: name_clusters (LLM) + citation_data (OpenAlex API)
    # Dean/Ghemawat: "reduce latency by overlapping independent operations"
    arxiv_ids = [p.arxiv_id for p in papers]
    paper_id_set = set(arxiv_ids)

    # Start both tasks
    cluster_names_task = clustering.name_clusters(papers, n_clusters, request.query)
    citation_task = asyncio.get_event_loop().run_in_executor(
        None, openalex.get_citation_data_batch, arxiv_ids
    )

    # Await both in parallel
    cluster_names, citation_data = await asyncio.gather(
        cluster_names_task, citation_task
    )

    # Apply cluster names
    for paper in papers:
        if paper.cluster in cluster_names:
            paper.cluster_name = cluster_names[paper.cluster][0]

    categories = clustering.build_categories(papers, cluster_names)

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


async def _run_parse_job(job: ParseJob):
    """Background task to parse PDF."""
    try:
        job.status = JobStatus.PROCESSING
        job.progress = 10

        # Parse PDF - processes ALL pages
        content = await chat.parse_pdf_with_vlm(job.pdf_url)

        if content:
            _paper_content_cache[job.arxiv_id] = content
            job.status = JobStatus.COMPLETED
            job.progress = 100
        else:
            job.status = JobStatus.FAILED
            job.error = "Failed to extract content from PDF"
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)


@app.post("/api/parse-paper", response_model=ParsePaperResponse)
async def parse_paper(request: ParsePaperRequest):
    """Start parsing a paper's PDF. Returns immediately with job ID."""
    # Check cache first - return immediately if already parsed
    if request.arxiv_id in _paper_content_cache:
        return ParsePaperResponse(
            job_id="cached",
            arxiv_id=request.arxiv_id,
            status=JobStatus.COMPLETED,
            cached=True
        )

    # Check if job already exists for this paper
    if request.arxiv_id in _arxiv_to_job:
        existing_job_id = _arxiv_to_job[request.arxiv_id]
        if existing_job_id in _parse_jobs:
            job = _parse_jobs[existing_job_id]
            return ParsePaperResponse(
                job_id=job.job_id,
                arxiv_id=request.arxiv_id,
                status=job.status,
                cached=False
            )

    # Create new job
    job = ParseJob(request.arxiv_id, request.pdf_url)
    _parse_jobs[job.job_id] = job
    _arxiv_to_job[request.arxiv_id] = job.job_id

    # Start background task
    asyncio.create_task(_run_parse_job(job))

    return ParsePaperResponse(
        job_id=job.job_id,
        arxiv_id=request.arxiv_id,
        status=job.status,
        cached=False
    )


@app.get("/api/parse-status/{job_id}", response_model=ParseStatusResponse)
async def parse_status(job_id: str):
    """Check status of a parse job."""
    # Handle cached responses
    if job_id == "cached":
        raise HTTPException(400, "Use arxiv_id to check cached papers")

    if job_id not in _parse_jobs:
        raise HTTPException(404, "Job not found")

    job = _parse_jobs[job_id]

    # Include content if completed
    content = None
    if job.status == JobStatus.COMPLETED and job.arxiv_id in _paper_content_cache:
        content = _paper_content_cache[job.arxiv_id]

    return ParseStatusResponse(
        job_id=job.job_id,
        arxiv_id=job.arxiv_id,
        status=job.status,
        progress=job.progress,
        content=content,
        error=job.error
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat about papers using cached PDF content."""
    if not request.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    if not request.papers:
        raise HTTPException(400, "At least one paper required for context")

    # No inline parsing - frontend handles parsing via polling
    # Just use whatever is in cache
    papers_parsed = [p.arxiv_id for p in request.papers if p.arxiv_id in _paper_content_cache]

    # Build paper dicts for chat
    paper_dicts = [
        {
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "pdf_url": p.pdf_url,
        }
        for p in request.papers
    ]

    # Convert history to dict format
    history = [{"role": m.role, "content": m.content} for m in request.history]

    # Chat with LLM
    response = await chat.chat_with_context(
        message=request.message,
        papers=paper_dicts,
        paper_contents=_paper_content_cache,
        history=history,
        use_web_search=request.use_web_search,
    )

    return ChatResponse(response=response, papers_parsed=papers_parsed)


if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
