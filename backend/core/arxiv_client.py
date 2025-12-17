"""Arxiv API client."""

import asyncio
import arxiv
from pydantic import BaseModel
from typing import Optional


class Paper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    pdf_url: str
    categories: list[str]
    cluster: Optional[int] = None
    cluster_name: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    neighbors: Optional[list[str]] = None  # arxiv_ids of nearest neighbors


async def fetch_papers(query: str, max_results: int = 200) -> list[Paper]:
    """Fetch papers from arxiv by query."""
    # Add delay to avoid rate limits (arxiv allows 1 request per 3 seconds)
    client = arxiv.Client(
        page_size=100,
        delay_seconds=3.0,
        num_retries=3,
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    loop = asyncio.get_event_loop()

    def fetch():
        papers = []
        try:
            for result in client.results(search):
                papers.append(result)
                if len(papers) >= max_results:
                    break
        except arxiv.HTTPError as e:
            print(f"Arxiv rate limit hit after {len(papers)} papers: {e}")
        return papers

    results = await loop.run_in_executor(None, fetch)

    papers = []
    for result in results:
        papers.append(Paper(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title.replace("\n", " "),
            abstract=result.summary.replace("\n", " "),
            authors=[a.name for a in result.authors[:5]],
            published=result.published.isoformat(),
            pdf_url=result.pdf_url,
            categories=list(result.categories),
        ))

    return papers
