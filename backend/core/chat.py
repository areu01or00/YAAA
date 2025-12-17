"""Chat module with VLM PDF parsing and LLM responses."""

import os
import io
import base64
import asyncio
import httpx
import fitz  # PyMuPDF

from core import web_search


async def fetch_pdf(url: str) -> bytes | None:
    """Fetch PDF content from URL."""
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except Exception as e:
        print(f"Failed to fetch PDF from {url}: {e}")
        return None


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 10, dpi: int = 150) -> list[str]:
    """Convert PDF pages to base64 encoded images.

    Returns list of base64 strings (PNG format).
    """
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = min(len(doc), max_pages)

        for page_num in range(num_pages):
            page = doc[page_num]
            # Render page to image
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            images.append(b64)

        doc.close()
    except Exception as e:
        print(f"Failed to convert PDF to images: {e}")

    return images


async def parse_pdf_with_vlm(pdf_url: str, max_pages: int = 5) -> str:
    """Fetch PDF, convert to images, and extract text using VLM.

    Returns extracted text content from the paper.
    """
    # Fetch PDF
    pdf_bytes = await fetch_pdf(pdf_url)
    if not pdf_bytes:
        return ""

    # Convert to images
    images = pdf_to_images(pdf_bytes, max_pages=max_pages)
    if not images:
        return ""

    # Use VLM to extract text from images concurrently
    api_key = os.getenv("OPENROUTER_API_KEY")
    vlm_model = os.getenv("OPENROUTER_MODEL_VLM", "qwen/qwen-2.5-vl-72b-instruct")

    async def extract_page(img_b64: str, page_num: int) -> str:
        """Extract text from a single page image."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": vlm_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{img_b64}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": "Extract all text content from this research paper page. Include section headers, body text, equations (describe them), figure captions, and table contents. Be thorough but concise."
                                    }
                                ]
                            }
                        ],
                        "temperature": 0.1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"VLM extraction failed for page {page_num}: {e}")
            return ""

    # Process pages concurrently (limit to 5 at a time)
    semaphore = asyncio.Semaphore(5)

    async def extract_with_limit(img: str, num: int) -> str:
        async with semaphore:
            return await extract_page(img, num)

    tasks = [extract_with_limit(img, i) for i, img in enumerate(images)]
    results = await asyncio.gather(*tasks)

    # Combine results
    combined = "\n\n".join([f"[Page {i+1}]\n{text}" for i, text in enumerate(results) if text])
    return combined


async def chat_with_context(
    message: str,
    papers: list[dict],
    paper_contents: dict[str, str],
    history: list[dict] | None = None,
    use_web_search: bool = True,
) -> str:
    """Chat with LLM using paper context and optional web search.

    Args:
        message: User's question
        papers: List of paper dicts with title, abstract, arxiv_id, pdf_url
        paper_contents: Dict mapping arxiv_id to parsed PDF content
        history: Optional conversation history
        use_web_search: Whether to search web for additional context

    Returns:
        LLM response
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

    # Build context from papers
    paper_context = []
    for p in papers:
        ctx = f"**{p['title']}** (arxiv:{p['arxiv_id']})\n"
        ctx += f"Abstract: {p['abstract']}\n"

        # Add parsed PDF content if available
        if p['arxiv_id'] in paper_contents:
            content = paper_contents[p['arxiv_id']]
            # Truncate if too long
            if len(content) > 8000:
                content = content[:8000] + "...[truncated]"
            ctx += f"\nFull Paper Content:\n{content}\n"

        paper_context.append(ctx)

    papers_text = "\n---\n".join(paper_context)

    # Optional web search
    web_context = ""
    if use_web_search:
        web_results = web_search.search_summary(message, max_results=3)
        if web_results:
            web_context = f"\n\nWeb Search Results:\n{web_results}"

    # Build system prompt
    system_prompt = f"""You are a research assistant helping analyze academic papers.

You have access to the following papers:

{papers_text}
{web_context}

Guidelines:
- Answer questions based on the paper content provided
- Cite specific papers when making claims (use arxiv ID)
- If information isn't in the papers, say so clearly
- Be concise but thorough
- For technical questions, explain concepts clearly"""

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    # Call LLM
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.3,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
