"""Chat module with VLM PDF parsing and LLM responses."""

import os
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
from openai import AsyncOpenAI
from dotenv import load_dotenv

from core import web_search

load_dotenv()

# Shared OpenRouter client - REUSED for all calls (connection pooling)
_openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=4)

# VLM concurrency limit
VLM_CONCURRENCY = 15


async def fetch_pdf(url: str) -> bytes | None:
    """Fetch PDF content from URL."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except Exception as e:
        print(f"Failed to fetch PDF from {url}: {e}")
        return None


def pdf_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    """Convert PDF pages to base64 images using parallel threads.

    Uses ThreadPoolExecutor with 10 workers for true parallel conversion.
    Returns list of base64 strings (PNG format).
    """
    # First, save to temp file for multi-threaded access
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        total_pages = len(doc)
        doc.close()

        def convert_page(page_num: int) -> tuple[int, str]:
            doc = fitz.open(tmp_path)
            page = doc[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            b64 = base64.b64encode(img_bytes).decode('utf-8')
            doc.close()
            return (page_num, b64)

        images = [None] * total_pages
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(convert_page, i): i for i in range(total_pages)}
            for future in as_completed(futures):
                page_num, b64 = future.result()
                images[page_num] = b64

        return images
    finally:
        # Clean up temp file
        import os as os_module
        try:
            os_module.unlink(tmp_path)
        except:
            pass


async def pdf_to_images_async(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
    """Async wrapper for pdf_to_images - runs in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, pdf_to_images, pdf_bytes, dpi)


async def parse_pdf_with_vlm(pdf_url: str) -> str:
    """Fetch PDF, convert to images, and extract text using VLM.

    Processes ALL pages with 15 concurrent VLM requests.
    Uses shared client for connection reuse.
    """
    # Fetch PDF
    pdf_bytes = await fetch_pdf(pdf_url)
    if not pdf_bytes:
        return ""

    # Convert ALL pages to images (parallel in thread pool)
    images = await pdf_to_images_async(pdf_bytes)
    if not images:
        return ""

    print(f"Processing {len(images)} pages from PDF with {VLM_CONCURRENCY} concurrent requests")

    vlm_model = os.getenv("OPENROUTER_MODEL_VLM", "qwen/qwen3-vl-235b-a22b-instruct")

    # Create semaphore inside async function to ensure correct event loop binding
    semaphore = asyncio.Semaphore(VLM_CONCURRENCY)

    async def extract_page(page_num: int, img_b64: str) -> tuple[int, str]:
        """Extract text from a single page image using shared client."""
        async with semaphore:
            print(f"  → Starting page {page_num + 1}/{len(images)}")
            try:
                response = await _openrouter_client.chat.completions.create(
                    model=vlm_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text content from this research paper page. Include section headers, body text, equations (describe them), figure captions, and table contents. Be thorough but concise."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=4000,
                    temperature=0,
                    timeout=120,
                )
                if response.choices:
                    print(f"  ✓ Page {page_num + 1} done")
                    return (page_num, response.choices[0].message.content)
                print(f"  ✗ Page {page_num + 1} - no response")
                return (page_num, "")
            except Exception as e:
                print(f"  ✗ Page {page_num + 1} - error: {e}")
                return (page_num, "")

    # Process ALL pages concurrently
    tasks = [extract_page(i, img) for i, img in enumerate(images)]
    results = await asyncio.gather(*tasks)

    # Sort by page number and combine
    results.sort(key=lambda x: x[0])
    combined = "\n\n".join([f"[Page {num+1}]\n{text}" for num, text in results if text])
    return combined


def _sync_web_search(query: str, max_results: int = 3) -> str:
    """Synchronous web search (runs in thread pool)."""
    return web_search.search_summary(query, max_results)


async def async_web_search(query: str, max_results: int = 3) -> str:
    """Async web search - runs in thread pool to not block."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _sync_web_search, query, max_results)


async def chat_with_context(
    message: str,
    papers: list[dict],
    paper_contents: dict[str, str],
    history: list[dict] | None = None,
    use_web_search: bool = True,
) -> str:
    """Chat with LLM using paper context and optional web search."""
    model = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

    # Build context from papers
    paper_context = []
    for p in papers:
        ctx = f"**{p['title']}** (arxiv:{p['arxiv_id']})\n"
        ctx += f"Abstract: {p['abstract']}\n"

        if p['arxiv_id'] in paper_contents:
            content = paper_contents[p['arxiv_id']]
            if len(content) > 15000:
                content = content[:15000] + "...[truncated]"
            ctx += f"\nFull Paper Content:\n{content}\n"

        paper_context.append(ctx)

    papers_text = "\n---\n".join(paper_context)

    # Optional web search (async - doesn't block)
    web_context = ""
    if use_web_search:
        web_results = await async_web_search(message, max_results=3)
        if web_results:
            web_context = f"\n\nWeb Search Results:\n{web_results}"

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

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    # Use shared client for LLM call too
    response = await _openrouter_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        timeout=120,
    )

    return response.choices[0].message.content
