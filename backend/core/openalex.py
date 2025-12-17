"""OpenAlex client for citation data."""

import pyalex
from pyalex import Works

# Be polite - set email for OpenAlex
pyalex.config.email = "papermap@example.com"


def get_citation_data(arxiv_ids: list[str]) -> dict[str, dict]:
    """Fetch citation data for a list of arxiv IDs.

    Returns dict mapping arxiv_id to:
        {
            "citation_count": int,
            "references": list[str]  # arxiv IDs of papers this one cites
        }
    """
    results = {}

    for arxiv_id in arxiv_ids:
        try:
            # Query OpenAlex by arxiv ID
            work = Works()["arxiv:" + arxiv_id.replace("v", ".").split(".")[0] + arxiv_id.replace("v", ".").split(".")[1].rstrip("0123456789")]
        except Exception:
            try:
                # Try with full ID
                work = Works()["arxiv:" + arxiv_id]
            except Exception:
                results[arxiv_id] = {"citation_count": 0, "references": []}
                continue

        try:
            citation_count = work.get("cited_by_count", 0) or 0

            # Get references that are also arxiv papers
            ref_arxiv_ids = []
            for ref in work.get("referenced_works", []) or []:
                try:
                    ref_work = Works()[ref]
                    # Check if it has an arxiv ID
                    if ref_work and ref_work.get("ids", {}).get("arxiv"):
                        ref_arxiv = ref_work["ids"]["arxiv"].replace("https://arxiv.org/abs/", "")
                        ref_arxiv_ids.append(ref_arxiv)
                except Exception:
                    continue

            results[arxiv_id] = {
                "citation_count": citation_count,
                "references": ref_arxiv_ids[:20]  # Limit to avoid too many
            }
        except Exception:
            results[arxiv_id] = {"citation_count": 0, "references": []}

    return results


def get_citation_data_batch(arxiv_ids: list[str]) -> dict[str, dict]:
    """Batch fetch citation data - more efficient for many papers."""
    results = {aid: {"citation_count": 0, "references": []} for aid in arxiv_ids}

    # Build filter for batch query
    try:
        # Query works that have these arxiv IDs
        works = Works().filter(ids={"arxiv": "|".join(arxiv_ids)}).get()

        for work in works:
            arxiv_url = work.get("ids", {}).get("arxiv", "")
            if arxiv_url:
                arxiv_id = arxiv_url.replace("https://arxiv.org/abs/", "")
                # Find matching ID in our list
                for aid in arxiv_ids:
                    if aid in arxiv_id or arxiv_id in aid:
                        results[aid] = {
                            "citation_count": work.get("cited_by_count", 0) or 0,
                            "references": []  # Would need separate calls for refs
                        }
                        break
    except Exception as e:
        print(f"OpenAlex batch query failed: {e}")

    return results
