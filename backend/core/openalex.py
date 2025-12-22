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
    """Batch fetch citation data - more efficient for many papers.

    Uses DOI filter since ids.arxiv is not a valid filter field.
    ArXiv DOI format: 10.48550/arXiv.{arxiv_id}
    """
    results = {aid: {"citation_count": 0, "references": []} for aid in arxiv_ids}

    if not arxiv_ids:
        return results

    # Convert arxiv IDs to DOIs (arxiv DOI format: 10.48550/arXiv.{id})
    # Strip version suffix (e.g., "2301.12345v1" -> "2301.12345")
    def to_doi(arxiv_id: str) -> str:
        # Remove version suffix if present
        clean_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        return f"https://doi.org/10.48550/arXiv.{clean_id}"

    arxiv_to_doi = {aid: to_doi(aid) for aid in arxiv_ids}
    doi_to_arxiv = {doi: aid for aid, doi in arxiv_to_doi.items()}

    try:
        # Query in batches of 50 (OpenAlex limit)
        batch_size = 50
        all_works = []

        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i:i + batch_size]
            dois = [to_doi(aid) for aid in batch]

            # Use pipe-separated DOIs for OR query
            works = Works().filter(doi="|".join(dois)).get()
            all_works.extend(works)

        for work in all_works:
            work_doi = work.get("doi", "")
            if work_doi and work_doi in doi_to_arxiv:
                arxiv_id = doi_to_arxiv[work_doi]
                results[arxiv_id] = {
                    "citation_count": work.get("cited_by_count", 0) or 0,
                    "references": []  # References would need separate calls
                }
            else:
                # Try matching via arxiv ID in ids field
                arxiv_url = work.get("ids", {}).get("arxiv", "")
                if arxiv_url:
                    arxiv_id_from_url = arxiv_url.replace("https://arxiv.org/abs/", "")
                    for aid in arxiv_ids:
                        if aid.split('v')[0] in arxiv_id_from_url or arxiv_id_from_url in aid:
                            results[aid] = {
                                "citation_count": work.get("cited_by_count", 0) or 0,
                                "references": []
                            }
                            break

    except Exception as e:
        print(f"OpenAlex batch query failed: {e}")

    return results
