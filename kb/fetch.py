"""arXiv metadata fetching, OpenAlex citation filtering, and PDF download.

Pipeline (driven by scripts/download_papers.py):
  fetch_arxiv_metadata -> enrich_citations -> select_papers -> download_pdfs -> write_metadata_csv

Design notes:
- **Slicing**: the arXiv API does not page reliably past ~10k results per query, so the
  metadata pool is fetched per (category, year, month) slice, up to `kb.fetch.per_slice`
  records per slice (100 per page, >=3s between requests, tenacity retries). Each slice is
  cached to `raw/arxiv_slices/*.jsonl`, so an interrupted fetch resumes where it stopped.
  Cross-listed papers appear in several slices and are deduplicated by paper_id.
- **Citations**: OpenAlex has no working arXiv-DOI lookup for most preprints, so works are
  matched on `locations.landing_page_url = http://arxiv.org/abs/<id>` in batches of
  `kb.fetch.openalex_batch` (OR filter). Several OpenAlex works can map to one arXiv id
  (preprint + published version); the max cited_by_count is kept. Unmatched papers get
  `kb.fetch.citation_default` and are logged. Results are cached to
  `raw/openalex_citations.jsonl`. Anonymous OpenAlex use is capped (~1000 calls/day);
  set OPENALEX_API_KEY in .env for full runs.
- **Selection**: citation_min filter, then a per-year quota (target/years) filled by
  citation count, so recent low-citation years are not crowded out; unused quota is
  redistributed by global citation rank.
- **PDF download**: from export.arxiv.org (arXiv's endpoint for programmatic access) with a
  global rate limiter (one request start per `request_interval_s`, shared by
  `max_concurrency` workers). Existing files that look like valid PDFs are skipped
  (resumable); failures go to `raw/download_failures.jsonl`.

Output `raw/metadata.csv` columns:
  paper_id, arxiv_id, title, abstract, year, categories, citation_count, pdf_path
(`categories` is a JSON array string; `pdf_path` is relative to raw/, "" if not downloaded.)

Constraints: public data only; respect arXiv/OpenAlex terms; no hardcoded paths.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://export.arxiv.org/pdf/{arxiv_id}"
OPENALEX_API = "https://api.openalex.org/works"

METADATA_COLUMNS = [
    "paper_id", "arxiv_id", "title", "abstract", "year", "categories", "citation_count", "pdf_path",
]

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_VERSION_RE = re.compile(r"v\d+$")
_MIN_PDF_BYTES = 10_000


class RateLimiter:
    """Thread-safe limiter: at most one request start per `interval_s` across all threads."""

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval_s
        if delay > 0:
            time.sleep(delay)


class _RetryableHTTPError(Exception):
    """Transient failure (429/5xx/empty arXiv page) worth retrying."""

    def __init__(self, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retryable_status(resp: httpx.Response, service: str) -> None:
    """Raise _RetryableHTTPError for 429/5xx (carrying any Retry-After seconds)."""
    if resp.status_code == 429 or resp.status_code >= 500:
        try:
            retry_after = float(resp.headers.get("retry-after") or 0)
        except ValueError:
            retry_after = 0.0
        raise _RetryableHTTPError(f"{service} HTTP {resp.status_code}", retry_after)


def _backoff(multiplier: float, max_s: float):
    """Exponential backoff that never waits less than the server's Retry-After."""
    exp = wait_exponential(multiplier=multiplier, max=max_s)

    def wait(retry_state) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        return max(exp(retry_state), getattr(exc, "retry_after", 0.0) or 0.0)

    return wait


class ArxivFetchIncomplete(RuntimeError):
    """Some arXiv slices failed; selecting from a partial pool would skew the year balance."""


class OpenAlexQuotaError(RuntimeError):
    """OpenAlex daily quota exhausted (HTTP 429 with no remaining credits / long Retry-After)."""


class CitationLookupIncomplete(RuntimeError):
    """Some OpenAlex batches failed; selecting papers now would treat them as uncited."""


def _user_agent(contact_email: str) -> str:
    ua = "PaperOdyssey/0.1 (AI6127 coursework RAG corpus builder)"
    return f"{ua} mailto:{contact_email}" if contact_email else ua


def clean_setting(value: object) -> str:
    """Config values like `${OPENALEX_API_KEY}` stay literal when the env var is unset."""
    text = str(value or "").strip()
    return "" if text.startswith("${") else text


def paper_id_from_arxiv_id(arxiv_id: str) -> str:
    """Internal paper_id = arXiv id without version ("/" in old-style ids -> "_")."""
    return _VERSION_RE.sub("", arxiv_id.strip()).replace("/", "_")


def _norm_space(text: str) -> str:
    return " ".join((text or "").split())


# ---------------------------------------------------------------- arXiv metadata

def _month_slices(year_min: int, year_max: int) -> list[tuple[int, int]]:
    return [(y, m) for y in range(year_min, year_max + 1) for m in range(1, 13)]


def _slice_query(category: str, year: int, month: int) -> str:
    last_day = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).day
    return (
        f"cat:{category} AND submittedDate:[{year}{month:02d}010000 TO "
        f"{year}{month:02d}{last_day:02d}2359]"
    )


def _parse_atom_feed(xml_text: str) -> tuple[int, list[dict]]:
    root = ET.fromstring(xml_text)
    total_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    total = int(total_el.text) if total_el is not None and total_el.text else 0
    records = []
    for entry in root.findall("a:entry", _ATOM):
        raw_id = (entry.findtext("a:id", default="", namespaces=_ATOM) or "").strip()
        if "/abs/" not in raw_id:
            continue  # arXiv returns an "Error" entry for malformed queries
        arxiv_id = raw_id.split("/abs/", 1)[1]
        published = entry.findtext("a:published", default="", namespaces=_ATOM) or ""
        categories = [c.get("term") for c in entry.findall("a:category", _ATOM) if c.get("term")]
        primary = entry.find("arxiv:primary_category", _ATOM)
        if primary is not None and primary.get("term") in categories:
            categories.remove(primary.get("term"))
            categories.insert(0, primary.get("term"))
        records.append({
            "paper_id": paper_id_from_arxiv_id(arxiv_id),
            "arxiv_id": arxiv_id,
            "title": _norm_space(entry.findtext("a:title", default="", namespaces=_ATOM)),
            "abstract": _norm_space(entry.findtext("a:summary", default="", namespaces=_ATOM)),
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "categories": categories,
        })
    return total, records


def _fetch_slice(
    client: httpx.Client,
    limiter: RateLimiter,
    category: str,
    year: int,
    month: int,
    per_slice: int,
    page_size: int,
) -> list[dict]:
    query = _slice_query(category, year, month)

    @retry(  # arXiv throttles bursts with 429s that can last minutes: back off for up to ~10 min
        retry=retry_if_exception_type((_RetryableHTTPError, httpx.TransportError)),
        stop=stop_after_attempt(8),
        wait=_backoff(multiplier=5, max_s=300),
        reraise=True,
    )
    def page(start: int) -> tuple[int, list[dict]]:
        limiter.wait()
        resp = client.get(ARXIV_API, params={
            "search_query": query, "start": start, "max_results": page_size,
            "sortBy": "relevance", "sortOrder": "descending",
        })
        _retryable_status(resp, "arXiv")
        resp.raise_for_status()
        return _parse_atom_feed(resp.text)

    out: list[dict] = []
    start = 0
    while len(out) < per_slice:
        # arXiv intermittently returns an empty page inside a non-empty result set: retry a few times
        for _ in range(3):
            total, records = page(start)
            if records or start >= total:
                break
        if not records:
            if start == 0 and total > 0:
                raise RuntimeError(f"arXiv returned no entries for a non-empty slice (total={total})")
            if start < total:
                logger.warning("arXiv %s %d-%02d: empty page at start=%d; keeping %d records",
                               category, year, month, start, len(out))
            break
        out.extend(records)
        start += page_size
        if start >= total:
            break
    return out[:per_slice]


def fetch_arxiv_metadata(
    categories: list[str],
    year_min: int,
    year_max: int,
    max_papers: int,
    *,
    per_slice: int = 150,
    page_size: int = 100,
    request_interval_s: float = 3.0,
    cache_dir: str | Path | None = None,
    contact_email: str = "",
    seed: int = 42,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Fetch arXiv metadata, sliced by (category, year, month).

    Args:
        categories: e.g. ["cs.CL", "cs.AI", "cs.LG"].
        year_min/year_max: year range (inclusive).
        max_papers: metadata cap (before filtering); 0 = no cap. With a cap, slices are
            visited in a seeded random order so small trial runs still span years.
        per_slice: max records per (category, year, month) slice.
        page_size: records per arXiv API request (<= 100 recommended).
        request_interval_s: minimum seconds between API requests.
        cache_dir: per-slice JSONL cache directory (enables resume); None disables caching.
        contact_email: optional contact for the User-Agent (arXiv etiquette).
        seed: seed for the slice visiting order.
        allow_partial: when False (default), raise ArxivFetchIncomplete if any slice failed
            (after trying all of them, so successful slices are cached for the rerun).

    Returns:
        A DataFrame (paper_id, arxiv_id, title, abstract, year, categories[list]),
        deduplicated by paper_id (keeping the latest version), years within range.

    Raises:
        RuntimeError: when every slice failed and no data is available at all.
        ArxivFetchIncomplete: some slices failed and allow_partial is False.
    """
    slices = [(c, y, m) for c in categories for (y, m) in _month_slices(year_min, year_max)]
    if max_papers:
        random.Random(seed).shuffle(slices)
        per_slice = min(per_slice, max_papers)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    limiter = RateLimiter(request_interval_s)
    rows: list[dict] = []
    seen: set[str] = set()
    failed = 0
    headers = {"User-Agent": _user_agent(contact_email)}
    with httpx.Client(timeout=60, headers=headers, follow_redirects=True) as client:
        bar = tqdm(slices, desc="arXiv slices", unit="slice")
        for category, year, month in bar:
            cache_file = cache / f"{category}_{year}_{month:02d}_{per_slice}.jsonl" if cache else None
            if cache_file and cache_file.exists():
                records = [json.loads(line) for line in cache_file.read_text("utf-8").splitlines() if line]
            else:
                try:
                    records = _fetch_slice(client, limiter, category, year, month, per_slice, page_size)
                except Exception as exc:  # noqa: BLE001 (log the slice and keep going)
                    failed += 1
                    logger.warning("arXiv slice %s %d-%02d failed: %s", category, year, month, exc)
                    continue
                if cache_file:
                    tmp = cache_file.with_suffix(".tmp")
                    tmp.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")
                    tmp.replace(cache_file)
            for r in records:
                if r["paper_id"] not in seen:
                    seen.add(r["paper_id"])
                    rows.append(r)
            bar.set_postfix(papers=len(rows), failed=failed)
            if max_papers and len(rows) >= max_papers:
                break

    if not rows:
        raise RuntimeError(f"no arXiv metadata fetched ({failed} slices failed)")
    if failed and not allow_partial:
        raise ArxivFetchIncomplete(
            f"{failed} arXiv slices failed; rerun to retry them (fetched slices are cached)"
        )
    df = pd.DataFrame(rows)
    df = df[df["year"].between(year_min, year_max)]
    if max_papers:
        df = df.head(max_papers)
    logger.info("fetched %d unique papers from %d slices (%d failed)", len(df), len(slices), failed)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- OpenAlex citations

def enrich_citations(
    df: pd.DataFrame,
    *,
    batch_size: int = 50,
    api_key: str = "",
    contact_email: str = "",
    citation_default: int = 0,
    cache_path: str | Path | None = None,
    request_interval_s: float = 0.2,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Fill `citation_count` from OpenAlex (batched landing-page-URL match).

    OpenAlex stores arXiv landing pages as `http://arxiv.org/abs/<id>` (versionless); the
    https form does not match, so the query uses http.

    Args:
        df: output of fetch_arxiv_metadata.
        batch_size: arXiv ids per OpenAlex request (OR filter; <= 50 keeps URLs short).
        api_key: optional OpenAlex API key (lifts the anonymous daily cap).
        contact_email: optional `mailto` for OpenAlex's polite pool.
        citation_default: value for papers OpenAlex does not know (logged).
        cache_path: JSONL cache {"paper_id", "citation_count"|null} (enables resume).
        request_interval_s: minimum seconds between OpenAlex requests.
        allow_missing: when False (default), raise CitationLookupIncomplete if any batch
            failed, so a transient outage never silently ranks papers as uncited. Resolved
            lookups stay cached either way; rerunning retries only the failed ones.

    Returns:
        A copy of df with an int `citation_count` column.

    Raises:
        OpenAlexQuotaError: the daily quota ran out (lookups so far are cached; rerun later or
            set OPENALEX_API_KEY).
        CitationLookupIncomplete: some batches failed and allow_missing is False.
    """
    api_key, contact_email = clean_setting(api_key), clean_setting(contact_email)
    cache = Path(cache_path) if cache_path else None
    known: dict[str, int | None] = {}
    if cache and cache.exists():
        for line in cache.read_text("utf-8").splitlines():
            try:
                rec = json.loads(line)
                known[rec["paper_id"]] = rec["citation_count"]
            except (json.JSONDecodeError, KeyError):
                continue  # torn line from an interrupted run; that id is looked up again

    todo = [pid for pid in df["paper_id"] if pid not in known]
    limiter = RateLimiter(request_interval_s)
    params_base: dict[str, str] = {"select": "id,cited_by_count,locations", "per-page": "200"}
    if api_key:
        params_base["api_key"] = api_key
    if contact_email:
        params_base["mailto"] = contact_email

    @retry(
        retry=retry_if_exception_type((_RetryableHTTPError, httpx.TransportError)),
        stop=stop_after_attempt(6),
        wait=_backoff(multiplier=2, max_s=120),
        reraise=True,
    )
    def lookup(client: httpx.Client, batch: list[str]) -> dict[str, int]:
        limiter.wait()
        urls = "|".join(f"http://arxiv.org/abs/{pid.replace('_', '/')}" for pid in batch)
        resp = client.get(OPENALEX_API, params={**params_base, "filter": f"locations.landing_page_url:{urls}"})
        if resp.status_code == 429:
            remaining = resp.headers.get("x-ratelimit-remaining")
            try:
                retry_after = float(resp.headers.get("retry-after") or 0)
            except ValueError:
                retry_after = 0.0
            if remaining == "0" or retry_after > 120:
                raise OpenAlexQuotaError(
                    "OpenAlex daily quota exhausted (anonymous use is capped at ~1000 calls/day). "
                    "Lookups so far are cached: rerun after the reset, or set OPENALEX_API_KEY in .env."
                )
        _retryable_status(resp, "OpenAlex")
        resp.raise_for_status()
        wanted = set(batch)
        found: dict[str, int] = {}
        for work in resp.json().get("results", []):
            for loc in work.get("locations") or []:
                url = loc.get("landing_page_url") or ""
                if "arxiv.org/abs/" not in url:
                    continue
                pid = paper_id_from_arxiv_id(url.split("arxiv.org/abs/", 1)[1])
                if pid in wanted:
                    found[pid] = max(found.get(pid, 0), int(work.get("cited_by_count") or 0))
        return found

    failed_batches = 0
    if todo:
        cache_fh = cache.open("a", encoding="utf-8") if cache else None
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        logger.info("OpenAlex: %d cached, %d to look up in %d calls", len(known), len(todo), len(batches))
        try:
            with httpx.Client(timeout=60, headers={"User-Agent": _user_agent(contact_email)}) as client:
                for batch in tqdm(batches, desc="OpenAlex citations", unit="batch"):
                    try:
                        found = lookup(client, batch)
                    except OpenAlexQuotaError:
                        raise
                    except Exception as exc:  # noqa: BLE001 (leave the batch for the next run)
                        failed_batches += 1
                        logger.warning("OpenAlex batch failed (%d ids): %s", len(batch), exc)
                        continue
                    for pid in batch:
                        known[pid] = found.get(pid)
                        if cache_fh:
                            cache_fh.write(json.dumps({"paper_id": pid, "citation_count": known[pid]}) + "\n")
                    if cache_fh:
                        cache_fh.flush()
        finally:
            if cache_fh:
                cache_fh.close()

    if failed_batches and not allow_missing:
        raise CitationLookupIncomplete(
            f"{failed_batches} OpenAlex batches failed; rerun to retry them (resolved lookups are cached)"
        )
    out = df.copy()
    counts = out["paper_id"].map(known)
    missing = int(counts.isna().sum())
    if missing:
        logger.info("%d/%d papers are unknown to OpenAlex (or unresolved); citation_count = %d",
                    missing, len(out), citation_default)
    out["citation_count"] = counts.fillna(citation_default).astype(int)
    return out


# ---------------------------------------------------------------- selection

def select_papers(
    df: pd.DataFrame,
    *,
    citation_min: int,
    target_papers: int,
    year_min: int,
    year_max: int,
    categories: list[str],
) -> pd.DataFrame:
    """Apply kb.filter.* rules and pick up to `target_papers` papers.

    Year-stratified: each year gets target/years slots filled by citation count (desc);
    unused slots are then filled from the remaining papers by global citation rank.
    Ties are broken by paper_id for determinism.
    """
    cats = set(categories)
    pool = df[
        df["year"].between(year_min, year_max)
        & (df["citation_count"] >= citation_min)
        & df["categories"].map(lambda cs: bool(cats.intersection(cs)))
    ].sort_values(["citation_count", "paper_id"], ascending=[False, True])
    if len(pool) <= target_papers:
        return pool.reset_index(drop=True)

    quota = math.floor(target_papers / (year_max - year_min + 1))
    chosen = pool.groupby("year", group_keys=False).head(quota)
    rest = pool.drop(chosen.index).head(target_papers - len(chosen))
    selected = pd.concat([chosen, rest]).sort_values(["citation_count", "paper_id"], ascending=[False, True])
    return selected.reset_index(drop=True)


# ---------------------------------------------------------------- PDF download

def _is_valid_pdf(path: Path) -> bool:
    try:
        if path.stat().st_size < _MIN_PDF_BYTES:
            return False
        with path.open("rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def download_pdfs(
    df: pd.DataFrame,
    raw_dir: str | Path,
    *,
    max_concurrency: int = 4,
    request_interval_s: float = 3.0,
    contact_email: str = "",
    failures_path: str | Path | None = None,
) -> pd.DataFrame:
    """Download PDFs to raw/pdf/<paper_id>.pdf (resumable, rate-limited).

    Returns:
        A copy of df with `pdf_path` (relative to raw_dir; "" when the download failed).
    """
    raw = Path(raw_dir)
    pdf_dir = raw / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(request_interval_s)
    headers = {"User-Agent": _user_agent(clean_setting(contact_email))}

    @retry(
        retry=retry_if_exception_type((_RetryableHTTPError, httpx.TransportError)),
        stop=stop_after_attempt(6),
        wait=_backoff(multiplier=5, max_s=300),
        reraise=True,
    )
    def fetch_one(client: httpx.Client, arxiv_id: str, dest: Path) -> None:
        limiter.wait()
        tmp = dest.with_suffix(".part")
        with client.stream("GET", ARXIV_PDF.format(arxiv_id=arxiv_id)) as resp:
            _retryable_status(resp, "arXiv PDF")
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(1 << 16):
                    fh.write(chunk)
        if not _is_valid_pdf(tmp):
            tmp.unlink(missing_ok=True)
            raise ValueError("response is not a PDF (withdrawn paper or source-only submission?)")
        tmp.replace(dest)

    paths: dict[str, str] = {}
    todo = []
    for pid, aid in zip(df["paper_id"], df["arxiv_id"], strict=True):
        dest = pdf_dir / f"{pid}.pdf"
        if _is_valid_pdf(dest):
            paths[pid] = str(dest.relative_to(raw))
        else:
            todo.append((pid, aid, dest))
    logger.info("PDFs: %d already present, %d to download", len(paths), len(todo))

    n_failed = 0
    fail_fh = None  # opened on the first failure
    with httpx.Client(timeout=120, headers=headers, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as pool:
            futures = {pool.submit(fetch_one, client, aid, dest): (pid, aid, dest) for pid, aid, dest in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="PDF download", unit="pdf"):
                pid, aid, dest = futures[fut]
                try:
                    fut.result()
                    paths[pid] = str(dest.relative_to(raw))
                except Exception as exc:  # noqa: BLE001 (log it now so it survives Ctrl+C, then continue)
                    n_failed += 1
                    if failures_path and fail_fh is None:
                        fail_fh = Path(failures_path).open("a", encoding="utf-8")
                    if fail_fh:
                        fail_fh.write(json.dumps({"paper_id": pid, "arxiv_id": aid,
                                                  "error": f"{type(exc).__name__}: {exc}",
                                                  "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                        fail_fh.flush()
    if fail_fh:
        fail_fh.close()

    if n_failed:
        logger.warning("%d PDF downloads failed (see %s); rerun with --resume to retry", n_failed, failures_path)

    out = df.copy()
    out["pdf_path"] = out["paper_id"].map(paths).fillna("")
    return out


# ---------------------------------------------------------------- metadata.csv

def write_metadata_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Write raw/metadata.csv (categories serialized as a JSON array string)."""
    out = df.copy()
    for col in METADATA_COLUMNS:
        if col not in out.columns:
            out[col] = "" if col == "pdf_path" else None
    out["categories"] = out["categories"].map(lambda c: c if isinstance(c, str) else json.dumps(list(c)))
    out[METADATA_COLUMNS].to_csv(path, index=False)


def read_metadata_csv(path: str | Path) -> pd.DataFrame:
    """Read raw/metadata.csv back (categories -> list, pdf_path NaN -> "")."""
    df = pd.read_csv(path, dtype={"paper_id": str, "arxiv_id": str, "pdf_path": str})
    df["categories"] = df["categories"].map(json.loads)
    df["pdf_path"] = df["pdf_path"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    return df
