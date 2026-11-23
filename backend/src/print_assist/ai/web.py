"""Clean web search and page-fetch helpers shared by the in-app chat and MCP server.

Uses documented public APIs (Brave Search API when `BRAVE_API_KEY` is configured, DuckDuckGo Instant
Answer JSON API, and Wikipedia MediaWiki Search API) so every AI harness can research characters,
flexi mechanisms, real-world dimensions, and hardware specs without bot-block failures.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any
from urllib.parse import quote

import httpx

_UA = "3DPrintAssist/1.0 (https://github.com/3d-print-assist)"


def _strip_html(raw: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw or ""))).strip()


def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search public web & encyclopedic sources for characters, 3D printing mechanisms, or hardware specs."""
    q = (query or "").strip()
    if not q:
        return {"error": "query is required"}
    limit = max(1, min(int(max_results), 10))
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    def _add(title: str, url: str, snippet: str) -> None:
        t = _strip_html(title)
        s = _strip_html(snippet)
        u = (url or "").strip()
        if not t or not s or (u and u in seen_urls):
            return
        if u:
            seen_urls.add(u)
        results.append({"title": t, "url": u, "snippet": s[:500]})

    # 1. Optional Brave Web Search API (full live web results when user has BRAVE_API_KEY set)
    brave_key = os.environ.get("BRAVE_API_KEY", "").strip()
    if brave_key:
        try:
            r = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": q, "count": limit},
                headers={"Accept": "application/json", "X-Subscription-Token": brave_key, "User-Agent": _UA},
                timeout=10,
            )
            if r.status_code == 200:
                for item in r.json().get("web", {}).get("results", [])[:limit]:
                    _add(item.get("title", ""), item.get("url", ""), item.get("description", ""))
        except Exception:  # noqa: BLE001
            pass

    # 2. DuckDuckGo Instant Answer JSON API (documented public API)
    if len(results) < limit:
        try:
            r = httpx.get(
                "https://api.duckduckgo.com/",
                params={"q": q, "format": "json", "no_html": 1, "skip_disambig": 1},
                headers={"User-Agent": _UA},
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json()
                if data.get("AbstractText"):
                    _add(data.get("Heading") or q, data.get("AbstractURL") or "", data["AbstractText"])
                for topic in data.get("RelatedTopics", []):
                    if len(results) >= limit:
                        break
                    if isinstance(topic, dict) and topic.get("Text"):
                        _add(topic["Text"].split(" - ", 1)[0][:80], topic.get("FirstURL", ""), topic["Text"])
        except Exception:  # noqa: BLE001
            pass

    # 3. Wikipedia MediaWiki Search API (rich factual descriptions of characters, creatures, movies, standards)
    if len(results) < limit:
        try:
            r = httpx.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": q,
                    "srlimit": limit,
                    "format": "json",
                },
                headers={"User-Agent": _UA},
                timeout=10,
            )
            if r.status_code == 200:
                hits = r.json().get("query", {}).get("search", [])
                for h in hits:
                    if len(results) >= limit:
                        break
                    title = str(h.get("title") or "")
                    slug = quote(title.replace(" ", "_"))
                    _add(title, f"https://en.wikipedia.org/wiki/{slug}", str(h.get("snippet") or ""))
        except Exception:  # noqa: BLE001
            pass

    return {"query": q, "results": results[:limit]}


def fetch_url(url: str) -> dict[str, Any]:
    """Fetch and extract readable plain text from an HTTP/HTTPS URL."""
    target = (url or "").strip()
    if not target.startswith(("http://", "https://")):
        return {"error": "url must start with http:// or https://"}
    try:
        r = httpx.get(target, headers={"User-Agent": _UA}, follow_redirects=True, timeout=15)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"fetch_url failed: {exc}"}
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.S | re.I)
    text = _strip_html(text)
    return {"url": str(r.url), "content": text[:6000]}
