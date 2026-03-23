from __future__ import annotations

import httpx
import pytest

from qudt_parsing.sparql_client import SparqlClient, SparqlError


@pytest.mark.asyncio
async def test_select_posts_query_and_returns_json() -> None:
    query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["accept"] = request.headers.get("accept", "")
        captured["content_type"] = request.headers.get("content-type", "")
        captured["body"] = request.content.decode()
        return httpx.Response(
            200,
            json={"head": {"vars": ["s"]}, "results": {"bindings": []}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sparql = SparqlClient("https://example.invalid/sparql", client=client)
        result = await sparql.select(query)

    assert result["results"]["bindings"] == []
    assert captured["method"] == "POST"
    assert captured["body"] == query
    assert captured["accept"] == "application/sparql-results+json"
    assert captured["content_type"] == "application/sparql-query"


@pytest.mark.asyncio
async def test_select_raises_on_http_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sparql = SparqlClient("https://example.invalid/sparql", client=client)
        with pytest.raises(SparqlError):
            await sparql.select("SELECT * WHERE { ?s ?p ?o }")
