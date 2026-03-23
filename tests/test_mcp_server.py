from __future__ import annotations

import pytest

from qudt_parsing import mcp_server


@pytest.mark.asyncio
async def test_get_entity_by_iri_builds_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, str] = {}

    async def fake_run_select(query: str, *, accept: str = mcp_server.JSON_ACCEPT):
        calls["query"] = query
        calls["accept"] = accept
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_run_select", fake_run_select)
    result = await mcp_server.get_entity_by_iri("http://example.org/x", limit=5)

    assert result == {"ok": True}
    assert "http://example.org/x" in calls["query"]
    assert "LIMIT 5" in calls["query"]
    assert calls["accept"] == mcp_server.JSON_ACCEPT


@pytest.mark.asyncio
async def test_get_entity_by_iri_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError):
        await mcp_server.get_entity_by_iri("http://example.org/x", limit=0)


@pytest.mark.asyncio
async def test_run_sparql_select_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_select(query: str, *, accept: str = mcp_server.JSON_ACCEPT):
        return {"query": query, "accept": accept}

    monkeypatch.setattr(mcp_server, "_run_select", fake_run_select)
    output = await mcp_server.run_sparql_select("SELECT * WHERE { ?s ?p ?o }")

    assert output["query"] == "SELECT * WHERE { ?s ?p ?o }"
    assert output["accept"] == mcp_server.JSON_ACCEPT


@pytest.mark.asyncio
async def test_list_known_prefixes_returns_expected_map() -> None:
    prefixes = await mcp_server.list_known_prefixes()
    assert prefixes["qudt"].startswith("http://qudt.org/")
    assert "rdf" in prefixes
