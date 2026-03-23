from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from qudt_parsing.sparql_client import (
    DEFAULT_SPARQL_ENDPOINT,
    JSON_ACCEPT,
    SparqlClient,
)

load_dotenv()

mcp = FastMCP(
    "qudt-sparql",
    "MCP server exposing the QUDT Fuseki SPARQL endpoint.",
)

PREFIXES: dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "qudt": "http://qudt.org/schema/qudt/",
    "unit": "http://qudt.org/vocab/unit/",
    "quantitykind": "http://qudt.org/vocab/quantitykind/",
    "prefix": "http://qudt.org/vocab/prefix/",
}
DEFAULT_LIMIT = 200


def _endpoint() -> str:
    return os.getenv("QUDT_SPARQL_ENDPOINT", DEFAULT_SPARQL_ENDPOINT)


async def _run_select(query: str, *, accept: str = JSON_ACCEPT) -> dict[str, Any]:
    async with SparqlClient(_endpoint()) as client:
        data = await client.select(query, accept=accept)
    return dict(data)


def _build_entity_lookup_query(iri: str, *, limit: int) -> str:
    return (
        "\n".join(
            [
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>",
                "SELECT ?predicate ?object",
                "WHERE {",
                f"  <{iri}> ?predicate ?object .",
                "}",
                f"LIMIT {limit}",
            ]
        )
        + "\n"
    )


@mcp.tool()
async def run_sparql_select(query: str, accept: str = JSON_ACCEPT) -> dict[str, Any]:
    """Run an arbitrary SPARQL SELECT query against QUDT."""

    return await _run_select(query, accept=accept)


@mcp.tool()
async def get_entity_by_iri(iri: str, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Return predicate/object pairs for a given IRI."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    query = _build_entity_lookup_query(iri, limit=limit)
    return await _run_select(query)


@mcp.tool()
async def list_known_prefixes() -> dict[str, str]:
    """Return a small prefix map useful for building queries."""

    return dict(PREFIXES)


if __name__ == "__main__":
    print("Starting QUDT MCP server...")
    mcp.run()
