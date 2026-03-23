from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

DEFAULT_SPARQL_ENDPOINT = "https://www.qudt.org/fuseki/qudt/sparql"
JSON_ACCEPT = "application/sparql-results+json"
RDF_ACCEPT = "text/turtle"


class SparqlError(RuntimeError):
    """Raised when the SPARQL endpoint returns an error response."""


class SparqlClient:
    """Minimal async client for the QUDT Fuseki SPARQL endpoint."""

    def __init__(
        self,
        endpoint_url: str = DEFAULT_SPARQL_ENDPOINT,
        *,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True
            )
            self._owns_client = True
        return self._client

    async def select(
        self, query: str, *, accept: str = JSON_ACCEPT
    ) -> Mapping[str, Any]:
        response = await self._post_query(query, accept=accept)
        if "json" in accept:
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise SparqlError("Unexpected response payload from SPARQL endpoint")
            return payload
        return {"content_type": accept, "payload": response.text}

    async def construct(self, query: str, *, accept: str = RDF_ACCEPT) -> str:
        response = await self._post_query(query, accept=accept)
        return response.text

    async def _post_query(self, query: str, *, accept: str) -> httpx.Response:
        client = await self._get_client()
        response = await client.post(
            self.endpoint_url,
            content=query.encode("utf-8"),
            headers={
                "Content-Type": "application/sparql-query",
                "Accept": accept,
            },
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - narrow branch
            message = (
                f"SPARQL endpoint error {exc.response.status_code}: {exc.response.text}"
            )
            raise SparqlError(message) from exc
        except httpx.HTTPError as exc:
            raise SparqlError(f"SPARQL endpoint request failed: {exc}") from exc
        return response

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> SparqlClient:
        await self._get_client()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
