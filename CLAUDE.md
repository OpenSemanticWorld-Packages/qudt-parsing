# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project parses the QUDT (Quantities, Units, Dimensions and Types) ontology and converts it into Open Semantic Lab (OSL) compatible data structures. The main purpose is to fill gaps in the QUDT ontology by generating missing units and enriching quantity kind definitions.

## Environment Setup

1. **Required Environment Variables** (create `.env` from `.env_example`):
   - `OSL_DOMAIN` - Your OSL wiki domain
   - `OSL_CRED_FP` - Path to accounts.pwd.yaml credentials file
   - Azure OpenAI credentials (if using Azure):
     - `AZURE_OPENAI_API_KEY`
     - `AZURE_OPENAI_ENDPOINT`
     - `AZURE_OPENAI_API_VERSION`
   - OpenAI credentials (if using OpenAI directly):
     - `OPENAI_PROJECT_ID`
     - `OPENAI_API_KEY`

2. **Package Manager**: This project uses `uv` for dependency management
   ```bash
   # Install dependencies
   uv sync

   # Install dev dependencies
   uv sync --group dev
   ```

## Development Commands

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_sparql_client.py

# Run with coverage
uv run pytest --cov=src/qudt_parsing --cov-report=term-missing
```

### Linting and Formatting
```bash
# Run ruff linter
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Type Checking
```bash
uv run mypy src/qudt_parsing
```

### Running the Main Script
```bash
# WARNING: main.py is designed for direct execution only, NOT for import
uv run python src/qudt_parsing/main.py
```

### MCP Server
The project includes an MCP (Model Context Protocol) server for SPARQL queries:
```bash
# Run the MCP server
uv run python -m qudt_parsing.mcp_server
```

The server is configured in `.mcp.json` for use with Claude Code.

## Architecture

### Core Data Flow

1. **Ontology Loading** (`prepare_all_ontologies()`):
   - Fetches QUDT ontology from remote URL (https://qudt.org/3.1.4/qudt-all.ttl)
   - Parses Turtle format using rdflib
   - Converts to JSON-LD using pyld
   - Caches to `data/qudt_dump.json`

2. **Index Building** (`build_indices()`):
   - Creates `id_dict`: Maps entity IDs to their full property dictionaries
   - Creates `id_to_index`: Maps entity IDs to their position in @graph array
   - Creates `type_dict`: Groups entities by rdf:type
   - Creates `type_index`: Maps types to list of entity indices

3. **Unit Processing Pipeline**:
   - **Classification** (`classify_and_enrich_qudt_units()`): Categorizes units as:
     - Base non-prefixed (e.g., `unit:M`)
     - Prefixed (e.g., `unit:KiloM`)
     - Non-prefixed composed (e.g., `unit:M-PER-SEC`)
     - Prefixed composed (e.g., `unit:KiloM-PER-SEC`)
   - **Gap Filling**: Identifies missing non-prefixed base units for prefixed composed units
   - **LLM Generation**: Uses Azure OpenAI to generate missing unit definitions
   - **Consistency Fixing**: Corrects inconsistencies in original QUDT data

4. **Quantity Kind Processing** (`create_quantity_value_types_and_characteristic_types()`):
   - Maps quantity kinds to applicable units
   - Creates OSL-compatible QuantityKind entities
   - **Known Issue**: 192 out of 1,214 quantity kinds lack applicable units in QUDT
   - These include specialized categories: pressure-based, temperature-based, inverse, squared, and domain-specific measurements

### Key Data Structures

**`ontologies` dict** - Central data structure containing:
```python
{
    "qudt": {
        "url": str,              # Source URL
        "format": str,           # "turtle"
        "context": dict,         # JSON-LD context with prefixes
        "dump_fp": Path,         # Cache file path
        "graph": Graph | None,   # rdflib Graph object
        "jsonld": dict | None,   # Full JSON-LD representation
        "id_dict": dict,         # Entity ID → properties
        "id_to_index": dict,     # Entity ID → @graph array index
        "type_dict": dict,       # Type → list of entities
        "type_index": dict,      # Type → list of indices
        "ids": list,             # All entity IDs
    }
}
```

### Function Call Tracking

The codebase uses a decorator-based system for tracking function dependencies:
- `@log_call` decorator logs all function invocations
- `FunctionCallHistory` class maintains call history
- `has_required_calls()` validates that prerequisite functions were executed
- This ensures the processing pipeline executes in correct order

### Module Organization

- **`main.py`**: Main processing script (execution only, not importable)
  - Contains all ontology parsing and entity creation logic
  - Uses decorators for dependency tracking
  - Directly modifies global `ontologies` dict

- **`ontology.py`**: Data models for ontology structures using Pydantic

- **`utility.py`**: Currently minimal/empty

- **`sparql_client.py`**: Async SPARQL client for querying QUDT Fuseki endpoint
  - Default endpoint: https://www.qudt.org/fuseki/qudt/sparql
  - Supports SELECT and CONSTRUCT queries

- **`mcp_server.py`**: MCP server exposing SPARQL tools
  - `run_sparql_select()`: Run arbitrary SPARQL queries
  - `get_entity_by_iri()`: Fetch entity properties by IRI
  - `list_known_prefixes()`: Get common QUDT prefixes

- **`ensure_dependencies.py`**: Fetches required OSL schemas from wiki
  - Defines dependency mapping for OSL entity types
  - Auto-downloads schemas on import

## Critical Design Decisions

### Why main.py Cannot Be Imported
The script enforces `if __name__ != "__main__": raise RuntimeError()` because:
- It operates on module-level global state (`ontologies` dict)
- Functions have ordering dependencies tracked via decorators
- It performs expensive operations (fetching/parsing large ontologies)
- It's designed as a pipeline, not a library

### Gap Filling Strategy
The project identifies two types of gaps in QUDT:

1. **Missing Units**: Prefixed composed units without base units (e.g., `KiloGM-PER-MOL-K` exists but `GM-PER-MOL-K` doesn't)
   - Solution: Use LLM to generate missing base unit definitions
   - Store in `data/qudt_missing_units.json`

2. **Quantity Kinds Without Units**: 192 quantity kinds have no applicable units
   - Categories: pressure-based, temperature-based, inverse, squared, thresholds
   - These require manual curation or dimensional analysis to synthesize appropriate units

### QUDT Naming Conventions
- Unit IDs: `unit:{PREFIX}{BASE}[-PER-{DENOM}]`
  - Example: `unit:KiloGM-PER-MOL-K` = kilograms per mole kelvin
- Quantity Kind IDs: `quantitykind:{Name}`
- Prefixes: `prefix:{PrefixName}` (Decimal: Kilo, Mega; Binary: Kibi, Mebi)

## Data Directory

- `qudt_dump.json`: Cached QUDT ontology in JSON-LD format
- `qudt_missing_units.json`: LLM-generated missing unit definitions
- `qudt_dump.enriched.json`: Final enriched ontology output
- Other dumps: `om2_dump.json`, `wikidata_dump.json`, `sdf_prefixes_dump.json`

## Common Issues

### Error: "No main unit found for quantity kind X"
This occurs when processing quantity kinds without applicable units in QUDT. The code currently raises `ValueError` at line ~2096 in main.py. Options:
- Skip the quantity kind (continue)
- Log warning instead of raising
- Synthesize units based on dimensional analysis

### Inconsistent QUDT Units
Some QUDT units have prefix in ID but lack `qudt:prefix` property, or vice versa. The `fix_inconsistent_units()` function handles known cases.

## Testing

Tests use `pytest` with async support (`pytest-asyncio`):
- `test_sparql_client.py`: Tests SPARQL client functionality
- `test_mcp_server.py`: Tests MCP server tools

Mock external dependencies when testing to avoid network calls.
