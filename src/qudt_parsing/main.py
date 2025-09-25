"""DISCLAIMER: This is a script for direct execution only! It is not meant to be
imported.
It loads the QUDT dump from a URL, processes it, and saves it as JSON-LD.
This script uses rdflib to parse the Turtle format and pyld to compact the
JSON-LD. It also provides functions to save and load JSON-LD data from files.

Design choices:
- The script operates on central data structures (ontologies dict) to keep track of
  the changes made to be able to derive a processing pipeline for Open Semantic Lab
  compatible QuantityUnits
- Function call logging and dependency checking is implemented via a decorator and a
  FunctionCallHistory class
-
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import uuid as uuid_module
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser

# from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from openai import Timeout
from osw.model import entity as model
from pydantic import BaseModel, Field
from pyld import jsonld as jsonld_module
from rdflib import Graph

if __name__ != "__main__":
    raise RuntimeError("This module is not intended to be imported.")

_logger = logging.getLogger(__name__)

ENV_FP = Path(__file__).parents[2] / ".env"
load_dotenv(ENV_FP)

LLM_MODEL = "gpt-5-2025-08-07"

osl_domain = os.getenv("OSL_DOMAIN")

this_file = Path(__file__)
project_root = this_file.parents[2]
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)

missing_units_fp = data_dir / "qudt_missing_units.json"

# llm = ChatOpenAI(model=LLM_MODEL, temperature=0, max_retries=3)
print("OpenAI configuration:")
print("Endpoint:", os.environ["AZURE_OPENAI_ENDPOINT"])
print("API version:", os.environ["AZURE_OPENAI_API_VERSION"])
# llm = AzureChatOpenAI(
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_version=os.environ["AZURE_OPENAI_API_VERSION"],
#     model="gpt-5-2025-08-07",
#     temperature=0,
#     max_retries=3,
#     timeout=300,
# )
#
# messages = [
#     (
#         "system",
#         "You are a helpful translator. Translate the user sentence to French.",
#     ),
#     ("human", "I love programming."),
# ]
#
# result = llm.invoke(messages)
# print("LLM-Testantwort:", result)


class FunctionCallHistory:
    def __init__(self):
        self._calls = []

    def add_call(self, func_name: str, args: dict[str, Any], result: Any):
        self._calls.append({"function": func_name, "args": args, "result": result})

    def get_history(self) -> list[dict[str, Any]]:
        return self._calls

    def call_in_history(self, func_name: str) -> bool:
        return any(call["function"] == func_name for call in self._calls)

    def has_required_calls(
        self,
        required_funcs: list[str | Callable],
        warn: bool = True,
        raise_err: bool = True,
    ) -> bool:
        required_funcs = [
            func if isinstance(func, str) else func.__name__ for func in required_funcs
        ]
        called_funcs = {call["function"] for call in self._calls}
        missing = {func for func in required_funcs if func not in called_funcs}
        result = all(func in called_funcs for func in required_funcs)
        if warn and missing:
            _logger.warning("Function calls missing: %s", ", ".join(missing))
        if raise_err and missing:
            raise RuntimeError(f"Function calls missing: {', '.join(missing)}")
        return result


func_log = FunctionCallHistory()


def log_call(func):
    """Decorator to log function calls."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        func_log.add_call(func.__name__, {"args": args, "kwargs": kwargs}, result)
        return result

    return wrapper


class Ontology(TypedDict):
    context: dict[str, str]
    format: str
    url: str
    dump_fp: str | Path
    graph: Graph | None
    jsonld: dict | None
    id_dict: dict[str, dict[str, Any]] | None
    id_to_index: dict[str, int] | None
    type_dict: dict[str, list[dict[str, Any]]] | None
    type_index: dict[str, list[int]] | None
    ids: list[str] | None


ontologies: dict[str, Ontology] = {
    "qudt": {
        "url": "https://qudt.org/3.1.4/qudt-all.ttl",
        "format": "turtle",
        "context": {
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "constant": "http://qudt.org/vocab/constant/",
            "cur": "http://qudt.org/vocab/currency/",
            "dc": "http://purl.org/dc/elements/1.1/",
            "dcterms": "http://purl.org/dc/terms/",
            "dtype": "http://www.linkedmodel.org/schema/dtype#",
            "prefix": "http://qudt.org/vocab/prefix/",
            "prov": "http://www.w3.org/ns/prov#",
            "qkdv": "http://qudt.org/vocab/dimensionvector/",
            "quantitykind": "http://qudt.org/vocab/quantitykind/",
            "qudt": "http://qudt.org/schema/qudt/",
            "qudt_type": "http://qudt.org/vocab/type/",
            "sh": "http://www.w3.org/ns/shacl#",
            "si_constant": "https://si-digital-framework.org/constants/",
            "si_prefix": "https://si-digital-framework.org/SI/prefixes/",
            "si_quantity": "https://si-digital-framework.org/quantities/",
            "si_unit": "https://si-digital-framework.org/SI/units/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "soqk": "http://qudt.org/vocab/soqk/",
            "sou": "http://qudt.org/vocab/sou/",
            "unit": "http://qudt.org/vocab/unit/",
            "vaem": "http://www.linkedmodel.org/schema/vaem#",
            "voag": "http://voag.linkedmodel.org/schema/voag#",
        },
        "dump_fp": data_dir / "qudt_dump.json",
        "graph": None,
        "jsonld": None,
        "id_dict": None,
        "id_to_index": None,
        "type_dict": None,
        "type_index": None,
        "ids": None,
    },
    "wikidata": {
        "url": "https://query.wikidata.org/sparql",
        "format": "xml",
        "context": {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        },
        "dump_fp": data_dir / "wikidata_dump.json",
        "graph": None,
        "jsonld": None,
        "id_dict": None,
        "id_to_index": None,
        "type_dict": None,
        "type_index": None,
        "ids": None,
    },  # todo: wiki data dump is not really useful
    "om2": {
        "url": "https://raw.githubusercontent.com/HajoRijgersberg/OM/refs/heads"
        "/master/om-2.0.rdf",
        "format": "xml",
        "context": {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "dc": "http://purl.org/dc/elements/1.1/",
            "dct": "http://purl.org/dc/terms/",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "om": "http://www.ontology-of-units-of-measure.org/resource/om-2/",
            "wv": "http://www.wurvoc.org/vocabularies/WV/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "bibo": "http://purl.org/ontology/bibo/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "ombibo": "http://www.wurvoc.org/bibliography/om-2/",
        },
        "dump_fp": data_dir / "om2_dump.json",
        "graph": None,
        "jsonld": None,
        "id_dict": None,
        "id_to_index": None,
        "type_dict": None,
        "type_index": None,
        "ids": None,
    },
    "sdf": {
        "url": "https://raw.githubusercontent.com/TheBIPM/SI_Digital_Framework/refs"
        "/heads/main/SI_Reference_Point/TTL/prefixes.ttl",
        "format": "turtle",
        "context": {
            "cgpm": "http://si-digital-framework.org/bodies/CGPM#",
            "dcterms": "http://purl.org/dc/terms/",
            "owl": "http://www.w3.org/2002/07/owl#",
            "prefixes": "http://si-digital-framework.org/SI/prefixes/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "si": "http://si-digital-framework.org/SI#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        },
        "dump_fp": data_dir / "sdf_prefixes_dump.json",
        "graph": None,
        "jsonld": None,
        "id_dict": None,
        "id_to_index": None,
        "type_dict": None,
        "type_index": None,
        "ids": None,
    },
    # todo: include other parts of the SI Digital Framework
    #  - quantities
    #  - units
    #  - constants
}

# List Prefixes with conversion factors
PREFIXES = {
    "Atto": 1e-18,
    "Centi": 1e-2,
    "Deca": 1e1,
    "Deci": 1e-1,
    "Deka": 1e1,
    "Exa": 1e18,
    "Exbi": 2**60,
    "Femto": 1e-15,
    "Gibi": 2**30,
    "Giga": 1e9,
    "Hecto": 1e2,
    "Kibi": 2**10,
    "Kilo": 1e3,
    "Mebi": 2**20,
    "Mega": 1e6,
    "Micro": 1e-6,
    "Milli": 1e-3,
    "Nano": 1e-9,
    "Pebi": 2**50,
    "Peta": 1e15,
    "Pico": 1e-12,
    "Quecto": 1e-30,
    "Quetta": 1e30,
    "Ronna": 1e27,
    "Ronto": 1e-27,
    "Tebi": 2**40,
    "Tera": 1e12,
    "Yobi": 2**80,
    "Yocto": 1e-24,
    "Yotta": 1e24,
    "Zebi": 2**70,
    "Zepto": 1e-21,
    "Zetta": 1e21,
}


def get_values[T](
    inp: dict[str, dict[str, T]] | list[dict[str, T]], key: str
) -> list[T]:
    """Returns values for a key on the first or second level. Stops if the key is not found."""
    if isinstance(inp, dict):
        if key in inp:
            return [inp[key]]
        # Only search one level deeper, do not recurse further
        for v in inp.values():
            if isinstance(v, dict) and key in v:
                return [v[key]]
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and key in item:
                        return [item[key]]
        # Key not found
        return []
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict) and key in item:
                return [item[key]]
        return []
    else:
        return []


def replace_keys[T](
    inp: dict[str, T], replacements: dict[str, str], keep_original_keys: bool = False
) -> dict[str, T]:
    """Replace keys in a dictionary according to a replacements mapping."""
    all_new = {replacements.get(k, k): v for k, v in inp.items()}
    if keep_original_keys:
        for k, v in inp.items():
            if k not in replacements:
                all_new[k] = v
    return all_new


def resolve_prefix(inp: str, ontology: str) -> str:
    """Resolve a prefixed IRI to a full IRI using the ontology context."""
    if ":" not in inp:
        raise ValueError(f"Input {inp} does not contain a prefix.")
    prefix, suffix = inp.split(":", 1)
    if prefix in ontologies[ontology]["context"]:
        return ontologies[ontology]["context"][prefix] + suffix
    raise ValueError(f"Prefix {prefix} not found in context of ontology {ontology}.")


def get_label_like_attr_from_dict(
    inp: dict[str, str | list[str | dict[str, str]] | dict[str, str]], attr_name: str
) -> list[dict[str, str]]:
    # todo: rework and check
    if attr_name not in inp:
        raise KeyError(f"Attribute {attr_name} not found in input dictionary.")
    attr = inp.get(attr_name)
    if not attr:
        return []
    if isinstance(attr, str):
        return [{"text": attr, "lang": "en"}]
    elif isinstance(attr, list):
        # todo: treat cases where not a list of dicts but a list of strings
        for ii, item in enumerate(attr):
            if isinstance(item, dict) and "@language" not in item:
                item["@language"] = "en"
            elif isinstance(item, str):
                attr[ii] = {"@value": item, "@language": "en"}
        return [
            replace_keys(item, {"@value": "text", "@language": "lang"})
            for item in attr
            if item["@language"] in ["en", "de"]
        ]
    elif isinstance(attr, dict):
        if "@language" not in attr:
            attr["@language"] = "en"
        return [replace_keys(attr, {"@value": "text", "@language": "lang"})]


def get_label_from_dict(
    inp: dict[str, str | list[dict[str, str]] | dict[str, str]],
) -> list[dict[str, str]]:
    return get_label_like_attr_from_dict(inp, "rdfs:label")


def get_desc_from_dict(inp: dict):
    if "qudt:plainTextDescription" in inp:
        return get_label_like_attr_from_dict(inp, "qudt:plainTextDescription")
    # The following might cause issues since this is a Latex string with special
    #  characters like \\{10^9\\} and similar:
    if "dcterms:description" in inp:
        return get_label_like_attr_from_dict(inp, "dcterms:description")
    return []


@log_call
def load_ontology_dump(ontology_acronym: str) -> dict:
    """Load QUDT dump from a URL into an rdflib Graph."""
    g = Graph()
    url = ontologies[ontology_acronym]["url"]
    g.parse(url, format=ontologies[ontology_acronym]["format"])
    _logger.info("Loaded %d triples from %s", len(g), url)
    jsonld_dict = json.loads(g.serialize(format="json-ld"))
    jsonld_compacted = jsonld_module.compact(
        jsonld_dict, ontologies[ontology_acronym]["context"]
    )
    return jsonld_compacted


def save_jsonld_to_file(jsonld_data: dict, filepath: Path):
    """Save JSON-LD data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(jsonld_data, f, indent=2)
    _logger.info("JSON-LD data saved to %s", filepath)


def load_jsonld_from_file(filepath: Path) -> dict:
    """Load JSON-LD data from a file."""
    with open(filepath, encoding="utf-8") as f:
        jsonld_data = json.load(f)
    _logger.info("JSON-LD data loaded from %s", filepath)
    return jsonld_data


def remove_prefixes(name: str, starts_with: bool = False) -> str:
    """Remove known prefixes from a unit name."""
    if starts_with:
        for prefix in PREFIXES:
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name
    result = name
    for prefix in PREFIXES:
        result = result.replace(prefix, "")
    return result


class UnitStringSplit(BaseModel):
    original: str
    without_prefixes: str
    # multiplication_factor: float
    removed_prefixes: list[str]
    removed_prefixes_left: list[str]
    removed_prefixes_right: list[str]


def remove_prefixes_calc_multiplication(string: str) -> UnitStringSplit:
    """Remove known prefixes from a unit name and calculate the multiplication factor."""
    # Split at PER
    parts = string.split("PER")
    removed_prefixes_left = []
    removed_prefixes_right = []
    # Remove prefixes from parts and list them
    for prefix in PREFIXES:
        if prefix in parts[0]:
            parts[0] = parts[0].replace(prefix, "")
            removed_prefixes_left.append(prefix)
        if len(parts) > 1 and prefix in parts[1]:
            parts[1] = parts[1].replace(prefix, "")
            removed_prefixes_right.append(prefix)
    result = "PER".join(parts)
    multiplication_factor = 1
    # todo: rework, this is to simple. We need to take the power into account
    #  - use regex to find prefixes and their powers
    for prefix in removed_prefixes_left:
        multiplication_factor *= PREFIXES[prefix]
    for prefix in removed_prefixes_right:
        multiplication_factor /= PREFIXES[prefix]
    return UnitStringSplit(
        original=string,
        without_prefixes=result,
        # multiplication_factor=multiplication_factor,
        removed_prefixes=removed_prefixes_left + removed_prefixes_right,
        removed_prefixes_left=removed_prefixes_left,
        removed_prefixes_right=removed_prefixes_right,
    )


@log_call
def resolve_factor_units(jsonld: dict, id_dict: dict):
    """Replaces hasFactorUnit references with the actual entries."""
    func_log.has_required_calls([prepare_all_ontologies, build_iri_dict])
    for item in jsonld.get("@graph", []):
        if "qudt:Unit" not in item.get("@type", []):
            continue
        if "qudt:hasFactorUnit" in item:
            factor_units = item["qudt:hasFactorUnit"]
            if not isinstance(factor_units, list):
                factor_units = [factor_units]
            replacement = []
            for factor_unit in factor_units:
                factor_unit_id = factor_unit.get("@id")
                if factor_unit_id and factor_unit_id not in id_dict:
                    _logger.warning(
                        "Factor unit IRI %s not found in iri_dict", factor_unit_id
                    )
                else:
                    replacement.append(id_dict.get(factor_unit_id))
            item["qudt:hasFactorUnit"] = replacement


@log_call
def build_iri_dict(jsonld: dict) -> tuple[dict, dict[str, int]]:
    """Creates a dictionary with IRIs as keys and items as values."""
    func_log.has_required_calls([load_ontology])
    iri_dict = {}
    iri_to_index = {}
    for ii, item in enumerate(jsonld.get("@graph", [])):
        iri = item.get("@id")
        if iri:
            if iri not in iri_dict:
                iri_dict[iri] = item
                iri_to_index[iri] = ii
            else:
                _logger.warning("Duplicate IRI found: %s", iri)
        else:
            _logger.warning("Item without @id found: %s", item)
    return iri_dict, iri_to_index


@log_call
def build_type_dict[T](jsonld: dict[str, list[T]]) -> dict[str, list[T]]:
    """Creates a dictionary with types as keys and lists of items as values."""
    func_log.has_required_calls([load_ontology])
    type_dict_: dict[str, list[T]] = {}
    # Iterate over the @graph and filter for @type
    for item in jsonld.get("@graph", []):
        if "@type" in item:
            if not isinstance(item["@type"], list):
                item["@type"] = [item["@type"]]
            _logger.debug(
                "Item: %s - Type: %s", item.get("@id", "No ID"), item["@type"]
            )
            for type_name in item["@type"]:
                type_dict_.setdefault(type_name, []).append(item)
        else:
            _logger.warning("Item: %s - No type found", item.get("@id", "No ID"))

    _logger.info("Types found in the QUDT dump:")
    for type_name, items in type_dict_.items():
        _logger.info(" - Type: %s - Count: %d", type_name, len(items))
    return type_dict_


@log_call
def build_type_index(jsonld: dict[str, list[dict[str, Any]]]) -> dict[str, list[int]]:
    """Creates a dictionary with types as keys and lists of indices as values. Can be
    used to access all entries of a certain type within the jsonld."""
    func_log.has_required_calls([load_ontology])
    type_index_: dict[str, list[int]] = {}
    for ii, item in enumerate(jsonld.get("@graph", [])):
        if "@type" in item:
            if isinstance(item["@type"], list):
                for type_name in item["@type"]:
                    if type_name not in type_index_:
                        type_index_[type_name] = []
                    type_index_[type_name].append(ii)
            else:
                if item["@type"] not in type_index_:
                    type_index_[item["@type"]] = []
                type_index_[item["@type"]].append(ii)

    return type_index_


def enrich_with_scaled_by(
    unit_id: str, base_unit_name: str, id_to_index: dict[str, int]
):
    """Enriches a base unit with information on a (prefixed) unit that scales it.

    Parameters
    ----------
    unit_id
        The unit to be listed as scaledBy
    base_unit_name
        The unit to be enriched
    id_to_index
        The dictionary serving as address mapping

    Returns
    -------

    """
    base_unit_dict_ = ontologies["qudt"]["jsonld"]["@graph"][
        id_to_index.get(base_unit_name)
    ]
    if "custom:scaledBy" not in base_unit_dict_:
        base_unit_dict_["custom:scaledBy"] = []
    if unit_id not in get_values(base_unit_dict_["custom:scaledBy"], "@id"):
        base_unit_dict_["custom:scaledBy"].append({"@id": unit_id})
        _logger.info(
            "Updated Non-prefixed base unit %s with scaledBy %s",
            base_unit_name,
            unit_id,
        )


def enrich_with_scaling_of(
    unit_id: str, base_unit_name: str, id_to_index: dict[str, int]
):
    """Enriches a (prefixed) unit with information on a base unit that it scales.

    Parameters
    ----------
    unit_id
        The unit to be enriched
    base_unit_name
        The unit to be listed as scalingOf
    id_to_index
        The dictionary serving as address mapping

    Returns
    -------

    """
    unit_dict_ = ontologies["qudt"]["jsonld"]["@graph"][id_to_index.get(unit_id)]
    if "qudt:scalingOf" not in unit_dict_:
        unit_dict_["qudt:scalingOf"] = []
    if base_unit_name not in get_values(unit_dict_["qudt:scalingOf"], "@id"):
        if isinstance(unit_dict_["qudt:scalingOf"], dict):
            unit_dict_["qudt:scalingOf"] = [unit_dict_["qudt:scalingOf"]]
            _logger.info("Converted scalingOf to list for unit %s", unit_id)
        unit_dict_["qudt:scalingOf"].append({"@id": base_unit_name})
        _logger.info(
            "Updated Prefixed unit %s with scalingOf %s",
            unit_id,
            base_unit_name,
        )


@log_call
def classify_and_enrich_qudt_units(
    type_dict: dict, id_dict: dict, id_to_index: dict
) -> dict:
    """Classifies units according to various criteria."""
    func_log.has_required_calls([resolve_factor_units])
    unit_type_dict: dict[str, list[dict]] = {
        "All": [],
        "SI unit": [],
        "Derived unit": [],
        "Composed unit": [],
        # Actually contains also composed units with prefix(es):
        "Unit without a prefix statement": [],  #
        "Non-prefixed unit": [],
        "Non-prefixed, non-composed unit": [],  #
        "Non-prefixed, composed unit": [],
        # Actually does not contain composed units with prefix(es):
        "Unit with a prefix statement": [],  #
        "Prefixed unit": [],
        "Prefixed, non-composed unit": [],
        "Prefixed, composed unit": [],
        "Prefixed, composed unit with no non-prefixed base": [],
        "Prefixed, composed unit with missing scalingOf": [],
        "Prefixed, composed unit with missing scalingOf but base unit available": [],
        "Prefixed, non-composed unit with no Non-prefixed base": [],
        "Prefixed unit with missing scalingOf": [],
    }

    def is_si_unit(unit_dict_: dict) -> bool:
        return unit_dict_.get("qudt:conversionMultiplier", {}).get("@value") == "1.0"

    def is_derived_unit(unit_dict_: dict) -> bool:
        return "qudt:DerivedUnit" in unit_dict_.get("@type", [])

    def has_prefix_statement(unit_dict_: dict) -> bool:
        return "qudt:prefix" in unit_dict_

    def has_factor_unit_statement(unit_dict_: dict) -> bool:
        return "qudt:hasFactorUnit" in unit_dict_

    def has_scaling_of_statement(unit_dict_: dict) -> bool:
        return "qudt:scalingOf" in unit_dict_

    def factor_units_are_prefixed(unit_dict_: dict) -> bool:
        if not has_factor_unit_statement(unit_dict_):
            return False
        factor_units_ = unit_dict_["qudt:hasFactorUnit"]
        if not isinstance(factor_units_, list):  # Single entry
            factor_units_ = [factor_units_]
        for factor_unit_ in factor_units_:
            factor_unit_id_ = factor_unit_.get(  # unresolved address
                "qudt:hasUnit"
            ).get("@id")
            factor_unit_dict_ = id_dict.get(factor_unit_id_, {})
            if "qudt:prefix" in factor_unit_dict_:
                return True
        return False

    def treat_prefixed_non_composed_unit(unit_dict_: dict):
        unit_type_dict["Prefixed, non-composed unit"].append(unit_dict)
        _logger.info(" - (also listed as Prefixed, non-composed unit)")
        # See if the unit can be reduced to a Non-prefixed base unit
        base_unit_name_ = remove_prefixes(unit_dict_["@id"])
        base_unit_found_ = False
        for candidate_ in type_dict.get("qudt:Unit", []):
            if base_unit_name_ == candidate_.get("@id", ""):
                base_unit_found_ = True
                _logger.info(" - Non-prefixed base unit found: %s", base_unit_name_)
                break
        if base_unit_found_ and "qudt:scalingOf" not in unit_dict_:
            unit_type_dict["Prefixed unit with missing scalingOf"].append(unit_dict_)
            _logger.info(" - (also listed as Prefixed unit with missing scalingOf)")
        # Prefixed unit with missing scalingOf
        if not base_unit_found_ and "qudt:scalingOf" not in unit_dict_:
            # This case should not happen, as all prefixed units should be reducible to
            # a Non-prefixed base unit, and if not then they should have a scalingOf
            # property. There is one case:
            # https://qudt.org/vocab/unit/FT3
            # https://qudt.org/vocab/unit/KiloCubicFT
            unit_type_dict[
                "Prefixed, non-composed unit with no Non-prefixed base"
            ].append(unit_dict_)
            _logger.info(
                " - (also listed as Prefixed, non-composed Unit with no Non-prefixed "
                "base)"
            )
        # Enrich the jsonld with scalingOf if missing and scaledBy
        if base_unit_found_:
            if "qudt:scalingOf" not in unit_dict_:
                # Enrich the jsonld with scalingOf if missing
                unit_dict_["qudt:scalingOf"] = {"@id": base_unit_name_}
            # Enrich the Non-prefixed base unit with scaledBy
            enrich_with_scaled_by(unit_dict_["@id"], base_unit_name_, id_to_index)

    def treat_prefixed_composed_unit(unit_dict_: dict):
        unit_type_dict["Prefixed, composed unit"].append(unit_dict_)
        _logger.info(
            " - (also listed as Prefixed, composed unit with missing scalingOf)"
        )
        # Check if the unit can be reduced to a Non-prefixed base unit
        base_unit_name_ = remove_prefixes(unit_dict["@id"])
        base_unit_found_ = False
        for candidate_ in type_dict.get("qudt:Unit", []):
            if base_unit_name_ == candidate_.get("@id", ""):
                base_unit_found_ = True
                _logger.info(" - Non-prefixed base unit found: %s", base_unit_name_)
                break
        if "qudt:scalingOf" not in unit_dict:
            unit_type_dict["Prefixed, composed unit with missing scalingOf"].append(
                unit_dict
            )
            _logger.info(
                " - (also listed as Prefixed, composed unit with missing scalingOf)"
            )
        if base_unit_found_:
            if "qudt:scalingOf" not in unit_dict:
                unit_type_dict[
                    "Prefixed, composed unit with missing scalingOf but base unit "
                    "available"
                ].append(unit_dict)
                _logger.info(
                    " - (also listed as Prefixed, composed unit with missing "
                    "scalingOf but base unit available)"
                )
                # Enrich the jsonld with scalingOf if missing
                unit_dict["qudt:scalingOf"] = {"@id": base_unit_name_}
                # todo: scalingOf is not missing anymore
            # Enrich the Non-prefixed base unit with scaledBy
            enrich_with_scaled_by(unit_dict_["@id"], base_unit_name_, id_to_index)
        else:  # if not base_unit_found:
            unit_type_dict["Prefixed, composed unit with no non-prefixed base"].append(
                unit_dict
            )
            _logger.info(
                " - (also listed as Prefixed, composed Unit with no Non-prefixed base)"
            )

    class FilterAction(TypedDict):
        function: Callable[[dict], bool]
        action: Callable[[dict], None]

    filter_action: dict[str, FilterAction] = {
        "'conversionMultiplier' in QUDT is one": {
            "function": is_si_unit,
            "action": lambda u: unit_type_dict["SI unit"].append(u),
        },
        "Listed as 'DerivedUnit' in QUDT": {
            "function": is_derived_unit,
            "action": lambda u: unit_type_dict["Derived unit"].append(u),
        },
        "QUDT states 'prefix'": {
            "function": has_prefix_statement,
            "action": lambda u: unit_type_dict["Unit with a prefix statement"].append(
                u
            ),
        },
        "QUDT does not state 'prefix'": {
            "function": lambda u: not has_prefix_statement(u),
            "action": lambda u: unit_type_dict[
                "Unit without a prefix statement"
            ].append(u),
        },
        "QUDT states 'hasFactorUnit'": {
            "function": has_factor_unit_statement,
            "action": lambda u: unit_type_dict["Composed unit"].append(u),
        },
        "QUDT states 'scalingOf'": {
            "function": has_scaling_of_statement,
            "action": lambda u: None,  # Not used directly
        },
        "The factor units are prefixed -> prefixed composed unit": {
            "function": lambda u: has_factor_unit_statement(u)
            and factor_units_are_prefixed(u),
            "action": lambda u: unit_type_dict["Prefixed, composed unit"].append(u),
        },
        "The factor units are not prefixed": {
            "function": lambda u: has_factor_unit_statement(u)
            and not factor_units_are_prefixed(u),
            "action": lambda u: unit_type_dict["Non-prefixed, composed unit"].append(u),
        },
        "QUDT does neither state 'prefix' nor 'hasFactorUnit' --> Non-prefixed "
        "non-composed unit": {
            "function": lambda u: not has_prefix_statement(u)
            and not has_factor_unit_statement(u),
            "action": lambda u: unit_type_dict[
                "Non-prefixed, non-composed unit"
            ].append(u),
        },
        "QUDT states 'prefix' but not 'hasFactorUnit' -> prefixed non-composed unit": {
            "function": lambda u: has_prefix_statement(u)
            and not has_factor_unit_statement(u),
            "action": treat_prefixed_non_composed_unit,
        },
        "QUDT states 'hasFactorUnit' and at least one factor unit is prefixed": {
            "function": lambda u: has_factor_unit_statement(u)
            and factor_units_are_prefixed(u),
            "action": treat_prefixed_composed_unit,
        },
    }

    # Iterate over all "qudt:Unit" items and print their @id and label
    for unit_id, unit_index in id_to_index.items():
        if "qudt:Unit" not in id_dict[unit_id].get("@type", []):
            continue
        unit_dict = ontologies["qudt"]["jsonld"]["@graph"][unit_index]
        unit_type_dict["All"].append(unit_dict)
        label = unit_dict.get("rdfs:label", ["No label"])
        if isinstance(label, list):
            label = label[0]
        _logger.debug("Unit IRI: %s - Label: %s", unit_id, label)

        for fa in filter_action.values():
            if fa["function"](unit_dict):
                fa["action"](unit_dict)

    # Units
    # - Non-prefixed unit
    #   - Units without property hasFactorUnit
    #   - Units with property hasFactorUnit but no prefix in any factor unit
    unit_type_dict["Non-prefixed unit"] = (
        unit_type_dict["Non-prefixed, non-composed unit"]
        + unit_type_dict["Non-prefixed, composed unit"]
    )
    # - Prefixed Unit (scaling of a Non-prefixed unit)
    #   - Units with property prefix
    #     - prefix
    #     - scalingOf -> Non-prefixed unit ==> Check if scalingOf is missing
    #   - Units with property hasFactorUnit and at least one prefix in a factor unit
    #     - strip all prefixes from the unit name / string -> Non-prefixed unit (listed?
    #       yes all)
    unit_type_dict["Prefixed unit"] = (
        unit_type_dict["Prefixed, non-composed unit"]
        + unit_type_dict["Prefixed, composed unit"]
    )
    # - Composed unit (combination of multiple prefixed and Non-prefixed units)
    #   - Units with property hasFactorUnit
    # unit_type_dict["Composed unit"]

    return unit_type_dict


@log_call
def load_qudt_missing_units() -> list[dict[str, Any]]:
    with open(missing_units_fp, encoding="utf-8") as f:
        missing_units_list = json.load(f)

    for mu in missing_units_list:
        scaled_by_list = get_values(mu["custom:scaledBy"], "@id")
        if not scaled_by_list:
            _logger.warning("Missing unit without scaledBy found: %s", mu)
            continue
        for id_ in scaled_by_list:
            enrich_with_scaling_of(id_, mu["@id"], ontologies["qudt"]["id_to_index"])
            _logger.info("Enriched missing unit %s with scaledBy %s", mu["@id"], id_)

    return missing_units_list


@log_call
def process_prefixed_composed_units_with_ai(
    unit_type_dict: dict, id_to_index: dict[str, int]
) -> list[dict[str, Any]]:
    """
    Basic idea:
    - Get all prefixed composed units without a Non-prefixed base unit
    - For each unit, strip all known prefixes from the unit name
    Now construct an entry within the QUDT ontology that serves as non-prefixed
     base unit, with the following keys:
    - @id: strip the prefixes from the unit name / @id
      - while doing so, track which prefixes were removed and in which order and if
         they were infront or behind a PER string
    - @type: ["qudt:Unit"]
    - dcterms:description: "AI-generated: modify description of the prefixed
       composed unit based on the modifications done by removing the prefixes
    qdt:plainTextDescription: "AI-generated: modify description of the prefixed
       composed unit based on the modifications done by removing the prefixes
    - qudt:conversionMultiplier: work with the conversionMultiplier of the
       prefixed composed unit, and the conversionMultipliers of the prefixes that
       were removed
    - qudt:conversionMultiplier: work with the conversionMultiplierSN of the
       prefixed composed unit, and the conversionMultiplierSN of the prefixes that
       were removed
    - qudt:hasDimensionVector: copy from the prefixed composed unit
    - qudt:hasFactorUnit: take the ones that are listed in the prefixed composed unit,
       but for each factor unit, strip the prefixes that were removed from the
       prefixed composed unit from the factor unit name as well, and look up the
       corresponding Non-prefixed unit in the QUDT dump to make sure the unit exists
       - if not, warn and raise error
    - qudt:hasQuantityKind: copy from the prefixed composed unit
    - qudt:symbol: genereate from the qudt:symbol of the prefixed composed unit by
       removing the prefixes from the symbol as well
    - qudt:ucumCode: should be derivable from the factor units and might be
       a good base for qudt:symbol
    - rdfs:isDefinedBy: copy from the prefixed composed unit
    - rdfs:label: copy from the prefixed composed unit, but remove the prefixes from the
       label as well
    - custom:scaledBy: add the @id of the prefixed composed unit in a dict with key @id
       to the list of custom:scaledBy entries of the Non-prefixed base unit
    - within the dict of the prefixed composed unit, add the key
       qudt:scalingOf with the value being a dict with key @id and value the @id of the
       Non-prefixed base unit
    """
    func_log.has_required_calls([classify_and_enrich_qudt_units])

    # Define the LLM and prompt components
    class PartiallyKnownComposedUnit(BaseModel):
        """A composed unit is a unit that is derived from two or more other units,
        typically through multiplication or division. Composed units are used to
        express complex physical quantities that cannot be represented by a single
        base unit. For example, speed is a composed unit that combines the base units
        of distance (meters) and time (seconds) to express how far an object travels
        in a given amount of time (meters per second, m/s). The composing units
        are states as factor units within QUDT."""

        at_id: str = Field(alias="@id")
        at_type: list[str] = Field(alias="@type")
        qudt_hasDimensionVector: dict[str, str] | None = Field(
            default=None,
            alias="qudt:hasDimensionVector",
            description="The Dimension Vector represents the dimensional formula of a unit in terms of the seven base quantities of the International System of Units (SI).",
        )
        qudt_hasQuantityKind: dict[str, str] | list[dict[str, str]] | None = Field(
            default={"@id": "quantitykind:Unknown"},
            alias="qudt:hasQuantityKind",
            description="The Quantity Kind represents the physical quantity that a unit measures, such as length, mass, time, electric current, temperature, amount of substance, luminous intensity, and many others.",
        )
        rdfs_isDefinedBy: dict[str, str] | None = Field(
            default=None,
            alias="rdfs:isDefinedBy",
            description="The isDefinedBy property is used to indicate a resource that provides a definition or description of the subject resource.",
        )
        custom_scaledBy: list[dict[str, str]] | None = Field(
            default=None,
            alias="custom:scaledBy",
            description="List of prefixed units that scale this Non-prefixed base unit.",
        )

    class ComposedUnit(PartiallyKnownComposedUnit):
        dcterms_description: dict[str, str] | str | None = Field(
            default=None,
            alias="dcterms:description",
            description="Description may include but is not limited to: an abstract, a table of contents, a graphical representation, or a free-text account of the resource.",
        )
        qudt_plainTextDescription: dict[str, str] | str | None = Field(
            default=None,
            alias="qudt:plainTextDescription",
            description="A plain text description is used to provide a description with only simple ASCII characters for cases where LaTeX , HTML or other markup would not be appropriate.",
        )
        qudt_conversionMultiplier: dict[str, str | float] | None = Field(
            default={"@type": "xsd:decimal", "@value": "0.0"},
            alias="qudt:conversionMultiplier",
            description="conversion multiplier",
        )
        qudt_conversionMultiplierSN: dict[str, str | float] | None = Field(
            default={"@type": "xsd:double", "@value": 0.0},
            alias="qudt:conversionMultiplierSN",
            description="conversion multiplier in scientific notation",
        )
        qudt_ucumCode: dict[str, str] | list[dict[str, str]] | None = Field(
            default=None,
            alias="qudt:ucumCode",
            description="The UCUM code is a string that represents the unit in the Unified Code for Units of Measure (UCUM) system. ucumCode associates a QUDT unit with its UCUM code (case-sensitive).",
        )
        qudt_symbol: dict[str, str] | str | None = Field(
            default=None,
            alias="qudt:symbol",
            description="A symbol is a short string of characters that represents the unit. Symbols are often used in scientific and technical contexts to denote units in equations, formulas, and measurements.",
        )
        rdfs_label: dict[str, str] | list[dict[str, str]] | str | None = Field(
            default=None,
            alias="rdfs:label",
            description="A label is a human-readable name or title for a resource. Labels are often used to provide a concise and meaningful representation of the resource in user interfaces, lists, and other contexts.",
        )
        qudt_hasFactorUnit: list[dict[str, str | dict[str, str | int]]] = Field(
            default=...,
            alias="qudt:hasFactorUnit",
            description="The factor units that compose this composed unit. Each "
            "factor unit is represented as a dicts with two keys. Example: "
            "{'qudt:exponent': {'@type': 'xsd:integer', '@value': 1}, 'qudt:hasUnit': {'@id': 'unit:GM'}}",
        )

    class PartiallyKnowWithAdditionalInfo(PartiallyKnownComposedUnit):
        removed_prefixes: list[str] = Field(
            default=...,
            description="List of prefixes that were removed from the original unit string to create the Non-prefixed base unit.",
        )
        removed_prefixes_left: list[str] = Field(
            default=...,
            description="List of prefixes that were removed from the left side of the original unit string (before 'PER') to create the Non-prefixed base unit.",
        )
        removed_prefixes_right: list[str] = Field(
            default=...,
            description="List of prefixes that were removed from the right side of the original unit string (after 'PER') to create the Non-prefixed base unit.",
        )
        without_prefixes: str = Field(
            default=...,
            description="The unit string after removing all known prefixes.",
        )
        guessed_factor_unit_ids: list[str] = Field(
            default=...,
            description="List of guessed Non-prefixed factor unit IDs based on the removed prefixes and the original factor units.",
        )

    class BaseComposedUnit(ComposedUnit):
        """A base composed unit is a composed unit that has no factor units with
        a unit prefix. In other words, all factor units of a base composed unit
        are Non-prefixed units.

        If the prefixed composed unit had a dcterms:description or
        qudt:plainTextDescription, modify it to reflect the removal of the prefixes
        from the original prefixed composed unit. At the beginning of the description,
        add "AI-generated: " to indicate that the description was modified by an AI.
        Don't describe the prefixes that were removed, but rather focus on the
        resulting unit and its properties. Stay as close as possible to the original
        description, but adapt it to the new unit.

        The qudt:symbol and qudt:ucumCode should be generated by removing the
        prefixes from the original prefixed composed unit's qudt:symbol and
        qudt:ucumCode. If the resulting symbol or ucumCode is not valid or does not
        make sense, try to improve it based on the factor units of the new
        Non-prefixed base unit. The qudt:symbol should be as short as possible,
        while still being unique and recognizable. The qudt:ucumCode should follow
        the UCUM rules as closely as possible.
        """

        custom_scaledBy: list[dict[str, str]] = Field(
            ...,
            alias="custom:scaledBy",
            description="List of prefixed composed units that scale this "
            "Non-prefixed base unit. Formatted as list of dicts with key '@id'.",
        )
        multiplication_factor: float = Field(
            ...,
            description="The multiplication factor that results from the removal of "
            "the prefixes from the original unit string and "
            "corresponds to the conversion of prefixed composed unit "
            "to base composed unit.",
        )

    class Input(BaseModel):
        """Provides the partially known Composed Unit with some additional info.
        A BaseComposedUnit that has no prefixes is to be derived fro this data
        and by processing a prefixed composed unit."""

        known_data: PartiallyKnowWithAdditionalInfo = Field(
            ...,
            description="The known data of the prefixed composed unit and the "
            "additional information about the removed prefixes.",
        )
        prefixed_composed: ComposedUnit = Field(
            ...,
            description="The full data of the prefixed composed unit as is in the "
            "QUDT dump.",
        )
        prefixes_and_multiplication_factors: dict[str, float] = Field(
            default=PREFIXES,
            description="The known prefixes and their corresponding multiplication "
            "factors.",
        )

    # check_if_id_exists = Tool(
    #     name="check_if_id_exists",
    #     func=lambda unit_id: unit_id in id_to_index,
    #     description=(
    #         "Use this function to check if a guessed (Non-prefixed factor) unit ID "
    #         "exists in the QUDT dump. The input is the unit ID as a string, e.g. "
    #         "'qudt:GM'. The output is a boolean indicating whether the ID exists."
    #     ),
    # )

    # Components to be used in the chain

    output_parser = PydanticOutputParser(pydantic_object=BaseComposedUnit)

    # format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=(
            "You are an expert assistant for structured data tasks.\n"
            "Answer the user query.\n{format_instructions}\n{query}\n"
        ),
        input_variables=["query"],
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    # Pickle file to store the results - avoid re-running the LLM for already
    #  processed units
    json_fp = (
        Path(__file__).parents[2] / "data" / "new_non_prefixed_base_composed_units.json"
    )
    # List to collect new Non-prefixed composed base units
    dict_list: list[dict[str, Any]] = []
    already_processed = set()

    # Load previously processed units from json file if it exists
    if json_fp.exists():
        _logger.info(f"Loading previously processed units from {json_fp}")
        with open(json_fp, "rb") as f:
            dict_list = json.load(f)
        already_processed = {d["@id"] for d in dict_list}
    else:
        _logger.info(f"No json file found at {json_fp}, starting fresh.")

    full_runs = 0
    # Process all prefixed composed units without a Non-prefixed base unit
    for pcu_dict in unit_type_dict["Prefixed, composed unit with no non-prefixed base"]:
        pcu_id = pcu_dict["@id"]
        _logger.info(f"Processing unit: {pcu_id}")
        pcu_id_wo_onto_prefix = pcu_id.split(":")[-1]
        pcu_index = ontologies["qudt"]["id_to_index"][pcu_id]
        # Get the enriched version from the jsonld
        pcu_dict = ontologies["qudt"]["jsonld"]["@graph"][pcu_index]

        # 1. Remove all known prefixes from the unit name
        split_result = remove_prefixes_calc_multiplication(pcu_id_wo_onto_prefix)
        # base_unit_id_wo_onto_prefix = split_result.without_prefixes
        base_unit_id = f"unit:{split_result.without_prefixes}"

        if base_unit_id in already_processed:
            _logger.info(
                f"Skipping already processed unit: {pcu_id}, from which "
                f"{base_unit_id} has beed created and loaded from json "
                f"file."
            )
            continue

        # Separately handled cases
        # Prefixed composed unit with no Non-prefixed base: guessed base unit does not
        #  exist but another scalingOf can be defined
        irregular = {
            # "unit:KiloCAL-PER-CentiM2": {  # Should now be handled with added unit:CAL
            #     "guessed base unit": "unit:CAL-PER-M2",
            #     "non existing non-prefixed unit": ["unit:CAL"],
            #     "correct scalingOf": "unit:J-PER-M2",
            # },
            # "unit:KiloCAL-PER-CentiM2-SEC": {  # Should now be handled with added unit:CAL
            #     "guessed base unit": "unit:CAL-PER-M2-SEC",
            #     "non existing non-prefixed unit": ["unit:CAL"],
            #     "correct scalingOf": "unit:J-PER-M2-SEC",  # but also missing
            # },
        }
        if pcu_id in irregular:
            _logger.warning(
                f"Unit {pcu_id} is known to be irregular, skipping creation of "
                f"Non-prefixed base unit {base_unit_id} and using predefined "
                f"scalingOf {irregular[pcu_id]['correct scalingOf']}"
            )
            continue

        # 2. Check if the Non-prefixed base unit already exists
        base_unit_exists = False
        for base_unit_dict in ontologies["qudt"]["jsonld"]["@graph"]:
            if base_unit_dict.get("@id") == base_unit_id:
                base_unit_exists = True
                break
        if base_unit_exists:
            _logger.info(
                f"Non-prefixed base unit {base_unit_id} already exists, skipping creation"
            )
            # But still add the scalingOf to the prefixed composed unit
            pcu_dict["qudt:scalingOf"] = {"@id": base_unit_id}
            # And add the scaledBy to the Non-prefixed base unit
            enrich_with_scaled_by(pcu_id, base_unit_id, id_to_index)
            continue  # Skip the rest, if already present

        _logger.info(
            f"Creating non-prefixed base unit {base_unit_id} from {pcu_id} "
            f"with the help of AI"
        )

        # 3.1 Provide data for the new Non-prefixed base unit
        base_unit_dict = {
            "@id": base_unit_id,
            "@type": ["qudt:Unit"],
            "qudt:hasDimensionVector": pcu_dict.get("qudt:hasDimensionVector"),
            "qudt:hasQuantityKind": pcu_dict.get(
                "qudt:hasQuantityKind", {"@id": "quantitykind:Unknown"}
            ),
            "rdfs:isDefinedBy": pcu_dict.get("rdfs:isDefinedBy"),
            "custom:scaledBy": [{"@id": pcu_id}],
        }
        applicable_systems = get_values(pcu_dict, "qudt:applicableSystem")
        if applicable_systems:
            base_unit_dict["qudt:applicableSystem"] = applicable_systems

        # 3.2 Use LLM to provide missing attributes
        factor_unit_ids = get_values(
            get_values(pcu_dict.get("qudt:hasFactorUnit", []), "qudt:hasUnit"), "@id"
        )

        guessed_factor_unit_ids = []
        for fu_id in factor_unit_ids:
            new_fu_id = remove_prefixes(fu_id)
            if new_fu_id not in id_to_index:
                _logger.error(
                    f"Guessed Non-prefixed factor unit ID {new_fu_id} does not exist in QUDT dump"
                )
                raise ValueError(
                    f"Guessed Non-prefixed factor unit ID {new_fu_id} does not exist in QUDT dump"
                )
            guessed_factor_unit_ids.append(new_fu_id)

        with_extra_info = {
            **base_unit_dict,
            **{
                "removed_prefixes": split_result.removed_prefixes,
                "removed_prefixes_left": split_result.removed_prefixes_left,
                "removed_prefixes_right": split_result.removed_prefixes_right,
                "without_prefixes": split_result.without_prefixes,
                "guessed_factor_unit_ids": guessed_factor_unit_ids,
            },
        }

        input_ = Input(
            known_data=PartiallyKnowWithAdditionalInfo(**with_extra_info),
            prefixed_composed=ComposedUnit(**pcu_dict),
        )

        input_message = (
            f"Create a Non-prefixed base composed unit from the following data:\n"
            f"{input_.model_dump_json(indent=2)}\n"
            f"The new Non-prefixed base composed unit should reflect the removal of "
            f"the prefixes from the original prefixed composed unit. This includes "
            f"mentioning the removed prefixes and adjusting the descriptions "
            f"accordingly. The multiplication factor should be calculated based on "
            f"the removed prefixes. Use logic to determine the power of the prefixes. "
            f"e.g. if PER-CentiM3 is part of the unit string its removal lead to a "
            f"multiplication_factor of 10**6.\n"
            f"The conversionMultiplier and conversionMultiplierSN have to be adjusted "
            f"based on the multiplication factor. If the prefixed composed unit had "
            f"a conversionMultiplier with value 0, the non-prefixed base composed unit "
            f"will be the same.\n"
            f"The rdf label and the symbl both has to reflect the removal of the "
            f"prefixes.\n"
            f"The factor units have to be adjusted as well by removing the prefixes "
            f"from the @id of the factor units. Make sure that the "
            f"Non-prefixed factor units exist in the QUDT dump, otherwise warn and "
            f"raise an error.\n"
            f"Provide the complete data for the new Non-prefixed base composed unit "
            f"and make sure that at_id, dcterms_description or "
            f"qudt_plainTextDescription, qudt_ucumCode, qudt_symbol, rdfs_label and "
            f"qudt_hasFactorunit are consistent."
        )

        def init_llm():
            return AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                model=LLM_MODEL,
                temperature=0,
                max_retries=3,
                timeout=300,
            )

        # Initialize the LLM (freshly every N executions)
        if full_runs % 4 == 0 or full_runs == 0:
            msg = (
                "Initializing the LLM"
                if full_runs == 0
                else "Re-initializing the LLM to avoid potential memory issues."
            )
            _logger.info(msg)
            llm = init_llm()
        # 3.3 Invoke the chain to get the new Non-prefixed base unit
        try:
            # Defining the chain to be used in the processing loop
            chain = prompt | llm | output_parser
            # Invoke the chain
            new_composed_base_unit = chain.invoke({"query": input_message})
        except Timeout:
            # Retry once after a short delay
            _logger.warning(
                f"Timeout occurred for unit {pcu_id}, retrying once with "
                f"freshly re-initialized LLM after short delay."
            )
            time.sleep(10)
            # Re-initialize the LLM
            llm = init_llm()
            # Defining the chain to be used in the processing loop
            chain = prompt | llm | output_parser
            # Invoke the chain with freshly initialized LLM
            new_composed_base_unit = chain.invoke({"query": input_message})

        # After we made sure that the new Non-prefixed base unit will exist, state the
        #  scalingOf in the prefixed composed unit
        if "custom:scaledBy" not in new_composed_base_unit.model_dump(by_alias=True):
            new_composed_base_unit.custom_scaledBy = []
        if {"@id": pcu_id} not in new_composed_base_unit.custom_scaledBy:
            new_composed_base_unit.custom_scaledBy.append({"@id": pcu_id})
        if "qudt:scalingOf" not in pcu_dict:
            pcu_dict["qudt:scalingOf"] = {"@id": base_unit_id}

        dict_list.append(new_composed_base_unit.model_dump(by_alias=True))
        # After successfully creating a new list element, dump the current state of
        #  the dict_list
        with open(json_fp, "w", encoding="utf-8") as f:
            json.dump(dict_list, f, indent=2, ensure_ascii=False)

        full_runs += 1

    # Save the list at the end as well
    with open(json_fp, "w", encoding="utf-8") as f:
        json.dump(dict_list, f, indent=2, ensure_ascii=False)

    return dict_list


@log_call
def classify_quantity_kinds(type_dict: dict) -> dict:  # , iri_dict: dict):
    """Classify quantity kinds into fundamental and non-fundamental."""
    func_log.has_required_calls([build_type_dict])
    quantity_kinds_: list[dict[str, Any]] = type_dict["qudt:QuantityKind"]
    quantity_kind_dict_ = {
        # Is pointed at with skos:broader but does not possess that property
        "Fundamental": [],
        # Possesses property skos:broader
        "Non-fundamental": [],
    }
    for qk in quantity_kinds_:
        if "skos:broader" in qk:
            quantity_kind_dict_["Non-fundamental"].append(qk)
        else:
            quantity_kind_dict_["Fundamental"].append(qk)

    _logger.info(
        "%d fundamental quantity kinds found", len(quantity_kind_dict_["Fundamental"])
    )
    _logger.info(
        "%d non-fundamental quantity kinds found",
        len(quantity_kind_dict_["Non-fundamental"]),
    )

    return quantity_kind_dict_


@log_call
def report_on_unit_types(type_dict: dict, unit_type_dict: dict):
    """Creates an overview of unit types by printing the categories and there overlaps.

    Parameters
    ----------
    type_dict:
        The type dictionary as created by `build_type_dict`.
    unit_type_dict:
        The unit type dictionary as created by `classify_and_enrich_qudt_units`.
    """
    func_log.has_required_calls([classify_and_enrich_qudt_units])
    _logger.info("Unit types found in the QUDT dump:")
    for type_name, items in unit_type_dict.items():
        # Use unique set of items
        items = list({item["@id"]: item for item in items}.values())
        unit_type_dict[type_name] = items
        _logger.info(" - Type: %s - Count: %d", type_name, len(items))
    _logger.info("Total units found: %d", len(type_dict.get("qudt:Unit", [])))
    _logger.info(
        "Untreated units: %d",
        len(type_dict.get("qudt:Unit", []))
        - sum(len(v) for v in unit_type_dict.values()),
    )
    # Intersection analysis
    unit_sets = {k: {u.get("@id") for u in v} for k, v in unit_type_dict.items()}
    # Calculate the intersection of the sets pairwise
    intersection_count = 0
    for i, (k1, v1) in enumerate(unit_sets.items()):
        for k2, v2 in list(unit_sets.items())[i + 1 :]:
            intersection = v1.intersection(v2)
            if intersection:
                intersection_count += len(intersection)
                _logger.info("Units in both %s and %s: %d", k1, k2, len(intersection))
    _logger.info("Total units listed in multiple categories: %d", intersection_count)


@log_call
def enrich_prefixes_with_ontology_matches(type_index: dict[str, list[int]]) -> None:
    """Enrich the prefixes in the QUDT with exact ontology matches."""
    func_log.has_required_calls([build_type_index])
    for ii in type_index["qudt:Prefix"]:
        prefix_dict = ontologies["qudt"]["jsonld"]["@graph"][ii]
        prefix_id = prefix_dict.get("@id", "")
        if not prefix_id:
            _logger.warning("Prefix at index %d has no @id: %s", ii, prefix_dict)
            continue
        prefix_dict["owl:sameAs"] = [{"@id": resolve_prefix(prefix_id, "qudt")}]
        om_id_guess = f"om:{prefix_id.split(':')[-1].lower()}"
        # OM2 - Ontology of Units of Measure
        if om_id_guess in ontologies["om2"]["id_dict"]:
            prefix_dict["owl:sameAs"].append(
                {"@id": resolve_prefix(om_id_guess, "om2")}
            )
            _logger.info(
                "Found exact ontology match for prefix %s: %s",
                prefix_id,
                om_id_guess,
            )
        # SI Digital Framework
        sdf_id_guess = f"prefixes:{prefix_id.split(':')[-1].lower()}"
        if sdf_id_guess in ontologies["sdf"]["id_dict"]:
            prefix_dict["owl:sameAs"].append(
                {"@id": resolve_prefix(sdf_id_guess, "sdf")}
            )
            _logger.info(
                "Found exact ontology match for prefix %s: %s",
                prefix_id,
                sdf_id_guess,
            )
        # Wikidata: no prefixes in Wikidata as of 2025-08-27


@log_call
def create_unit_prefix_entities() -> list[model.UnitPrefix]:
    """Create UnitPrefix entities from the QUDT prefixes."""
    func_log.has_required_calls([enrich_prefixes_with_ontology_matches])
    # Handling of prefixes
    prefixes = [
        ontologies["qudt"]["jsonld"]["@graph"][ii]
        for ii in ontologies["qudt"]["type_index"]["qudt:Prefix"]
    ]
    unit_prefix_entities_ = []
    for prefix_dict in prefixes:
        data = replace_keys(prefix_dict, {"qudt:symbol": "symbol"})
        uuid = str(
            uuid_module.uuid5(
                namespace=uuid_module.NAMESPACE_URL,
                name=data["@id"],  # todo: check correct field name
            )
        )
        label = get_label_from_dict(data)
        exact_ontology_match = [
            sameas_dict["@id"] for sameas_dict in data.get("owl:sameAs", [])
        ]
        if "qudt:siExactMatch" in data:
            exact_ontology_match.extend(
                [
                    resolve_prefix(item, "qudt")
                    for item in get_values(data["qudt:siExactMatch"], "@id")
                ]
            )
        if "qudt:dbpediaMatch" in data:
            exact_ontology_match.extend(get_values(data["qudt:dbpediaMatch"], "@value"))
        if "qudt:exactMatch" in data:
            exact_ontology_match.extend(
                [
                    resolve_prefix(item, "qudt")
                    for item in get_values(data["qudt:exactMatch"], "@id")
                ]
            )
        data.update(
            {
                "uuid": uuid,
                "osw_id": f"Item:OSW{uuid.replace('-', '')}",
                "name": label[0]["text"] if label else data["@id"].split(":")[-1],
                "exact_ontology_match": list(set(exact_ontology_match)),
                "label": label,
                "description": get_desc_from_dict(data),
                "type": ["Category:OSW99e0f46a40ca4129a420b4bb89c4cc45"],  # Unit prefix
                "factor": data["qudt:prefixMultiplier"]["@value"],
            }
        )
        # todo: use data["qudt:informativeReference"]
        # todo: use data["qudt:ucumCode"]
        unit_prefix_entities_.append(model.UnitPrefix(**data))
    return unit_prefix_entities_


@log_call
def create_quantity_unit_entities(unit_type_dict: dict) -> list[model.QuantityUnit]:
    func_log.has_required_calls([classify_and_enrich_qudt_units])

    # Processing non-composed units with/without prefix
    non_prefixed_unit_entities = []
    for npu_dict in unit_type_dict["Non-prefixed, non-composed unit"]:
        npu_id = npu_dict["@id"]
        npu_index = ontologies["qudt"]["id_to_index"][npu_id]
        # Get the enriched version of the non-prefixed unit from the jsonld
        npu_dict = ontologies["qudt"]["jsonld"]["@graph"][npu_index]
        npu_uuid = str(
            uuid_module.uuid5(
                namespace=uuid_module.NAMESPACE_URL,
                name=resolve_prefix(npu_id, "qudt"),
            )
        )
        npu_osw_id = f"Item:OSW{npu_uuid.replace('-', '')}"
        # Process all prefixed unit that are listed in the custom:scaledBy property
        prefixed_composed_units_ids = get_values(
            npu_dict.get("custom:scaledBy", []), "@id"
        )
        prefixed_unit_entities = []
        for prefixed_composed_unit_id in prefixed_composed_units_ids:
            pcu_index = ontologies["qudt"]["id_to_index"][prefixed_composed_unit_id]
            pcu_dict = ontologies["qudt"]["jsonld"]["@graph"][pcu_index]
            pcu_data = replace_keys(
                pcu_dict,
                {
                    "qudt:symbol": "main_symbol"
                    # todo: check if correct or should here only be a ref to the npu?
                },
            )
            prefix_id = pcu_data["qudt:prefix"]["@id"]
            prefix_index = ontologies["qudt"]["id_to_index"][prefix_id]
            prefix_uuid = str(
                uuid_module.uuid5(
                    namespace=uuid_module.NAMESPACE_URL,
                    name=resolve_prefix(prefix_id, "qudt"),
                )
            )
            pu_uuid = str(
                uuid_module.uuid5(
                    namespace=uuid_module.NAMESPACE_URL,
                    name=resolve_prefix(prefixed_composed_unit_id, "qudt"),
                )
            )
            pcu_data.update(
                {
                    "uuid": pu_uuid,
                    "osw_id": f"{npu_osw_id}#OSW{pu_uuid.replace('-', '')}",
                    "label": get_label_from_dict(pcu_data),
                    "description": get_desc_from_dict(pcu_data),
                    "prefix": f"Item:OSW{prefix_uuid.replace('-', '')}",
                    "prefix_symbol": ontologies["qudt"]["jsonld"]["@graph"][
                        prefix_index
                    ]["qudt:symbol"],
                    "ucum_codes": get_values(pcu_data["qudt:ucumCode"], "@value")
                    if "qudt:ucumCode" in pcu_data
                    else [],
                }
            )
            # todo: OntologyRelated properties
            #  - [ ] exact_ontology_match --> look in other ontologies
            #  - [ ] close_ontology_match
            #  - [x] qudt:plainTextDescription / dcterms:description
            #  - [x] label
            #  - [ ] qudt:informativeReference
            #  - [x] qudt:ucumCode
            #  - [x] qudt:symbol
            #  - [ ] qudt:iec61360Code
            #  - [ ] qudt:uneceCommonCode
            #  - [ ] qudt:applicableSystem
            #  - [ ] qudt:hasQuantityKind
            #  - [ ] qudt:scalingOf / custom:scaledBy
            prefixed_unit_entities.append(model.PrefixUnit(**pcu_data))
        # Creating the non-prefixed unit entity
        npu_data = replace_keys(
            npu_dict,
            {
                "qudt:symbol": "main_symbol",
                "qudt:currencyCode": "main_symbol",
            },
        )
        npu_data.update(
            {
                "uuid": npu_uuid,
                "osw_id": npu_osw_id,
                "label": get_label_from_dict(npu_data),
                "description": get_desc_from_dict(npu_data),
                "conversion_factor_from_si": npu_data["qudt:conversionMultiplier"].get(
                    "@value"
                ),  # todo: copy from Andreas / Matthias
                "ucum_codes": get_values(npu_data["qudt:ucumCode"], "@value")
                if "qudt:ucumCode" in npu_data
                else [],
                "prefix_units": prefixed_unit_entities,
            }
        )
        if "main_symbol" not in npu_data:
            npu_data["main_symbol"] = npu_id.split(":")[-1]
        # todo: OntologyRelated properties
        #  - [ ] exact_ontology_match --> look in other ontologies
        #  - [ ] close_ontology_match
        #  - [ ] qudt:hasDimensionVector
        #  - [x] qudt:plainTextDescription / dcterms:description
        #  - [x] label
        #  - [ ] qudt:informativeReference
        #  - [x] qudt:ucumCode
        #  - [x] symbol
        #  - [ ] qudt:iec61360Code
        #  - [ ] qudt:uneceCommonCode
        #  - [ ] qudt:applicableSystem
        #  - [ ] qudt:hasQuantityKind
        #  - [ ] qudt:scalingOf / custom:scaledBy
        # todo: check usage of conversionMultiplier
        non_prefixed_unit_entities.append(model.QuantityUnit(**npu_data))

    # Processing composed units with/without prefix(es)
    for npcu_dict in unit_type_dict["Non-prefixed, composed unit"]:
        npcu_id = npcu_dict["@id"]
        npcu_index = ontologies["qudt"]["id_to_index"][npcu_id]
        # Get the enriched version of the non-prefixed composed unit from the jsonld
        npcu_dict = ontologies["qudt"]["jsonld"]["@graph"][npcu_index]
        npcu_data = replace_keys(npcu_dict, {"qudt:symbol": "main_symbol"})
        npcu_uuid = str(
            uuid_module.uuid5(
                namespace=uuid_module.NAMESPACE_URL,
                name=resolve_prefix(npcu_id, "qudt"),
            )
        )
        npcu_osw_id = f"Item:OSW{npcu_uuid.replace('-', '')}"
        # Process all prefixed unit that are listed in the custom:scaledBy property
        prefixed_composed_units_ids = get_values(
            npcu_dict.get("custom:scaledBy", []), "@id"
        )
        prefixed_composed_unit_entities = []
        for prefixed_composed_unit_id in prefixed_composed_units_ids:
            pcu_index = ontologies["qudt"]["id_to_index"][prefixed_composed_unit_id]
            pcu_dict = ontologies["qudt"]["jsonld"]["@graph"][pcu_index]
            pcu_uuid = str(
                uuid_module.uuid5(
                    namespace=uuid_module.NAMESPACE_URL,
                    name=resolve_prefix(prefixed_composed_unit_id, "qudt"),
                )
            )
            pcu_data = replace_keys(pcu_dict, {"qudt:symbol": "main_symbol"})
            pcu_data.update(
                {
                    "main_symbol": npcu_dict["qudt:symbol"],
                    "uuid": pcu_uuid,
                    "osw_id": f"{npcu_osw_id}#OSW{pcu_uuid.replace('-', '')}",
                    "label": get_label_from_dict(pcu_data),
                    "description": get_desc_from_dict(pcu_data),
                    # todo: factor_units
                    "ucum_codes": get_values(pcu_data["qudt:ucumCode"], "@value")
                    if "qudt:ucumCode" in pcu_data
                    else [],
                    "conversion_factor_from_si": npcu_data[
                        "qudt:conversionMultiplier"
                    ].get("@value"),  # todo: copy from Andreas / Matthias
                }
            )
            # todo: OntologyRelated properties
            #  - [ ] exact_ontology_match --> look in other ontologies
            #  - [ ] close_ontology_match
            #  - [ ] qudt:hasDimensionVector
            #  - [x] qudt:plainTextDescription / dcterms:description
            #  - [x] label
            #  - [ ] qudt:informativeReference
            #  - [x] qudt:ucumCode
            #  - [x] symbol
            #  - [ ] qudt:iec61360Code
            #  - [ ] qudt:uneceCommonCode
            #  - [ ] qudt:applicableSystem
            #  - [ ] qudt:hasQuantityKind
            #  - [ ] qudt:scalingOf / custom:scaledBy
            prefixed_composed_unit_entities.append(
                model.ComposedQuantityUnitWithUnitPrefix(**pcu_data)
            )
        # Creating the non-prefixed composed unit entity
        npcu_data.update(
            {
                "uuid": npcu_uuid,
                "osw_id": npcu_osw_id,
                "label": get_label_from_dict(npcu_data),
                "description": get_desc_from_dict(npcu_data),
                "conversion_factor_from_si": npcu_data["qudt:conversionMultiplier"].get(
                    "@value"
                ),  # todo: copy from Andreas / Matthias
                "ucum_codes": get_values(npcu_data["qudt:ucumCode"], "@value")
                if "qudt:ucumCode" in npcu_data
                else [],
                "prefixed_composed_units": prefixed_composed_unit_entities,
            }
        )
        if "main_symbol" not in npcu_data:
            npcu_data["main_symbol"] = npcu_id.split(":")[-1]
        # todo: check if correct model is used here:
        non_prefixed_unit_entities.append(model.ComposedUnit(**npcu_data))

    return non_prefixed_unit_entities

    # Non-composed units
    # - get non-prefixed non-composed units
    # - access the scaledBy property to find prefixed units
    # - create PrefixedQuantityUnit entities for each prefixed unit
    # - create the QuantityUnit entity for the non-prefixed base unit and list the
    #   PrefixedQuantityUnit entities in the 'prefixed_units' property
    # Composed units
    # - get non-prefixed composed units
    # - access the scaledBy property to find prefixed units
    # - create PrefixedComposedQuantityUnit entities for each prefixed unit
    # - create ComposedQuantityUnit entities for each composed unit and list the
    #   PrefixedComposedQuantityUnit entities in the 'scaledBy' property


@log_call
def load_ontology(ontology_acronym: str = "qudt", use_cache: bool = True) -> None:
    od = ontologies.get(ontology_acronym)
    """ontology dict"""
    if use_cache and od["dump_fp"] and Path(od["dump_fp"]).exists():
        od["jsonld"] = load_jsonld_from_file(Path(od["dump_fp"]))
        _logger.info(
            "Loaded cached ontology %s dump from cache at %s",
            ontology_acronym,
            od["dump_fp"],
        )
    else:
        _logger.info("Downloading ontology dump from %s", od["url"])
        od["jsonld"] = load_ontology_dump(ontology_acronym)
        save_jsonld_to_file(od["jsonld"], Path(od["dump_fp"]))
        _logger.info(
            "Ontology dump processing finished and saved to %s",
            od["dump_fp"],
        )


@log_call
def prepare_ontology(ontology_acronym: str, use_cache: bool = True) -> None:
    od = ontologies.get(ontology_acronym)
    """ontology dict"""
    if not od:
        raise ValueError(f"Ontology {ontology_acronym} not found.")

    load_ontology(ontology_acronym, use_cache)

    # Create iri_dict, type_dict and type_index
    od["id_dict"], od["id_to_index"] = build_iri_dict(od["jsonld"])
    od["type_dict"] = build_type_dict(od["jsonld"])
    od["type_index"] = build_type_index(od["jsonld"])
    od["ids"] = list(od["id_dict"].keys())


@log_call
def prepare_all_ontologies(use_cache: bool = True) -> None:
    _logger.info("Loading ontologies...")
    for ontology, value in ontologies.items():
        _logger.info(" - %s: %s", ontology, value["url"])
        prepare_ontology(ontology, use_cache)
        _logger.info("Ontology %s loaded and prepared.", ontology)


def build_indices():
    qudt_id_dict_, qudt_id_to_index_ = build_iri_dict(ontologies["qudt"]["jsonld"])
    ontologies["qudt"]["id_dict"] = qudt_id_dict_
    ontologies["qudt"]["id_to_index"] = qudt_id_to_index_
    ontologies["qudt"]["type_dict"] = build_type_dict(ontologies["qudt"]["jsonld"])
    ontologies["qudt"]["type_index"] = build_type_index(ontologies["qudt"]["jsonld"])


# ---------------------
# Main script execution
# ---------------------
logging.basicConfig(level=logging.INFO)

prepare_all_ontologies(use_cache=True)
build_indices()

# Resolve factor unit pointers (something like
# {"@id": "_:n9239b5585098485ba36e2d5401954d25b4796"})
resolve_factor_units(ontologies["qudt"]["jsonld"], ontologies["qudt"]["id_dict"])
# Handling of units
qudt_unit_type_dict = classify_and_enrich_qudt_units(
    ontologies["qudt"]["type_dict"],
    ontologies["qudt"]["id_dict"],
    ontologies["qudt"]["id_to_index"],
)
# Until now the QUDT dump should not have been altered but the units should
#  have been classified and scaledB / scalingOf should be present
# Create units that are missing in the QUDT
missing_units = load_qudt_missing_units()
# Add these to the QUDT jsonld and type dictionary
ontologies["qudt"]["jsonld"]["@graph"].extend(missing_units)
# Rebuild the id_dict and type_dict to include the new units
build_indices()

# Process prefixed, composed units with no Non-prefixed base unit with the help of AI
ai_generated_composed_base_units = process_prefixed_composed_units_with_ai(
    qudt_unit_type_dict, ontologies["qudt"]["id_to_index"]
)
# Add the AI-generated units to the QUDT jsonld and type dictionary
ontologies["qudt"]["jsonld"]["@graph"].extend(ai_generated_composed_base_units)
# Rebuild the id_dict and type_dict to include the new units
build_indices()


# Rerun the classification again to include the new units
qudt_unit_type_dict = classify_and_enrich_qudt_units(
    ontologies["qudt"]["type_dict"],
    ontologies["qudt"]["id_dict"],
    ontologies["qudt"]["id_to_index"],
)
if len(qudt_unit_type_dict["Prefixed, composed unit with no non-prefixed base"]) != 0:
    raise ValueError(
        "There are still prefixed, composed units with no Non-prefixed base unit left "
        "after processing with AI."
    )
save_jsonld_to_file(
    jsonld_data=ontologies["qudt"]["jsonld"],
    filepath=ontologies["qudt"]["dump_fp"].with_suffix(".enriched.json"),
)

# Report on unit types
report_on_unit_types(
    ontologies["qudt"]["type_dict"],
    qudt_unit_type_dict,
)

# Handling of quantity kinds
qudt_quantity_kind_dict = classify_quantity_kinds(ontologies["qudt"]["type_dict"])


# Create prefixes
enrich_prefixes_with_ontology_matches(ontologies["qudt"]["type_index"])
unit_prefix_entities = create_unit_prefix_entities()
# Create units and prefixed units
quantity_unit_entities = create_quantity_unit_entities(qudt_unit_type_dict)


# todo: find why there are still units with missing scalingOf listed (181) and also
#  as missing scalingOf and base unit available (181)
