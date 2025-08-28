"""Loads the QUDT dump from a URL, processes it, and saves it as JSON-LD.
This script uses rdflib to parse the Turtle format and pyld to compact the
JSON-LD. It also provides functions to save and load JSON-LD data from files.
"""

from __future__ import annotations

import json
import logging
import os
import uuid as uuid_module
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from osw.model import entity as model
from pyld import jsonld
from rdflib import Graph

_logger = logging.getLogger(__name__)

ENV_FP = Path(__file__).parents[2] / ".env"
load_dotenv(ENV_FP)

osl_domain = os.getenv("OSL_DOMAIN")

this_file = Path(__file__)
project_root = this_file.parents[2]
data_dir = project_root / "data"
data_dir.mkdir(exist_ok=True)


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
        self, required_funcs: list[str], warn: bool = True, raise_err: bool = True
    ) -> bool:
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
    iri_dict: dict[str, dict[str, Any]] | None
    iri_to_index: dict[str, int] | None
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
        "dump_fp": data_dir / "qudt_dump.jsonld",
        "graph": None,
        "jsonld": None,
        "iri_dict": None,
        "iri_to_index": None,
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
        "dump_fp": data_dir / "wikidata_dump.jsonld",
        "graph": None,
        "jsonld": None,
        "iri_dict": None,
        "iri_to_index": None,
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
        "dump_fp": data_dir / "om2_dump.jsonld",
        "graph": None,
        "jsonld": None,
        "iri_dict": None,
        "iri_to_index": None,
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
        "dump_fp": data_dir / "sdf_prefixes_dump.jsonld",
        "graph": None,
        "jsonld": None,
        "iri_dict": None,
        "iri_to_index": None,
        "type_dict": None,
        "type_index": None,
        "ids": None,
    },
    # todo: include other parts of the SI Digital Framework
    #  - quantities
    #  - units
    #  - constants
}

PREFIXES = [
    "Atto",
    "Centi",
    "Deca",
    "Deci",
    "Deka",
    "Exa",
    "Exbi",
    "Femto",
    "Gibi",
    "Giga",
    "Hecto",
    "Kibi",
    "Kilo",
    "Mebi",
    "Mega",
    "Micro",
    "Milli",
    "Nano",
    "Pebi",
    "Peta",
    "Pico",
    "Quecto",
    "Quetta",
    "Ronna",
    "Ronto",
    "Tebi",
    "Tera",
    "Yobi",
    "Yocto",
    "Yotta",
    "Zebi",
    "Zepto",
    "Zetta",
]


def get_values[T](
    inp: dict[str, dict[str, T]] | list[dict[str, T]], key: str
) -> list[T]:
    """Get values from a dictionary, inside a list of dictionaries or a single
    dictionary."""
    match inp:
        case dict():
            if key in inp:
                return [inp[key]]
            else:
                return [val for v in inp.values() for val in get_values(v, key) if v]
        case list():
            return [val for item in inp for val in get_values(item, key)]


def replace_keys[T](inp: dict[str, T], replacements: dict[str, str]) -> dict[str, T]:
    """Replace keys in a dictionary according to a replacements mapping."""
    return {replacements.get(k, k): v for k, v in inp.items()}


def resolve_prefix(inp: str, ontology: str) -> str:
    """Resolve a prefixed IRI to a full IRI using the ontology context."""
    if ":" not in inp:
        raise ValueError(f"Input {inp} does not contain a prefix.")
    prefix, suffix = inp.split(":", 1)
    if prefix in ontologies[ontology]["context"]:
        return ontologies[ontology]["context"][prefix] + suffix
    raise ValueError(f"Prefix {prefix} not found in context of ontology {ontology}.")


@log_call
def load_ontology_dump(ontology_acronym: str) -> dict:
    """Load QUDT dump from a URL into an rdflib Graph."""
    g = Graph()
    url = ontologies[ontology_acronym]["url"]
    g.parse(url, format=ontologies[ontology_acronym]["format"])
    _logger.info("Loaded %d triples from %s", len(g), url)
    jsonld_dict = json.loads(g.serialize(format="json-ld"))
    jsonld_compacted = jsonld.compact(
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


@log_call
def resolve_factor_units(graph: dict, iri_dict: dict):
    """Replaces hasFactorUnit references with the actual entries."""
    func_log.has_required_calls(["prepare_all_ontologies", "build_iri_dict"])
    for item in graph.get("@graph", []):
        if "qudt:Unit" not in item.get("@type", []):
            continue
        if "qudt:hasFactorUnit" in item:
            factor_units = item["qudt:hasFactorUnit"]
            if not isinstance(factor_units, list):
                factor_units = [factor_units]
            replacement = []
            for factor_unit in factor_units:
                factor_unit_iri = factor_unit.get("@id")
                if factor_unit_iri and factor_unit_iri not in iri_dict:
                    _logger.warning(
                        "Factor unit IRI %s not found in iri_dict", factor_unit_iri
                    )
                else:
                    replacement.append(iri_dict.get(factor_unit_iri, factor_unit))
            item["qudt:hasFactorUnit"] = replacement


@log_call
def build_iri_dict(jsonld: dict) -> tuple[dict, dict[str, int]]:
    """Creates a dictionary with IRIs as keys and items as values."""
    func_log.has_required_calls(["load_ontology"])
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
    func_log.has_required_calls(["load_ontology"])
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
    func_log.has_required_calls(["load_ontology"])
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


@log_call
def classify_qudt_units(type_dict: dict, iri_dict: dict, iri_to_index: dict) -> tuple:
    """Classifies units according to various criteria."""
    func_log.has_required_calls(["resolve_factor_units"])
    unit_type_dict = {
        "All": [],
        "SI unit": [],
        "Derived unit": [],
        "Composed unit": [],
        # Actually contains also composed units  with prefix(es):
        "Units without a prefix statement": [],
        "Non-prefixed unit": [],
        "Non-prefixed, non-composed unit": [],
        "Non-prefixed, composed unit": [],
        # Actually does not contain composed units with prefix(es):
        "Units with a prefix statement": [],
        "Prefixed unit": [],
        "Prefixed, non-composed unit": [],
        "Prefixed, composed unit": [],
        "Prefixed, composed unit with no Non-prefixed base": [],
        "Prefixed, composed unit with missing scalingOf": [],
        "Prefixed, composed unit with missing scalingOf but base unit available": [],
        "Prefixed, non-composed unit with no Non-prefixed base": [],
        "Prefixed unit with missing scalingOf": [],
    }
    # non prefixed (base) units and their prefixed composed units
    base_units_to_prefixed_composed_units = {}

    # Iterate over all "qudt:Unit" items and print their @id and label
    for unit_dict in type_dict.get("qudt:Unit", []):
        unit_type_dict["All"].append(unit_dict)
        unit_id = unit_dict.get("@id", "No ID")
        unit_index = iri_to_index.get(unit_id)
        # Use the pointer to the jsonld graph to update the original jsonld:
        unit_dict = ontologies["qudt"]["jsonld"]["@graph"][unit_index]
        label = unit_dict.get("rdfs:label", ["No label"])
        if isinstance(label, list):
            label = label[0]
        _logger.debug("Unit ID: %s - Label: %s", unit_id, label)

        # SI units (can be also derived units)
        if unit_dict.get("qudt:conversionMultiplier", {}).get("@value") == "1.0":
            unit_type_dict["SI unit"].append(unit_dict)
            _logger.info("SI Unit found: %s - Label: %s", unit_id, label)
        # Derived units
        if "qudt:DerivedUnit" in unit_dict.get("@type", []):
            unit_type_dict["Derived unit"].append(unit_dict)
            _logger.info("Derived Unit found: %s - Label: %s", unit_id, label)
        # Prefixed units
        if "qudt:prefix" in unit_dict:
            unit_type_dict["Units with a prefix statement"].append(unit_dict)
            _logger.info("Prefixed Unit found: %s - Label: %s", unit_id, label)

            # Prefixed non-composed unit
            if "qudt:hasFactorUnit" not in unit_dict:
                unit_type_dict["Prefixed, non-composed unit"].append(unit_dict)
                _logger.info(" - (also listed as Non-prefixed, non-composed Unit)")
            # See if the unit can be reduced to a Non-prefixed base unit
            base_unit_name = remove_prefixes(unit_dict["@id"])
            found_base_units = list(
                {
                    base_unit_name
                    for candidate in type_dict.get("qudt:Unit", [])
                    if base_unit_name in candidate.get("@id", "")
                }
            )

            for base_unit in found_base_units:
                _logger.info(" - Base Non-prefixed unit found: %s", base_unit)
            base_unit_found = any(
                base_unit_name in candidate.get("@id", "")
                for candidate in type_dict.get("qudt:Unit", [])
            )
            if base_unit_found and "qudt:scalingOf" not in unit_dict:
                unit_type_dict["Prefixed unit with missing scalingOf"].append(unit_dict)
                _logger.info(" - (also listed as Prefixed unit with missing scalingOf)")
            # Prefixed unit with missing scalingOf
            if not base_unit_found and "qudt:scalingOf" not in unit_dict:
                # This case should not happen, as all prefixed units should be
                # reducible to a Non-prefixed base unit, and if not then they
                # should have a scalingOf property
                # There is one case:
                # https://qudt.org/vocab/unit/FT3
                # https://qudt.org/vocab/unit/KiloCubicFT
                unit_type_dict[
                    "Prefixed, non-composed unit with no Non-prefixed base"
                ].append(unit_dict)
                _logger.info(
                    " - (also listed as Prefixed, non-composed Unit with no "
                    "Non-prefixed base)"
                )

        # Non-prefixed units
        if "qudt:prefix" not in unit_dict:
            unit_type_dict["Units without a prefix statement"].append(unit_dict)
            _logger.info("Non-prefixed Unit found: %s - Label: %s", unit_id, label)
            # Non-prefixed non-composed unit
            if "qudt:hasFactorUnit" not in unit_dict:
                unit_type_dict["Non-prefixed, non-composed unit"].append(unit_dict)
                _logger.info(
                    "Non-prefixed, non-composed Unit found: %s - Label: %s",
                    unit_id,
                    label,
                )
        # Composed units
        if "qudt:hasFactorUnit" in unit_dict:
            unit_type_dict["Composed unit"].append(unit_dict)
            _logger.info("Composite Unit found: %s - Label: %s", unit_id, label)
            # Non-prefixed unit composed units
            prefixed = False
            for factor_unit in unit_dict["qudt:hasFactorUnit"]:
                factor_unit_id = factor_unit.get("qudt:hasUnit", {}).get("@id")
                factor_unit_dict = iri_dict.get(factor_unit_id, {})
                if "qudt:prefix" in factor_unit_dict:
                    prefixed = True
                    break
            # Prefixed composed unit
            if prefixed:
                unit_type_dict["Units with a prefix statement"].append(unit_dict)
                _logger.info(" - (also listed as Prefixed Unit)")
                unit_type_dict["Prefixed, composed unit"].append(unit_dict)
                _logger.info(" - (also listed as Prefixed, composed Unit)")
                # Check if the unit can be reduced to a Non-prefixed base unit
                base_unit_name = remove_prefixes(unit_dict["@id"])
                base_unit_found = False
                for candidate in type_dict.get("qudt:Unit", []):
                    if (
                        base_unit_name in candidate.get("@id", "")
                        and "qudt:prefix" not in candidate
                    ):
                        base_unit_found = True
                        base_units_to_prefixed_composed_units.setdefault(
                            base_unit_name, []
                        ).append(unit_dict["@id"])
                        _logger.info(
                            " - Base Non-prefixed unit found: %s", base_unit_name
                        )

                        break
                if "qudt:scalingOf" not in unit_dict:
                    unit_type_dict[
                        "Prefixed, composed unit with missing scalingOf"
                    ].append(unit_dict)
                    _logger.info(
                        " - (also listed as Prefixed, composed unit with missing "
                        "scalingOf)"
                    )
                if base_unit_found and "qudt:scalingOf" not in unit_dict:
                    unit_type_dict[
                        (
                            "Prefixed, composed unit with missing scalingOf but base "
                            "unit available"
                        )
                    ].append(unit_dict)
                    _logger.info(
                        " - (also listed as Prefixed, composed unit with missing "
                        "scalingOf but base unit available)"
                    )
                if not base_unit_found:
                    unit_type_dict[
                        "Prefixed, composed unit with no Non-prefixed base"
                    ].append(unit_dict)
                    _logger.info(
                        " - (also listed as Prefixed, composed Unit with no "
                        "Non-prefixed base)"
                    )
            # Non-prefixed composed unit
            else:
                unit_type_dict["Units without a prefix statement"].append(unit_dict)
                _logger.info("Non-prefixed Unit found: %s - Label: %s", unit_id, label)
                unit_type_dict["Non-prefixed, composed unit"].append(unit_dict)
                _logger.info(" - (also listed as Non-prefixed Composed Unit)")

        # Handle composite units with hasFactorUnit - don't exist!

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

    # todo: process (Non-)prefixed and composed units and write the relationships to
    #  the jsonld
    #  - find Non-prefixed base unit for each Prefixed unit (composed and non-composed)
    #  - add them to the jsonld as qudt:scalingOf (if not already present)
    #  - for non-prefixed base units add the prefixed units as qudt:scaledBy

    # Going through the non-composed prefixed units and adding the scaledBy to the
    # non-composed non-prefixed base units

    for prefixed_unit in unit_type_dict["Prefixed, non-composed unit"]:
        base_unit_name = remove_prefixes(prefixed_unit["@id"])
        base_unit_found = False
        for candidate in type_dict.get("qudt:Unit", []):
            if (
                base_unit_name in candidate.get("@id", "")
                and "qudt:prefix" not in candidate
            ):
                base_unit_found = True
                if "qudt:scaledBy" not in candidate:
                    candidate["qudt:scaledBy"] = []
                if prefixed_unit["@id"] not in get_values(
                    candidate["qudt:scaledBy"], "@id"
                ):
                    candidate["qudt:scaledBy"].append({"@id": prefixed_unit["@id"]})
                    _logger.info(
                        "Added qudt:scaledBy %s to Non-prefixed base unit %s",
                        prefixed_unit["@id"],
                        candidate["@id"],
                    )
                if "qudt:scalingOf" not in prefixed_unit:
                    prefixed_unit["qudt:scalingOf"] = {"@id": candidate["@id"]}
                    _logger.info(
                        "Added qudt:scalingOf %s to Prefixed unit %s",
                        candidate["@id"],
                        prefixed_unit["@id"],
                    )
                break
        if not base_unit_found:
            _logger.warning(
                "No Non-prefixed base unit found for Prefixed unit %s",
                prefixed_unit["@id"],
            )

    return unit_type_dict, base_units_to_prefixed_composed_units


@log_call
def report_on_unit_types(
    type_dict: dict, unit_type_dict: dict, base_units_to_prefixed_composed_units: dict
):
    """Erstellt eine Übersicht der gefundenen Einheitentypen."""
    func_log.has_required_calls(["classify_qudt_units"])
    _logger.info("Unit types found in the QUDT dump:")
    for type_name, items in unit_type_dict.items():
        # Use unique set of items
        items = list({item["@id"]: item for item in items}.values())
        unit_type_dict[type_name] = items
        _logger.info(" - Type: %s - Count: %d", type_name, len(items))
    _logger.info(
        "Number of prefixed composed units with their Non-prefixed base units: %d",
        len(base_units_to_prefixed_composed_units),
    )
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
def classify_quantity_kinds(type_dict: dict) -> dict:  # , iri_dict: dict):
    """Classify quantity kinds into fundamental and non-fundamental."""
    func_log.has_required_calls(["build_type_dict"])
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
def enrich_prefixes_with_ontology_matches(type_index: dict[str, list[int]]) -> None:
    """Enrich the prefixes in the QUDT with exact ontology matches."""
    func_log.has_required_calls(["build_type_index"])
    for ii in type_index["qudt:Prefix"]:
        prefix_dict = ontologies["qudt"]["jsonld"]["@graph"][ii]
        prefix_id = prefix_dict.get("@id", "")
        if not prefix_id:
            _logger.warning("Prefix at index %d has no @id: %s", ii, prefix_dict)
            continue
        prefix_dict["owl:sameAs"] = [{"@id": resolve_prefix(prefix_id, "qudt")}]
        om_id_guess = f"om:{prefix_id.split(':')[-1].lower()}"
        # OM2 - Ontology of Units of Measure
        if om_id_guess in ontologies["om2"]["iri_dict"]:
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
        if sdf_id_guess in ontologies["sdf"]["iri_dict"]:
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
    func_log.has_required_calls(["enrich_prefixes_with_ontology_matches"])
    # Handling of prefixes
    prefixes = [
        ontologies["qudt"]["jsonld"]["@graph"][ii]
        for ii in ontologies["qudt"]["type_index"]["qudt:Prefix"]
    ]

    unit_prefix_entities = []
    for prefix_dict in prefixes:
        data = replace_keys(prefix_dict, {"qudt:symbol": "symbol"})
        data["uuid"] = uuid_module.uuid5(
            namespace=uuid_module.NAMESPACE_URL,
            name=data["@id"],  # todo: check correct field name
        )
        label_list = get_values(data, "rdfs:label")
        data["name"] = label_list[0]["@value"]
        data["exact_ontology_match"] = [
            sameas_dict["@id"] for sameas_dict in data.get("owl:sameAs", [])
        ]
        if "qudt:siExactMatch" in data:
            data["exact_ontology_match"].extend(
                [
                    resolve_prefix(item, "qudt")
                    for item in get_values(data["qudt:siExactMatch"], "@id")
                ]
            )
        if "qudt:dbpediaMatch" in data:
            data["exact_ontology_match"].extend(
                get_values(data["qudt:dbpediaMatch"], "@value")
            )
        if "qudt:exactMatch" in data:
            data["exact_ontology_match"].extend(
                [
                    resolve_prefix(item, "qudt")
                    for item in get_values(data["qudt:exactMatch"], "@id")
                ]
            )
        data["exact_ontology_match"] = list(set(data["exact_ontology_match"]))
        data["label"] = [
            replace_keys(label_dict, {"@value": "text", "@language": "lang"})
            for label_dict in label_list
        ]
        desc = (
            data["dcterms:description"]
            if isinstance(data.get("dcterms:description"), list)
            else (
                [data.get("dcterms:description", {})]
                if data.get("dcterms:description")
                else []
            )
        )
        # Might cause issues since this is a Latex string with special characters like
        #  \\{10^9\\} and similar
        data["description"] = [
            replace_keys(desc_dict, {"@value": "text", "@language": "lang"})
            for desc_dict in desc
        ]
        data["type"] = (
            ["Category:OSW99e0f46a40ca4129a420b4bb89c4cc45"],
        )  # Unit prefix
        data["factor"] = data["qudt:prefixMultiplier"]["@value"]
        unit_prefix_entities.append(model.UnitPrefix(**data))
    return unit_prefix_entities


@log_call
def prepare_ontology(ontology_acronym: str, use_cache: bool = True) -> None:
    od = ontologies.get(ontology_acronym)
    """ontology dict"""
    if not od:
        raise ValueError(f"Ontology {ontology_acronym} not found.")

    @log_call
    def load_ontology():
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

    load_ontology()

    # Create iri_dict, type_dict and type_index
    od["iri_dict"], od["iri_to_index"] = build_iri_dict(od["jsonld"])
    od["type_dict"] = build_type_dict(od["jsonld"])
    od["type_index"] = build_type_index(od["jsonld"])
    od["ids"] = list(od["iri_dict"].keys())


@log_call
def prepare_all_ontologies(use_cache: bool = True) -> None:
    _logger.info("Loading ontologies...")
    for ontology, value in ontologies.items():
        _logger.info(" - %s: %s", ontology, value["url"])
        prepare_ontology(ontology, use_cache)
        _logger.info("Ontology %s loaded and prepared.", ontology)


# ---------------------
# Main script execution
# ---------------------
logging.basicConfig(level=logging.INFO)

prepare_all_ontologies(use_cache=True)

qudt_jsonld = ontologies["qudt"]["jsonld"]

qudt_iri_dict = ontologies["qudt"]["iri_dict"]
qudt_iri_to_index = ontologies["qudt"]["iri_to_index"]
qudt_type_dict = ontologies["qudt"]["type_dict"]
qudt_type_index = ontologies["qudt"]["type_index"]

resolve_factor_units(qudt_jsonld, qudt_iri_dict)
# Handling of units
qudt_unit_type_dict, base_units_to_prefixed_composed_units = classify_qudt_units(
    qudt_type_dict, qudt_iri_dict, qudt_iri_to_index
)
report_on_unit_types(
    qudt_type_dict, qudt_unit_type_dict, base_units_to_prefixed_composed_units
)

# Handling of quantity kinds
qudt_quantity_kind_dict = classify_quantity_kinds(qudt_type_dict)

# Next
# Create units and prefixed units
enrich_prefixes_with_ontology_matches(qudt_type_index)
unit_prefix_entities = create_unit_prefix_entities()
