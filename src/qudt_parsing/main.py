"""Loads the QUDT dump from a URL, processes it, and saves it as JSON-LD.
This script uses rdflib to parse the Turtle format and pyld to compact the
JSON-LD. It also provides functions to save and load JSON-LD data from files.
"""

import json
import logging
from pathlib import Path

from pyld import jsonld
from rdflib import Graph

_logger = logging.getLogger(__name__)

QUDT_CONTEXT = {
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


def load_qudt_dump(url: str) -> dict:
    """Load QUDT dump from a URL into an rdflib Graph."""
    g = Graph()
    g.parse(url, format="turtle")
    _logger.info("Loaded %d triples from %s", len(g), url)
    jsonld_dict = json.loads(g.serialize(format="json-ld"))
    jsonld_compacted = jsonld.compact(jsonld_dict, QUDT_CONTEXT)
    return jsonld_compacted


def save_jsonld_to_file(jsonld_data: dict, filepath: Path):
    """Save JSON-LD data to a file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(jsonld_data, f, indent=2)
    _logger.info("JSON-LD data saved to %s", filepath)


def load_jsonld_from_file(filepath: Path) -> dict:
    """Load JSON-LD data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
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


def build_iri_dict(graph: dict) -> dict:
    """Creates a dictionary with IRIs as keys and items as values."""
    iri_dict = {}
    for item in graph.get("@graph", []):
        iri = item.get("@id")
        if iri:
            if iri not in iri_dict:
                iri_dict[iri] = item
            else:
                _logger.warning("Duplicate IRI found: %s", iri)
        else:
            _logger.warning("Item without @id found: %s", item)
    return iri_dict


def resolve_factor_units(graph: dict, iri_dict: dict):
    """Replaces hasFactorUnit references with the actual entries."""
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


def build_type_dict(graph: dict) -> dict:
    """Creates a dictionary with types as keys and lists of items as values."""
    type_dict = {}
    # Iterate over the @graph and filter for @type
    for item in graph.get("@graph", []):
        if "@type" in item:
            if not isinstance(item["@type"], list):
                item["@type"] = [item["@type"]]
            _logger.debug(
                "Item: %s - Type: %s", item.get("@id", "No ID"), item["@type"]
            )
            for type_name in item["@type"]:
                type_dict.setdefault(type_name, []).append(item)
        else:
            _logger.warning("Item: %s - No type found", item.get("@id", "No ID"))

    _logger.info("Types found in the QUDT dump:")
    for type_name, items in type_dict.items():
        _logger.info(" - Type: %s - Count: %d", type_name, len(items))
    return type_dict


def classify_units(type_dict: dict, iri_dict: dict) -> tuple:
    """Classifies units according to various criteria."""
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
    for unit in type_dict.get("qudt:Unit", []):
        unit_type_dict["All"].append(unit)
        unit_id = unit.get("@id", "No ID")
        label = unit.get("rdfs:label", ["No label"])
        if isinstance(label, list):
            label = label[0]
        _logger.debug("Unit ID: %s - Label: %s", unit_id, label)

        # SI units (can be also derived units)
        if unit.get("qudt:conversionMultiplier", {}).get("@value") == "1.0":
            unit_type_dict["SI unit"].append(unit)
            _logger.info("SI Unit found: %s - Label: %s", unit_id, label)
        # Derived units
        if "qudt:DerivedUnit" in unit.get("@type", []):
            unit_type_dict["Derived unit"].append(unit)
            _logger.info("Derived Unit found: %s - Label: %s", unit_id, label)
        # Prefixed units
        if "qudt:prefix" in unit:
            unit_type_dict["Units with a prefix statement"].append(unit)
            _logger.info("Prefixed Unit found: %s - Label: %s", unit_id, label)

            # Prefixed non-composed unit
            if "qudt:hasFactorUnit" not in unit:
                unit_type_dict["Prefixed, non-composed unit"].append(unit)
                _logger.info(" - (also listed as Non-prefixed, non-composed Unit)")
            # See if the unit can be reduced to a Non-prefixed base unit
            base_unit_name = remove_prefixes(unit["@id"])
            found_base_units = [
                base_unit_name
                for candidate in type_dict.get("qudt:Unit", [])
                if base_unit_name in candidate.get("@id", "")
            ]
            for base_unit in found_base_units:
                _logger.info(" - Base Non-prefixed unit found: %s", base_unit)
            base_unit_found = any(
                base_unit_name in candidate.get("@id", "")
                for candidate in type_dict.get("qudt:Unit", [])
            )
            if base_unit_found and "qudt:scalingOf" not in unit:
                unit_type_dict["Prefixed unit with missing scalingOf"].append(unit)
                _logger.info(" - (also listed as Prefixed unit with missing scalingOf)")
            # Prefixed unit with missing scalingOf
            if not base_unit_found and "qudt:scalingOf" not in unit:
                # This case should not happen, as all prefixed units should be
                # reducible to a Non-prefixed base unit, and if not then they
                # should have a scalingOf property
                # There is one case:
                # https://qudt.org/vocab/unit/FT3
                # https://qudt.org/vocab/unit/KiloCubicFT
                unit_type_dict[
                    "Prefixed, non-composed unit with no Non-prefixed base"
                ].append(unit)
                _logger.info(
                    " - (also listed as Prefixed, non-composed Unit with no "
                    "Non-prefixed base)"
                )

        # Non-prefixed units
        if "qudt:prefix" not in unit:
            unit_type_dict["Units without a prefix statement"].append(unit)
            _logger.info("Non-prefixed Unit found: %s - Label: %s", unit_id, label)
            # Non-prefixed non-composed unit
            if "qudt:hasFactorUnit" not in unit:
                unit_type_dict["Non-prefixed, non-composed unit"].append(unit)
                _logger.info(
                    "Non-prefixed, non-composed Unit found: %s - Label: %s",
                    unit_id,
                    label,
                )
        # Composed units
        if "qudt:hasFactorUnit" in unit:
            unit_type_dict["Composed unit"].append(unit)
            _logger.info("Composite Unit found: %s - Label: %s", unit_id, label)
            # Non-prefixed unit composed units
            prefixed = False
            for factor_unit in unit["qudt:hasFactorUnit"]:
                factor_unit_id = factor_unit.get("qudt:hasUnit", {}).get("@id")
                factor_unit_dict = iri_dict.get(factor_unit_id, {})
                if "qudt:prefix" in factor_unit_dict:
                    prefixed = True
                    break
            # Prefixed composed unit
            if prefixed:
                unit_type_dict["Units with a prefix statement"].append(unit)
                _logger.info(" - (also listed as Prefixed Unit)")
                unit_type_dict["Prefixed, composed unit"].append(unit)
                _logger.info(" - (also listed as Prefixed, composed Unit)")
                # Check if the unit can be reduced to a Non-prefixed base unit
                base_unit_name = remove_prefixes(unit["@id"])
                base_unit_found = False
                for candidate in type_dict.get("qudt:Unit", []):
                    if (
                        base_unit_name in candidate.get("@id", "")
                        and "qudt:prefix" not in candidate
                    ):
                        base_unit_found = True
                        base_units_to_prefixed_composed_units.setdefault(
                            base_unit_name, []
                        ).append(unit["@id"])
                        _logger.info(
                            " - Base Non-prefixed unit found: %s", base_unit_name
                        )
                        break
                if "qudt:scalingOf" not in unit:
                    unit_type_dict[
                        "Prefixed, composed unit with missing scalingOf"
                    ].append(unit)
                    _logger.info(
                        " - (also listed as Prefixed, composed unit with missing "
                        "scalingOf)"
                    )
                if base_unit_found and "qudt:scalingOf" not in unit:
                    unit_type_dict[
                        (
                            "Prefixed, composed unit with missing scalingOf but base unit "
                            "available"
                        )
                    ].append(unit)
                    _logger.info(
                        " - (also listed as Prefixed, composed unit with missing "
                        "scalingOf but base unit available)"
                    )
                if not base_unit_found:
                    unit_type_dict[
                        "Prefixed, composed unit with no Non-prefixed base"
                    ].append(unit)
                    _logger.info(
                        " - (also listed as Prefixed, composed Unit with no "
                        "Non-prefixed base)"
                    )
            # Non-prefixed composed unit
            else:
                unit_type_dict["Units without a prefix statement"].append(unit)
                _logger.info("Non-prefixed Unit found: %s - Label: %s", unit_id, label)
                unit_type_dict["Non-prefixed, composed unit"].append(unit)
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
    return unit_type_dict, base_units_to_prefixed_composed_units


def report_unit_types(
    type_dict: dict, unit_type_dict: dict, base_units_to_prefixed_composed_units: dict
):
    """Erstellt eine Übersicht der gefundenen Einheitentypen."""
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
    unit_sets = {k: set(u.get("@id") for u in v) for k, v in unit_type_dict.items()}
    # Calculate the intersection of the sets pairwise
    intersection_count = 0
    for i, (k1, v1) in enumerate(unit_sets.items()):
        for k2, v2 in list(unit_sets.items())[i + 1 :]:
            intersection = v1.intersection(v2)
            if intersection:
                intersection_count += len(intersection)
                _logger.info("Units in both %s and %s: %d", k1, k2, len(intersection))
    _logger.info("Total units listed in multiple categories: %d", intersection_count)


def main():
    logging.basicConfig(level=logging.INFO)
    url_to_qudt = "https://qudt.org/3.1.4/qudt-all.ttl"
    path_to_dump = Path("qudt_dump.jsonld")
    use_cache = True

    if use_cache and path_to_dump.exists():
        graph = load_jsonld_from_file(path_to_dump)
        _logger.info("Loaded cached QUDT dump from %s", path_to_dump)
    else:
        _logger.info("Downloading QUDT dump from %s", url_to_qudt)
        graph = load_qudt_dump(url_to_qudt)
        _logger.info("Processing QUDT dump")
        save_jsonld_to_file(graph, path_to_dump)
        _logger.info("QUDT dump processing finished and saved to %s", path_to_dump)

    iri_dict = build_iri_dict(graph)
    resolve_factor_units(graph, iri_dict)
    type_dict = build_type_dict(graph)
    unit_type_dict, base_units_to_prefixed_composed_units = classify_units(
        type_dict, iri_dict
    )
    report_unit_types(type_dict, unit_type_dict, base_units_to_prefixed_composed_units)
    # ...further processing, e.g. handling of quantity kinds...


if __name__ == "__main__":
    main()
