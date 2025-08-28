from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

from osw.defaults import params as default_params  # noqa: E402
from osw.defaults import paths as default_paths  # noqa: E402

default_params.wiki_domain = os.getenv("OSL_DOMAIN")
default_paths.cred_filepath = os.getenv("OSL_CRED_FP")

from osw.express import OswExpress, import_with_fallback  # noqa: E402

osw_obj = OswExpress(
    domain=default_params.wiki_domain, cred_filepath=default_paths.cred_filepath
)

DEPENDENCIES = {
    "QuantityKind": "Category:OSW00fbd6feecb5408997ca18d4e681a131",
    "QuantityUnit": "Category:OSWd2520fa016844e01af0097a85bb25b25",
    "UnitPrefix": "Category:OSW99e0f46a40ca4129a420b4bb89c4cc45",
    "Characteristic": "Category:OSW93ccae36243542ceac6c951450a81d47",
    "QuantityValue": "Category:OSW4082937906634af992cf9a1b18d772cf",
    "FundamentalQuantityValueType": "Category:OSWc7f9aec4f71f4346b6031f96d7e46bd7",
    "CharacteristicType": "Category:OSWffe74f291d354037b318c422591c5023",
    "QuantityValueType": "Category:OSWac07a46c2cf14f3daec503136861f5ab",
    "ComposedQuantityUnitWithUnitPrefix": "Category:OSW268cc84d3dff4a7ba5fd489d53254cb0",
    "MainQuantityProperty": "Category:OSW1b15ddcf042c4599bd9d431cbfdf3430",
    "SubQuantityProperty": "Category:OSW69f251a900944602a08d1cca830249b5",
}

import_with_fallback(DEPENDENCIES, globals(), osw_express=osw_obj)
