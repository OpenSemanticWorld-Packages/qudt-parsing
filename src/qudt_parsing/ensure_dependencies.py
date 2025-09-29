from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parents[2] / ".env")

from osw.defaults import params as default_params  # noqa: E402
from osw.defaults import paths as default_paths  # noqa: E402

default_params.wiki_domain = os.getenv("OSL_DOMAIN")
default_paths.cred_filepath = os.getenv("OSL_CRED_FP")

from osw.express import OswExpress  # noqa: E402

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
    "ComposedUnit": "Category:OSW6c2aea028a8647cd97f5d7c65c09cd44",
    "MainQuantityProperty": "Category:OSW1b15ddcf042c4599bd9d431cbfdf3430",
    "SubQuantityProperty": "Category:OSW69f251a900944602a08d1cca830249b5",
    "SystemOfQuantitiesAndUnits": "Category:OSW27782669526d4d9a8de83659c03c64d5",
    "CgsGaussSystemOfQuantitiesAndUnits": "Item:OSWd3a5d38f09d45265973a2564d7b19381",
    "ImperialSystemOfQuantitiesAndUnits": "Item:OSW950eb435703055a1b2fd0181d2f3a6d8",
    "InternationalSystemOfQuantitiesAndUnitsSi": "Item:OSW2f435fc0b3a75d29bcb1de887c14d753",
    "IsoSystemOfQuantitiesAndUnitsIsq": "Item:OSW437b30977e515659918229f97aa55985",
    "QuantityKindDimensionVectorType": "Category:OSWc3ea087b511947feaec050e00154c46c",
    "QuantityKindDimensionVector": "Category:OSW310a7c6a8c394bb89e4e840f18fcf05f",
    "QuantityKindDimensionVectorCGS": "Category:OSWd579625d68cf574c9154efcc3e042d87",
    "QuantityKindDimensionVectorImperial": "Category:OSW04e71bf0f6545881acb4e70306d5d9e2",
    "QuantityKindDimensionVectorSi": "Category:OSWd9049d763b9e52cd8249f0af4733a8ae",
    "QuantityKindDimensionVectorIso": "Category:OSW66635d7d0bb4522795d44ed22bc59094",
}

# import_with_fallback(DEPENDENCIES, globals(), osw_express=osw_obj)

osw_obj.fetch_schema(
    osw_obj.FetchSchemaParam(
        schema_title=list(DEPENDENCIES.values()),
        mode="replace",
    )
)
