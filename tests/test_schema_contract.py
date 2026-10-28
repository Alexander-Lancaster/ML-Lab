import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from ml_lab.config import ConfigurationError, parse_config


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema" / "architecture.schema.json").read_text())
FIXTURES = json.loads((ROOT / "schema" / "validation-fixtures.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


@pytest.mark.parametrize("case", FIXTURES, ids=lambda case: case["name"])
def test_shared_architecture_validation_contract(case):
    schema_errors = list(VALIDATOR.iter_errors(case["document"]))
    assert (not schema_errors) is case["valid"], [error.message for error in schema_errors]

    if case["valid"]:
        parse_config(case["document"])
    else:
        with pytest.raises(ConfigurationError):
            parse_config(case["document"])
