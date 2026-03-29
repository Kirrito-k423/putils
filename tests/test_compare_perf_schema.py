import re

import pytest

from putils.compare_perf.schema import (
    SCHEMA_VERSION,
    SchemaValidationError,
    build_schema,
    validate_schema,
)


def _valid_payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "events": [],
        "run_metadata": {},
        "alignment": {},
        "summary": {},
    }


def test_schema_version_required():
    payload = _valid_payload()
    payload.pop("schema_version")

    with pytest.raises(SchemaValidationError, match="Missing required field: schema_version"):
        validate_schema(payload)


def test_incompatible_schema_rejected():
    payload = _valid_payload()
    payload["schema_version"] = SCHEMA_VERSION + 1

    expected = (
        f"Incompatible schema_version: expected one of [{SCHEMA_VERSION}], "
        f"got {SCHEMA_VERSION + 1}"
    )
    with pytest.raises(SchemaValidationError, match=re.escape(expected)):
        validate_schema(payload)


def test_build_schema_has_required_top_level_sections():
    payload = build_schema(events=[], run_metadata={}, alignment={}, summary={})

    assert payload["schema_version"] == SCHEMA_VERSION
    assert "events" in payload
    assert "run_metadata" in payload
    assert "alignment" in payload
    assert "summary" in payload
