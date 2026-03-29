from typing import Any, Dict, List, Mapping


SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION,)
REQUIRED_TOP_LEVEL_SECTIONS = ("events", "run_metadata", "alignment", "summary")


class SchemaValidationError(ValueError):
    pass


def build_schema(
    events: List[Any],
    run_metadata: Dict[str, Any],
    alignment: Dict[str, Any],
    summary: Dict[str, Any],
    schema_version: int = SCHEMA_VERSION,
) -> Dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "events": events,
        "run_metadata": run_metadata,
        "alignment": alignment,
        "summary": summary,
    }
    validate_schema(payload)
    return payload


def validate_schema(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SchemaValidationError("Schema payload must be a mapping")

    if "schema_version" not in payload:
        raise SchemaValidationError("Missing required field: schema_version")

    schema_version = payload["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        expected_versions = ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
        raise SchemaValidationError(
            f"Incompatible schema_version: expected one of [{expected_versions}], got {schema_version}"
        )

    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in payload:
            raise SchemaValidationError(f"Missing required field: {section}")

    return payload


def parse_schema(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return validate_schema(payload)
