"""Deterministic OpenAPI export tests."""

import json

from scripts.export_openapi import rendered_openapi


def test_openapi_export_is_deterministic_and_contains_both_projections() -> None:
    first = rendered_openapi()
    second = rendered_openapi()

    assert first == second
    assert first.endswith("\n")

    schema = json.loads(first)
    paths = schema["paths"]
    assert "/api/v1/predictions" in paths
    assert "/api/v1/la/predictions" in paths
    assert schema["info"]["title"] == "Spine Screening API"
