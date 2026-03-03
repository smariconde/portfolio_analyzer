from __future__ import annotations

from typing import Any


REQUIRED_TOP_LEVEL_KEYS = {
    "ticker",
    "valuation_method",
    "assumptions",
    "scenarios",
    "result",
    "diagnostics",
    "rationale",
}


def validate_output_schema(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL_KEYS - set(payload.keys())
    if missing:
        errors.append(f"missing top-level keys: {sorted(missing)}")

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict):
        errors.append("scenarios must be a dictionary")
    else:
        for name in ("bear", "base", "bull"):
            if name not in scenarios:
                errors.append(f"missing scenario: {name}")
                continue
            scenario = scenarios[name]
            if "intrinsic_value_per_share" not in scenario:
                errors.append(f"scenario {name} missing intrinsic_value_per_share")
            if "upside_downside_pct" not in scenario:
                errors.append(f"scenario {name} missing upside_downside_pct")

    result = payload.get("result")
    if not isinstance(result, dict):
        errors.append("result must be a dictionary")
    else:
        for key in (
            "intrinsic_value_per_share",
            "current_price",
            "upside_downside_pct",
            "margin_of_safety_price",
        ):
            if key not in result:
                errors.append(f"result missing {key}")

    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        errors.append("diagnostics must be a dictionary")
    else:
        for key in ("model_risk", "data_quality", "checks"):
            if key not in diagnostics:
                errors.append(f"diagnostics missing {key}")

    return (len(errors) == 0, errors)
