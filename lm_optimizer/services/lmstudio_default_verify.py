"""Async verification of saved LM Studio defaults via an echo load.

Loads the model with an empty ``LoadConfiguration()`` so LM Studio resolves
the saved default, compares the echoed load config against the expected
fields, and always unloads the model in ``finally``.
"""

from __future__ import annotations

from typing import Any

from lm_optimizer.domain.models import LoadConfiguration
from lm_optimizer.services.lmstudio_defaults import expected_echo_fields


async def verify_defaults(
    model_id: str,
    config: LoadConfiguration,
    *,
    client: Any,
    home: Any = None,
) -> dict[str, Any]:
    expected = expected_echo_fields(config)
    unverified = [
        attr
        for attr in ("gpu_ratio", "rope_freq_base", "rope_freq_scale")
        if getattr(config, attr, None) is not None
    ]

    identifier: str | None = None
    result: dict[str, Any] | None = None

    try:
        response = await client.load_model(model_id, LoadConfiguration())
        if not getattr(response, "success", False):
            result = {
                "verified": False,
                "checked": {},
                "mismatches": {},
                "unverified": unverified,
                "unloaded": False,
                "error": getattr(response, "error", None) or "load failed",
            }
        else:
            identifier = getattr(response, "identifier", None)
            loaded = getattr(response, "loaded_config", None)
            if loaded is None:
                loaded = getattr(response, "load_config", None)
            if loaded is None:
                result = {
                    "verified": False,
                    "checked": {},
                    "mismatches": {},
                    "unverified": unverified,
                    "unloaded": False,
                    "error": "server did not return load_config",
                }
            else:
                actual = loaded.model_dump(exclude_none=True)
                mismatches = {
                    key: {"expected": value, "actual": actual.get(key)}
                    for key, value in expected.items()
                    if actual.get(key) != value
                }
                result = {
                    "verified": not mismatches,
                    "checked": {
                        key: {
                            "expected": value,
                            "actual": actual.get(key),
                            "match": actual.get(key) == value,
                        }
                        for key, value in expected.items()
                    },
                    "mismatches": mismatches,
                    "unverified": unverified,
                    "unloaded": False,
                    "error": None,
                }
    except Exception as exc:  # noqa: BLE001 - report any failure
        result = {
            "verified": False,
            "checked": {},
            "mismatches": {},
            "unverified": unverified,
            "unloaded": False,
            "error": str(exc),
        }

    finally:
        unloaded = False
        if identifier is not None:
            try:
                await client.unload_model(identifier=identifier)
                unloaded = True
            except Exception:  # noqa: BLE001
                unloaded = False
        if result is not None:
            result["unloaded"] = unloaded

    return result if result is not None else {
        "verified": False,
        "checked": {},
        "mismatches": {},
        "unverified": unverified,
        "unloaded": False,
        "error": "unknown error",
    }
