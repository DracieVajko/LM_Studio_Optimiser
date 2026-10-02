"""BackendClient seam + conformance helper.

Canonical contract (plan §Task 1): ``backend_name``, ``connect()``,
``list_models()``, ``load_model(model_id, config)``,
``generate(model_id, prompt, options)``, ``unload(model_id)``,
``capabilities()``, ``close()``.

Reality adaptations (LMStudioClient predates the seam and is async):
- All I/O methods are ``async``; the Protocol reflects that.
- ``load_model`` accepts the second parameter under either name
  (``config`` per contract, ``load_config`` per LMStudioClient); callers
  should pass it positionally.
- ``capabilities`` MAY be exposed as a read-only property (as
  LMStudioClient does) instead of a method. ``assert_conforms`` uses
  ``inspect.getattr_static`` so it never invokes the property getter
  (which raises when unconnected) — presence is what is checked.
- ``generate`` / ``unload`` are thin shims on LMStudioClient over
  ``chat_completion`` / ``ensure_unloaded`` (fail-closed); see
  ``lm_optimizer/services/lm_studio.py``.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BackendClient(Protocol):
    """Structural contract every backend client must satisfy."""

    backend_name: str

    async def connect(self, *args: Any, **kwargs: Any) -> None:
        """Establish connection and detect capabilities."""
        ...

    async def list_models(self, *args: Any, **kwargs: Any) -> list:
        """List available models."""
        ...

    async def load_model(self, model_id: str, config: Any, *args: Any, **kwargs: Any) -> Any:
        """Load a model with the given configuration."""
        ...

    async def generate(
        self,
        model_id: str,
        prompt: str,
        options: dict | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Generate a completion for a prompt."""
        ...

    async def unload(self, model_id: str, *args: Any, **kwargs: Any) -> bool:
        """Unload a model (fail-closed: verified gone)."""
        ...

    def capabilities(self, *args: Any, **kwargs: Any) -> Any:
        """Return backend capabilities.

        Implementations MAY expose this as a read-only property instead
        of a method (LMStudioClient does); both satisfy conformance.
        """
        ...

    async def close(self) -> None:
        """Close the client connection."""
        ...


_ASYNC_METHODS = ("connect", "list_models", "load_model", "generate", "unload", "close")

_LOAD_CONFIG_NAMES = frozenset({"config", "load_config", "load_configuration", "cfg"})
_GENERATE_OPTIONS_NAMES = frozenset({"options", "opts", "config", "params", "generation_kwargs"})


def _require_async_method(client: Any, name: str) -> None:
    try:
        static = inspect.getattr_static(client, name)
    except AttributeError:
        raise AssertionError(f"BackendClient non-conformance: missing method {name!r}")
    if isinstance(static, property):
        raise AssertionError(
            f"BackendClient non-conformance: {name!r} must be an async method, found property"
        )
    attr = getattr(client, name)
    if not callable(attr):
        raise AssertionError(f"BackendClient non-conformance: {name!r} is not callable")
    if not inspect.iscoroutinefunction(attr):
        raise AssertionError(f"BackendClient non-conformance: {name!r} must be async")


def assert_conforms(client: Any) -> None:
    """Assert that ``client`` structurally satisfies BackendClient.

    Checks presence/types only — never performs I/O and never invokes
    ``capabilities`` (the LMStudio property getter raises when
    unconnected). Raises AssertionError with a diagnostic otherwise.
    """
    # backend_name: non-empty str (class attr or instance attr; never invoke).
    try:
        raw_name = inspect.getattr_static(client, "backend_name")
    except AttributeError:
        raise AssertionError("BackendClient non-conformance: missing 'backend_name'")
    if isinstance(raw_name, property):
        try:
            raw_name = raw_name.fget(client)  # type: ignore[misc]
        except Exception as e:
            raise AssertionError(f"BackendClient non-conformance: 'backend_name' unreadable: {e}")
    if not isinstance(raw_name, str) or not raw_name:
        raise AssertionError(
            f"BackendClient non-conformance: 'backend_name' must be a non-empty str, "
            f"got {raw_name!r}"
        )

    for name in _ASYNC_METHODS:
        _require_async_method(client, name)

    # load_model(model_id, config): second positional param under either name.
    # NOTE: getattr returns the BOUND method, so 'self' is already excluded.
    try:
        params = list(inspect.signature(getattr(client, "load_model")).parameters)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"BackendClient non-conformance: 'load_model' unsignable: {e}")
    if len(params) < 2 or params[1] not in _LOAD_CONFIG_NAMES:
        raise AssertionError(
            "BackendClient non-conformance: 'load_model' must accept "
            f"(model_id, config) (second-param aliases: {sorted(_LOAD_CONFIG_NAMES)}), "
            f"got params {params}"
        )

    # generate(model_id, prompt, options). Bound method: 'self' excluded.
    try:
        params = list(inspect.signature(getattr(client, "generate")).parameters)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"BackendClient non-conformance: 'generate' unsignable: {e}")
    if len(params) < 3 or params[2] not in _GENERATE_OPTIONS_NAMES:
        raise AssertionError(
            "BackendClient non-conformance: 'generate' must accept "
            f"(model_id, prompt, options) (third-param aliases: "
            f"{sorted(_GENERATE_OPTIONS_NAMES)}), got params {params}"
        )

    # unload(model_id). Bound method: 'self' excluded.
    try:
        params = list(inspect.signature(getattr(client, "unload")).parameters)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"BackendClient non-conformance: 'unload' unsignable: {e}")
    if len(params) < 1:
        raise AssertionError(
            f"BackendClient non-conformance: 'unload' must accept (model_id), got {params}"
        )

    # capabilities: property OR sync/async method — presence only, never invoke.
    try:
        inspect.getattr_static(client, "capabilities")
    except AttributeError:
        raise AssertionError("BackendClient non-conformance: missing 'capabilities'")

    if not isinstance(client, BackendClient):
        raise AssertionError("BackendClient non-conformance: isinstance(client, BackendClient) is False")
