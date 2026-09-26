"""LM Studio service with capability discovery.

Live-verified contract (2026-09-19, LM Studio bundled server, `lms` commit 69d945a):
- GET /api/v1/models -> {"models": [{key, display_name, architecture,
  quantization{name}, size_bytes, max_context_length, loaded_instances:[...], ...}]}
- POST /api/v1/models/load takes load parameters FLAT (no config/identifier
  wrapper). Unknown key -> HTTP 400 `unrecognized_keys` WITHOUT loading the
  model (safe probing). Success -> {"status": "loaded", "instance_id": ...}.
  Full applied config is echoed in the model's `loaded_instances[].config`.
- POST /api/v1/models/unload takes {"instance_id": ...} -> 200 {"instance_id"}.
- POST /api/v1/chat is the native endpoint: {"model", "input": str, ...}.
  /api/v1/chat/completions does NOT exist in this version.
"""

import time
import uuid
from dataclasses import dataclass

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lm_optimizer.config import config
from lm_optimizer.domain.models import (
    LMStudioCapabilities,
    LoadConfiguration,
    ModelIdentity,
)
from lm_optimizer.logging_config import get_logger

logger = get_logger(__name__)

# Fake model used for safe capability probing: key validation runs before model
# resolution, so unknown keys 400 without loading anything, known keys 404.
_PROBE_MODEL = "lms-probe-nonexistent-model"

# Slow-model allowances: big loads/generations legitimately take minutes.
LOAD_TIMEOUT_S = 600.0
CHAT_TIMEOUT_FLOOR_S = 300.0

# Candidate load keys with type-correct probe values (reasoning_budget_message
# is a string; numbers are rejected with invalid_type, not unrecognized_keys).
_PROBE_VALUES: dict[str, object] = {
    "context_length": 512,
    "flash_attention": True,
    "offload_kv_cache_to_gpu": True,
    "eval_batch_size": 256,
    "physical_batch_size": 512,
    "parallel": 2,
    "context_checkpoints": 16,
    "reasoning_budget_message": "",
    "num_experts": 8,
    "speculative_draft_max_tokens": 8,
    "gpu_ratio": 0.5,
    "rope_freq_base": 10000.0,
    "rope_freq_scale": 1.0,
}


@dataclass
class LoadModelResult:
    """Result of a load request."""

    success: bool
    identifier: str | None = None
    error: str | None = None
    loaded_config: LoadConfiguration | None = None
    echo_mismatches: dict | None = None


def _error_message(exc: httpx.HTTPStatusError) -> str:
    """Extract server error message from an HTTP error response."""
    try:
        data = exc.response.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message", err))
        return str(err)
    except Exception:
        return f"HTTP {exc.response.status_code}"


def _is_unrecognized_key_error(exc: httpx.HTTPStatusError, key: str) -> bool:
    """True when the server 400s on an unrecognized key naming `key`."""
    if exc.response.status_code != 400:
        return False
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == "unrecognized_keys" and key in str(err.get("message", ""))
    except Exception:
        return False


def _is_no_reasoning_error(exc: httpx.HTTPStatusError) -> bool:
    """True when the model exposes no reasoning config at all.

    Only this shape poisons the remember-set; bad reasoning VALUES on
    supporting models (allowed sets vary: off/on vs full range) retry
    without remembering.
    """
    if exc.response.status_code != 400:
        return False
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == "invalid_value" and "does not expose reasoning" in str(
            err.get("message", "")
        )
    except Exception:
        return False


def _is_reasoning_error(exc: httpx.HTTPStatusError) -> bool:
    """True for any reasoning-key 400 (no-config AND bad-value shapes)."""
    if exc.response.status_code != 400:
        return False
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == "invalid_value" and "reasoning" in str(
            err.get("message", "")
        ).lower()
    except Exception:
        return False


class LMStudioClient:
    """LM Studio API client with capability discovery."""

    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base_url = (base_url or config.lm_studio.base_url).rstrip("/")
        self.timeout = timeout or config.lm_studio.timeout
        self._client: httpx.AsyncClient | None = None
        self._capabilities: LMStudioCapabilities | None = None
        self._models_cache: list[ModelIdentity] = []
        self._loaded_models: dict[str, str] = {}  # model_id -> instance_id
        self._no_reasoning_models: set[str] = set()
        # Exact rejected reasoning VALUES per model (sets vary: off/on vs
        # full range). Skips only values already proven rejected server-side.
        self._bad_reasoning_values: dict[str, set[str]] = {}
        self._echo_probed: bool = False

    async def __aenter__(self) -> "LMStudioClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def connect(self, echo_probe: bool = True) -> None:
        """Establish connection and detect capabilities.

        The safe key-probing phase never loads anything. The echo phase loads
        the smallest model once (with unload in `finally`) to verify applied
        configs; pass echo_probe=False for load-free checks (status/models).
        """
        if self._client is not None:
            # Upgrade a load-free connection to a full one when requested.
            if echo_probe and not getattr(self, "_echo_probed", False):
                self._skip_echo_probe = False
                await self._probe_echo_phase()
            return
        self._skip_echo_probe = not echo_probe

        from urllib.parse import urlparse

        host = (urlparse(self.base_url).hostname or "").lower()
        if host not in ("127.0.0.1", "localhost", "::1", ""):
            logger.warning(
                "Remote LM Studio: local hardware must NOT score the remote model. "
                "Set HW_ overrides or treat hardware as Unknown.",
                base_url=self.base_url,
            )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

        await self._detect_capabilities()
        await self.list_models(force_refresh=True)
        logger.info(
            "Connected to LM Studio", base_url=self.base_url, version=self._capabilities.version
        )

    async def close(self) -> None:
        """Close the client connection."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client."""
        if self._client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

    @property
    def capabilities(self) -> LMStudioCapabilities:
        """Get detected API capabilities."""
        if self._capabilities is None:
            raise RuntimeError("Capabilities not detected. Call connect() first.")
        return self._capabilities

    async def _detect_capabilities(self) -> None:
        """Detect LM Studio API version and capabilities."""
        try:
            response = await self._request_with_retry("GET", "/api/v1/models")
            response.json()  # validates shape; parse happens in list_models
            self._capabilities = LMStudioCapabilities(version="v1")
            await self._probe_load_parameters()
        except Exception as e:
            logger.warning("Failed to detect capabilities, using defaults", error=str(e))
            self._capabilities = LMStudioCapabilities(version="v1")

    async def _probe_key(self, key: str, value: object) -> bool:
        """Safe-probe one load key against a fake model. True = accepted.

        Unknown keys 400 before model resolution (nothing loads); known keys
        404 model_not_found.
        """
        try:
            await self._request_with_retry(
                "POST", "/api/v1/models/load", json_data={"model": _PROBE_MODEL, key: value}
            )
            return True  # unexpected 200 with fake model; treat as accepted
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                try:
                    code = e.response.json().get("error", {}).get("code", "")
                except Exception:
                    code = ""
                return code != "unrecognized_keys"
            return True  # 404 model_not_found (or similar) = key recognized
        except Exception:
            return False

    def _apply_accepted(self, accepted: set[str]) -> None:
        """Set capability flags from the probed accepted-key set."""
        caps = self._capabilities
        caps.load_parameters = sorted(accepted)
        caps.supports_context_length = "context_length" in accepted
        caps.supports_gpu_ratio = "gpu_ratio" in accepted
        caps.supports_flash_attention = "flash_attention" in accepted
        caps.supports_kv_cache_placement = "offload_kv_cache_to_gpu" in accepted
        caps.supports_eval_batch_size = "eval_batch_size" in accepted
        caps.supports_physical_batch_size = "physical_batch_size" in accepted
        caps.supports_parallel = "parallel" in accepted
        caps.supports_context_checkpoints = "context_checkpoints" in accepted
        caps.supports_num_experts = "num_experts" in accepted
        caps.supports_rope_scaling = "rope_freq_base" in accepted

    async def _probe_load_parameters(self) -> None:
        """Probe which load parameters are supported.

        Safe phase (no loads): per-key 400-probing against a fake model.
        Echo phase: one real load of the smallest LLM with unload in `finally`.
        """
        if not self._capabilities:
            return

        accepted = {
            key for key, value in _PROBE_VALUES.items() if await self._probe_key(key, value)
        }
        self._apply_accepted(accepted)

        if getattr(self, "_skip_echo_probe", False):
            logger.debug("Echo probe skipped (load-free connect)")
            return

        await self._probe_echo_phase()

    async def _probe_echo_phase(self) -> None:
        """One real load of the smallest LLM with unload in `finally`."""
        # Echo phase with the smallest LLM (never the first/biggest in the list).
        smallest = None
        identifier = None
        try:
            models = await self.list_models()
            candidates = [
                m
                for m in models
                if (m.size_bytes or float("inf")) < float("inf")
                and "embed" not in (m.id or "").lower()
            ]
            if not candidates:
                return
            smallest = min(candidates, key=lambda m: m.size_bytes or 0)
            probe_ctx = min(512, smallest.context_limit or 512)
            result = await self.load_model(smallest.id, LoadConfiguration(context_length=probe_ctx))
            if result.success:
                identifier = result.identifier
                echo = result.loaded_config.to_dict() if result.loaded_config else {}
                logger.info("Probe echo", model=smallest.id, echo=echo)
        except Exception as e:
            logger.warning("Probe echo load failed", error=str(e))
        finally:
            self._echo_probed = True
            # Unload by captured identifier (cache-independent); fall back to model id.
            if identifier:
                try:
                    await self.unload_model(identifier=identifier)
                except Exception:
                    pass
            elif smallest is not None:
                try:
                    await self.unload_model(model_id=smallest.id)
                except Exception:
                    pass

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json_data: dict | None = None,
        params: dict | None = None,
        timeout_s: float | None = None,
    ) -> httpx.Response:
        """Make HTTP request with retry logic.

        timeout_s overrides the default client timeout for slow operations
        (big-model loads/generations legitimately take minutes).
        """
        retry_config = AsyncRetrying(
            stop=stop_after_attempt(config.lm_studio.max_retries),
            wait=wait_exponential(multiplier=config.lm_studio.retry_delay, min=1, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)
            ),
            reraise=True,
        )

        request_timeout = httpx.Timeout(timeout_s) if timeout_s else None
        async for attempt in retry_config:
            with attempt:
                kwargs: dict = {}
                if request_timeout is not None:
                    kwargs["timeout"] = request_timeout
                response = await self.client.request(
                    method, path, json=json_data, params=params, **kwargs
                )
                response.raise_for_status()
                return response

        raise RuntimeError("Retry logic failed unexpectedly")

    async def list_models(self, force_refresh: bool = False) -> list[ModelIdentity]:
        """List all available models."""
        if self._models_cache and not force_refresh:
            return self._models_cache

        response = await self._request_with_retry("GET", "/api/v1/models")
        data = response.json()

        models = []
        items = data.get("models", []) if isinstance(data, dict) else []

        for item in items:
            quant = item.get("quantization") or {}
            model = ModelIdentity(
                id=item.get("key", ""),
                name=item.get("display_name") or item.get("key", ""),
                architecture=item.get("architecture"),
                parameter_count=item.get("parameter_count"),
                quantization=quant.get("name") if isinstance(quant, dict) else quant,
                context_limit=item.get("max_context_length"),
                is_moe=self._detect_moe(item),
                num_experts=item.get("num_experts"),
                size_bytes=item.get("size_bytes"),
            )
            models.append(model)

        self._models_cache = models
        return models

    async def get_loaded_instances(self) -> list[dict]:
        """List currently loaded instances across all models (server state)."""
        response = await self._request_with_retry("GET", "/api/v1/models")
        data = response.json()
        items = data.get("models", []) if isinstance(data, dict) else []
        instances = []
        for item in items:
            for inst in item.get("loaded_instances") or []:
                instances.append(
                    {
                        "instance_id": inst.get("id"),
                        "model": item.get("key", ""),
                        "config": inst.get("config") or {},
                    }
                )
        # Sync client-side cache with server state
        self._loaded_models = {
            inst["model"]: inst["instance_id"]
            for inst in instances
            if inst["model"] and inst["instance_id"]
        }
        return instances

    def _detect_moe(self, item: dict) -> bool:
        """Detect if model is Mixture of Experts."""
        arch = (item.get("architecture") or "").lower()
        model_type = (item.get("type") or "").lower()
        name = (item.get("display_name") or item.get("key") or "").lower()

        moe_indicators = [
            "moe",
            "mixtral",
            "grok",
            "deepseek-moe",
            "qwen-moe",
            "phi-moe",
            "gpt-oss",
            "gpt_oss",
        ]
        return any(
            indicator in arch or indicator in model_type or indicator in name
            for indicator in moe_indicators
        )

    async def get_model(self, model_id: str) -> ModelIdentity | None:
        """Get specific model information."""
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        return None

    async def _read_echo_config(self, instance_id: str) -> LoadConfiguration | None:
        """Read the server-echoed applied config for an instance (best effort)."""
        try:
            instances = await self.get_loaded_instances()
            for inst in instances:
                if inst["instance_id"] == instance_id:
                    echo = inst.get("config") or {}
                    fields = LoadConfiguration.__dataclass_fields__
                    return LoadConfiguration(**{k: v for k, v in echo.items() if k in fields})
        except Exception as e:
            logger.debug("Echo read failed", error=str(e))
        return None

    async def load_model(
        self,
        model_id: str,
        load_config: LoadConfiguration,
    ) -> LoadModelResult:
        """Load a model with specified configuration (flat parameters).

        Sends echo_load_config=true and verifies REQUESTED vs APPLIED:
        mismatches are recorded (never silently accepted).
        """
        api_params = load_config.to_api_params()
        body = {"model": model_id, **api_params, "echo_load_config": True}

        try:
            response = await self._request_with_retry(
                "POST", "/api/v1/models/load", json_data=body, timeout_s=LOAD_TIMEOUT_S
            )
        except httpx.HTTPStatusError as e:
            # Older servers may not know echo_load_config: retry without it
            # (400s happen before any load, so this is safe).
            if _is_unrecognized_key_error(e, "echo_load_config"):
                logger.debug("Server lacks echo_load_config; retrying without", model=model_id)
                body = {"model": model_id, **api_params}
                try:
                    response = await self._request_with_retry(
                        "POST", "/api/v1/models/load", json_data=body,
                        timeout_s=LOAD_TIMEOUT_S,
                    )
                except httpx.HTTPStatusError as e2:
                    return LoadModelResult(success=False, error=_error_message(e2))
            else:
                return LoadModelResult(success=False, error=_error_message(e))

        result_data = response.json()
        identifier = result_data.get("instance_id")
        if result_data.get("status") == "loaded" and identifier:
            self._loaded_models[model_id] = identifier
            loaded_config = self._config_from_echo(result_data.get("load_config"))
            if loaded_config is None:
                loaded_config = await self._read_echo_config(identifier)
            mismatches = self._compare_echo(api_params, loaded_config)
            if mismatches:
                logger.warning(
                    "Requested vs applied mismatch", model=model_id, mismatches=mismatches
                )
            return LoadModelResult(
                success=True,
                identifier=identifier,
                loaded_config=loaded_config,
                echo_mismatches=mismatches or None,
            )
        return LoadModelResult(
            success=False, error=result_data.get("error") or f"Unexpected response: {result_data}"
        )

    @staticmethod
    def _config_from_echo(echo: dict | None) -> LoadConfiguration | None:
        """Build LoadConfiguration from an echo dict (load response or instances)."""
        if not isinstance(echo, dict) or not echo:
            return None
        fields = LoadConfiguration.__dataclass_fields__
        return LoadConfiguration(**{k: v for k, v in echo.items() if k in fields})

    @staticmethod
    def _compare_echo(requested: dict, applied: LoadConfiguration | None) -> dict:
        """REQUESTED vs APPLIED diff on keys we sent (empty = exact match)."""
        if applied is None:
            return {}
        actual = applied.to_dict()
        return {
            k: {"requested": v, "applied": actual.get(k)}
            for k, v in requested.items()
            if actual.get(k) != v
        }

    async def unload_model(
        self, model_id: str | None = None, identifier: str | None = None
    ) -> bool:
        """Unload a model by instance id."""
        if identifier is None and model_id is not None:
            identifier = self._loaded_models.get(model_id)

        if identifier is None:
            return False

        try:
            await self._request_with_retry(
                "POST", "/api/v1/models/unload", json_data={"instance_id": identifier}
            )
        except httpx.HTTPStatusError as e:
            logger.warning("Unload failed", identifier=identifier, error=_error_message(e))
            return False
        except Exception:
            return False

        if model_id and model_id in self._loaded_models:
            del self._loaded_models[model_id]
        else:
            # Purge any stale cache entry pointing at this instance
            for mid, iid in list(self._loaded_models.items()):
                if iid == identifier:
                    del self._loaded_models[mid]
        return True

    async def unload_all(self) -> dict[str, bool]:
        """Unload all currently loaded models (server state wins)."""
        results: dict[str, bool] = {}
        try:
            instances = await self.get_loaded_instances()
        except Exception:
            instances = []
        for inst in instances:
            if inst["instance_id"]:
                results[inst["model"]] = await self.unload_model(identifier=inst["instance_id"])
        for model_id, identifier in list(self._loaded_models.items()):
            if model_id not in results:
                results[model_id] = await self.unload_model(identifier=identifier)
        return results

    async def ensure_unloaded(self, model_id: str) -> bool:
        """Unload a model and verify it is really gone (server state).

        Retries once by instance id: covers stale client cache and transient
        unload failures. Returns True only when no instance of the model
        remains loaded.
        """
        for _ in range(2):
            try:
                await self.unload_model(model_id=model_id)
            except Exception as e:
                logger.warning("Unload attempt failed", model=model_id, error=str(e))
            try:
                instances = await self.get_loaded_instances()
            except Exception as e:
                logger.warning("State check failed", model=model_id, error=str(e))
                continue
            leftovers = [i for i in instances if i["model"] == model_id and i["instance_id"]]
            if not leftovers:
                self._loaded_models.pop(model_id, None)
                return True
            for inst in leftovers:
                try:
                    await self.unload_model(identifier=inst["instance_id"])
                except Exception as e:
                    logger.warning(
                        "Forced unload failed",
                        instance=inst["instance_id"],
                        error=str(e),
                    )
        try:
            instances = await self.get_loaded_instances()
        except Exception:
            return False
        left = [i for i in instances if i["model"] == model_id]
        if left:
            logger.warning("Model still loaded after ensure_unloaded", model=model_id)
            return False
        self._loaded_models.pop(model_id, None)
        return True

    async def chat_completion(
        self,
        model: str,
        input_text: str,
        temperature: float = 0.7,
        max_output_tokens: int = 512,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
        presence_penalty: float | None = None,
        reasoning: str | None = None,
    ) -> dict:
        """Generate completion via the native /api/v1/chat endpoint.

        Returns an OpenAI-shaped dict (choices[0].message.content + usage)
        converted from the native output[]/stats response, plus "_stats"
        with the raw native stats (real TTFT, tok/s, load time).
        Non-reasoning models reject the reasoning key: retry once without it
        and remember the verdict per model. Exact rejected VALUES are also
        remembered per model so later calls skip the doomed first attempt.
        """
        if reasoning is not None and reasoning in self._bad_reasoning_values.get(model, ()):
            logger.debug("Skipping known-rejected reasoning value", model=model)
            reasoning = None
        use_reasoning = reasoning if model not in self._no_reasoning_models else None
        body: dict = {
            "model": model,
            "input": input_text,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        for key, value in (
            ("top_p", top_p),
            ("top_k", top_k),
            ("min_p", min_p),
            ("presence_penalty", presence_penalty),
            ("reasoning", use_reasoning),
        ):
            if value is not None:
                body[key] = value

        chat_timeout = max(CHAT_TIMEOUT_FLOOR_S, float(max_output_tokens))
        try:
            response = await self._request_with_retry(
                "POST", "/api/v1/chat", json_data=body, timeout_s=chat_timeout
            )
        except httpx.HTTPStatusError as e:
            if use_reasoning is not None and _is_reasoning_error(e):
                if _is_no_reasoning_error(e):
                    logger.info("Model has no reasoning config; retrying without", model=model)
                    self._no_reasoning_models.add(model)
                else:
                    logger.info("Reasoning value rejected; retrying without", model=model)
                    self._bad_reasoning_values.setdefault(model, set()).add(use_reasoning)
                body.pop("reasoning", None)
                response = await self._request_with_retry(
                    "POST", "/api/v1/chat", json_data=body, timeout_s=chat_timeout
                )
            else:
                raise
        data = response.json()

        parts = data.get("output") or []
        texts = [p.get("content", "") for p in parts if p.get("type") == "message"]
        if not texts:
            texts = [p.get("content", "") for p in parts if p.get("content")]
        text = "\n".join(texts)

        stats = data.get("stats") or {}
        prompt_tokens = int(stats.get("input_tokens", 0))
        completion_tokens = int(stats.get("total_output_tokens", 0))
        return {
            "id": data.get("response_id", f"resp_{uuid.uuid4().hex}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": None,
            "_stats": stats,
        }

    async def health_check(self) -> bool:
        """Check if LM Studio is responsive."""
        try:
            await self._request_with_retry("GET", "/api/v1/models")
            return True
        except Exception:
            return False

    def get_loaded_model(self, model_id: str) -> str | None:
        """Get loaded model identifier."""
        return self._loaded_models.get(model_id)

    def model_known_no_reasoning(self, model_id: str) -> bool:
        """True when this model already rejected the reasoning key."""
        return model_id in self._no_reasoning_models


async def create_client(
    base_url: str | None = None, echo_probe: bool = True
) -> LMStudioClient:
    """Factory function to create and connect a client."""
    client = LMStudioClient(base_url=base_url)
    await client.connect(echo_probe=echo_probe)
    return client
