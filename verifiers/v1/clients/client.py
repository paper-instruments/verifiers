"""Client interfaces for model inference and relay."""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from uuid import uuid4

import httpx

from verifiers.v1.clients.base import join_url
from verifiers.v1.configs.client import (
    BaseClientConfig,
    ClientConfig,
    RegisteredClientConfig,
    TrainClientConfig,
    resolve_api_key,
)
from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, Sampling, SamplingConfig

SESSION_ID_HEADER = "X-Session-ID"
_RELEASE_RETRY_DELAYS = (0.05, 0.1)

logger = logging.getLogger(__name__)

_CLIENT_FACTORIES: dict[str, Callable[[], "Client"]] = {}

"""Per-rollout routing header (the trace id, same value every turn), so a session-affinity
router pins a rollout's turns to one engine and its growing prefix stays KV-cached."""


async def release_router_session(
    client: httpx.AsyncClient,
    config: BaseClientConfig,
    session_id: str,
) -> None:
    """Best-effort release of router state for one completed rollout."""
    if config.session_release_path is None:
        return
    for attempt in range(len(_RELEASE_RETRY_DELAYS) + 1):
        try:
            headers = httpx.Headers(config.headers)
            headers.setdefault("Authorization", f"Bearer {resolve_api_key(config)}")
            headers[SESSION_ID_HEADER] = session_id
            response = await client.delete(
                join_url(config.base_url, config.session_release_path),
                headers=headers,
            )
            response.raise_for_status()
            return
        except Exception:  # Cleanup must never invalidate a rollout.
            if attempt == len(_RELEASE_RETRY_DELAYS):
                logger.warning(
                    "router session release failed after %d attempts (session %s)",
                    attempt + 1,
                    session_id,
                    exc_info=True,
                )
                return
            await asyncio.sleep(_RELEASE_RETRY_DELAYS[attempt])


@dataclass
class RelayReply:
    """A relayed upstream response: content type, complete SSE events, and connection cleanup."""

    content_type: str
    chunks: AsyncIterator[bytes]
    close: Callable[[], Awaitable[None]]


class Client(ABC):
    @abstractmethod
    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        """Run one completion -> a vf `Response`. The eval client forwards the native JSON
        and parses a copy via `dialect`; the train client renders `body` to token ids.
        `session_id` is the rollout's trace id (sent as `SESSION_ID_HEADER`); `turn` is the
        graph-resolved prompt prefix, used by train clients for renderer bridging."""

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        """Stream a (possibly SSE) response back, relaying the provider's bytes — the proxy's
        path for a streaming request. Only the relay (eval) client supports it; the renderer
        generates and cannot stream."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    async def relay_aux(
        self,
        dialect: Dialect,
        route: str,
        body: dict,
        headers: Mapping[str, str] | None = None,
    ) -> dict:
        """Relay a non-model-turn side request (an `aux_route`, e.g. Anthropic's `count_tokens`)
        as native JSON and return the provider JSON. Only the relay (eval) client supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not relay aux routes")

    async def release_session(self, session_id: str) -> None:
        """Release router state for a completed rollout when its client opts in."""

    async def close(self) -> None:
        pass


@contextmanager
def register_client_factory(
    factory: Callable[[], Client],
) -> Iterator[RegisteredClientConfig]:
    """Expose a process-local client factory through a serializable config."""
    config = RegisteredClientConfig(key=uuid4().hex)
    _CLIENT_FACTORIES[config.key] = factory
    try:
        yield config
    finally:
        del _CLIENT_FACTORIES[config.key]


def resolve_client(config: ClientConfig) -> Client:
    if isinstance(config, RegisteredClientConfig):
        try:
            factory = _CLIENT_FACTORIES[config.key]
        except KeyError as exc:
            raise RuntimeError(
                f"No client factory is registered for key {config.key!r} in this process"
            ) from exc
        return factory()
    if isinstance(config, TrainClientConfig):
        from verifiers.v1.clients.train import TrainClient

        return TrainClient(config)
    from verifiers.v1.clients.eval import EvalClient

    return EvalClient(config)


@dataclass(frozen=True)
class ModelContext:
    """Model, endpoint config, and sampling for one rollout."""

    model: str
    client: ClientConfig
    sampling: Sampling = field(default_factory=Sampling)
