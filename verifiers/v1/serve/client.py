"""ZMQ client for the env server.

A DEALER socket + msgpack, with a single receive loop matching responses to
per-request futures by `request_id`. Speaks the typed pydantic request/response
models (`serve/types.py`) end-to-end: a request is `model_dump`ed onto the wire
and the reply is `model_validate`d back — `Trace`s come back typed as
`Trace[WireTaskData]` (non-strict task, so env fields survive without importing the
env). Health is just another request (no dedicated probe thread).
"""

import asyncio
import contextlib
import logging
import time
import uuid
from typing import TypeVar

import msgpack
import zmq
import zmq.asyncio

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.episode import WireEpisode
from verifiers.v1.serve.types import (
    BaseRequest,
    BaseResponse,
    CancelRequest,
    CancelResponse,
    HealthRequest,
    HealthResponse,
    InfoRequest,
    InfoResponse,
    RunGroupRequest,
    RunGroupResponse,
    RunRequest,
    RunResponse,
)
from verifiers.v1.trace import WireTrace
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseResponse)
_REMOTE_CANCEL_TIMEOUT = 30.0
"""Most seconds cancellation waits for the server to finish remote cleanup."""


class EnvClient:
    def __init__(self, address: str = "tcp://127.0.0.1:5000") -> None:
        self.address = address
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.connect(address)
        self._pending: dict[str, asyncio.Future[bytes]] = {}
        self._receiver: asyncio.Task | None = None
        self._receiver_error: ConnectionError | None = None
        self._decode_slots = asyncio.BoundedSemaphore(1)
        self._closed = False

    def _ensure_receiver(self) -> None:
        if self._closed:
            raise RuntimeError("env client is closed")
        if self._receiver_error is not None:
            raise self._receiver_error
        if self._receiver is None:
            self._receiver = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        try:
            while True:
                request_id_bytes, data = await self.socket.recv_multipart()
                future = self._pending.pop(request_id_bytes.decode(), None)
                if future is not None and not future.done():
                    future.set_result(data)
        except asyncio.CancelledError:
            if not self._closed:
                self._fail_pending(
                    ConnectionError("env server response receiver stopped unexpectedly")
                )
        except Exception as error:
            failure = ConnectionError(
                f"env server response receiver failed: {type(error).__name__}: {error}"
            )
            self._fail_pending(failure)
            logger.warning("%s", failure, exc_info=True)

    def _fail_pending(self, error: ConnectionError) -> None:
        self._receiver_error = error
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)

    async def _request(
        self,
        request: BaseRequest,
        response_type: type[ResponseT],
        timeout: float | None = None,
        cancel_remote: bool = False,
    ) -> ResponseT:
        """Send a typed request and validate the reply into `response_type`. A
        `timeout` is only used for health polling — rollouts run untimed."""
        self._ensure_receiver()
        request_id = uuid.uuid4().hex
        future: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            payload = msgpack.packb(request.model_dump(mode="json"), use_bin_type=True)
            await self.socket.send_multipart(
                [request_id.encode(), request.method.encode(), payload]
            )
            data = await asyncio.wait_for(future, timeout)
        except TimeoutError:
            self._discard_pending(request_id, future)
            raise
        except asyncio.CancelledError:
            self._discard_pending(request_id, future)
            if cancel_remote:
                try:
                    await self._wait_for_remote_cancel(request_id)
                except (Exception, asyncio.CancelledError):
                    logger.warning(
                        "remote cancellation acknowledgement failed for request %s",
                        request_id,
                        exc_info=True,
                    )
            raise
        except BaseException:
            self._discard_pending(request_id, future)
            raise
        if response_type in (CancelResponse, HealthResponse, InfoResponse):
            response = response_type.model_validate(msgpack.unpackb(data, raw=False))
        else:
            # Keep large trace replies compact on the loop and expand only one at a time.
            await self._decode_slots.acquire()
            decoding = asyncio.create_task(
                asyncio.to_thread(
                    lambda: response_type.model_validate(
                        msgpack.unpackb(data, raw=False)
                    )
                )
            )
            # Hold the slot until the worker finishes so cancellation cannot overlap decodes.
            decoding.add_done_callback(lambda _: self._decode_slots.release())
            try:
                response = await asyncio.shield(decoding)
            except asyncio.CancelledError:
                decoding.add_done_callback(
                    lambda task: None if task.cancelled() else task.exception()
                )
                raise
        if not response.success:
            raise RuntimeError(response.error or "env server request failed")
        return response

    def _discard_pending(self, request_id: str, future: asyncio.Future[bytes]) -> None:
        if self._pending.get(request_id) is future:
            del self._pending[request_id]
        if not future.done():
            future.cancel()

    async def _wait_for_remote_cancel(self, request_id: str) -> None:
        cancellation = asyncio.create_task(self.cancel(request_id))
        deadline = asyncio.get_running_loop().time() + _REMOTE_CANCEL_TIMEOUT
        try:
            while not cancellation.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        "remote cancellation acknowledgement exceeded "
                        f"{_REMOTE_CANCEL_TIMEOUT:g}s"
                    )
                try:
                    await asyncio.wait_for(
                        asyncio.shield(cancellation), timeout=remaining
                    )
                except asyncio.CancelledError:
                    continue
            cancellation.result()
        finally:
            if not cancellation.done():
                cancellation.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await cancellation

    async def health(self, timeout: float = 2.0) -> bool:
        try:
            return (
                await self._request(HealthRequest(), HealthResponse, timeout=timeout)
            ).success
        except TimeoutError:
            return False

    async def wait_for_server_startup(
        self, timeout: float = 120.0, interval: float = 1.0
    ) -> None:
        """Poll `health` until the server answers or `timeout` elapses."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await self.health(timeout=min(interval, 2.0)):
                return
            await asyncio.sleep(interval)
        raise TimeoutError(
            f"env server at {self.address} did not become healthy in {timeout}s"
        )

    async def info(self) -> InfoResponse:
        """Return the taskset `num_tasks` + whether its tasks group-score (legacy v0 only)."""
        return await self._request(InfoRequest(), InfoResponse)

    async def cancel(self, target_request_id: str) -> CancelResponse:
        """Cancel one request and return after its remote cleanup has finished."""
        return await self._request(
            CancelRequest(target_request_id=target_request_id), CancelResponse
        )

    async def run(
        self,
        client: ClientConfig,
        model: str,
        sampling: SamplingConfig,
        task_data: dict | None = None,
        # TODO: remove task_idx addressing once v0 (the legacy bridge) is deprecated.
        task_idx: int | None = None,
    ) -> WireEpisode:
        """Run one rollout; return its episode record — flat traces (typed
        `Trace[WireTaskData]`) plus the shared stamp. A v1 server takes the task
        itself (`task_data`, its dumped `TaskData`); the legacy bridge addresses
        its server-side dataset by `task_idx`."""
        response = await self._request(
            RunRequest(
                task_data=task_data,
                task_idx=task_idx,
                client=client,
                model=model,
                sampling=sampling,
            ),
            RunResponse,
            cancel_remote=True,
        )
        return response.episode

    async def run_group(
        self,
        task_idx: int,
        n: int,
        client: ClientConfig,
        model: str,
        sampling: SamplingConfig,
    ) -> list[WireTrace]:
        """Run `n` rollouts for `task_idx` as a scored group — the legacy (v0) route;
        a v1 server refuses it. Returns typed `WireTrace`s."""
        response = await self._request(
            RunGroupRequest(
                task_idx=task_idx, n=n, client=client, model=model, sampling=sampling
            ),
            RunGroupResponse,
            cancel_remote=True,
        )
        return response.traces

    async def close(self) -> None:
        """Close an idle client.

        Cancel and await in-flight rollout requests first; their request-level
        cancellation handshake is what tears down remote work.
        """
        if self._closed:
            return
        self._closed = True
        if self._receiver is not None:
            self._receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receiver
            self._receiver = None
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(ConnectionError("env client closed"))
        self.socket.close()
        self.ctx.term()
