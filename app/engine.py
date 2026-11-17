"""Single-process micro-batch inference engine.

Concurrency model
-----------------
* Everything except the blocking model forward passes runs on the FastAPI
  asyncio event loop, so no locks are needed for the shared state below.
* Exactly one worker task forms batches and exactly one executor thread runs
  the TorchScript model, therefore at most ONE batch is ever actually
  inferring at any moment.
* Capacity accounting: an order occupies a slot while it is queued AND while
  its batch is computing (even if it was cancelled or expired mid-flight).
  The slot is released only when the batch compute actually finishes.

State machine per order
-----------------------
queued -> processing -> completed
queued -> cancelled            (cancelled before entering a batch; never runs)
queued -> expired              (deadline passed while queued; never runs)
processing -> cancelled        (marked immediately, result discarded, slot held)
processing -> expired          (deadline passed mid-flight; result discarded)
processing -> failed           (model raised; defensive only)
Late results are written back ONLY to orders still in `processing` whose
deadline has not passed, so a late result can never revive a cancelled or
expired order.

Occlusion explanations
----------------------
An explanation request is only accepted for a COMPLETED order. It reuses the
order's stored 64x64 tensor, masks the 16 cells of a 4x4 grid one at a time
(cell pixels set to the normalized value 0, i.e. the raw 127/128 grey level),
and computes `p_orig[predicted] - p_masked[predicted]` per cell. Negative
values are kept as-is. The explanation runs on the SAME single executor
thread as frame inference ("one compute channel"), so it never overlaps a
frame batch. To keep frames responsive, a round computes AT MOST 4 occlusion
images; between rounds the worker drains all queued frame orders first
(frame priority). The per-order 16-cell work therefore spans 4 rounds.

Explanations are tracked separately from order capacity (they are not images
waiting to be classified) in their own FIFO with a cap of `max_capacity`
pending explanations; a full explanation queue returns None (HTTP 429).
`GET /orders/{id}/explanation` reports `queued | processing | completed |
failed`. A heatmap is published only when all 16 cells succeeded; on any
error the job is marked `failed`, no partial heatmap is exposed, and the
original order/result is never touched. Explanations have no deadline and
cannot be used to cancel or otherwise mutate the order.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import torch

log = logging.getLogger("batch-engine")

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
CANCELLED = "cancelled"
EXPIRED = "expired"
FAILED = "failed"

# Occlusion explanation constants: 4x4 grid over the 64x64 input, masked cells
# get the normalized pixel value 0 (the midpoint of [-1, 1], i.e. raw ~127/128).
GRID = 4
CELL = 64 // GRID  # 16 px per cell
OCCLUSION_VALUE = 0.0
OCCLUSION_PER_ROUND = 4  # at most 4 occlusion images per compute round


# eq=False: orders are compared by identity. The default dataclass __eq__
# would compare tensors element-wise and raise on multi-element tensors.
@dataclass(eq=False)
class Order:
    id: str
    tensor: torch.Tensor
    deadline_ms: int
    deadline_at: float  # time.monotonic() at which the order expires
    state: str = QUEUED
    result: Optional[dict] = None
    batch_id: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)


@dataclass(eq=False)
class OcclusionJob:
    """Explanation job for one completed order (shares the order id)."""

    order_id: str
    tensor: torch.Tensor
    predicted_class: int
    original_prob: float
    state: str = QUEUED
    heatmap: Optional[list[list[float]]] = None  # 4x4 deltas, published atomically
    error: Optional[str] = None
    cells_done: int = 0
    created_at: float = field(default_factory=time.monotonic)
    partial: list[float] = field(default_factory=list, repr=False)  # internal only


class BatchEngine:
    def __init__(
        self,
        model_path: str,
        max_capacity: int = 16,
        batch_size: int = 4,
        batch_wait_ms: int = 50,
        inference_delay_ms: int = 0,
    ) -> None:
        self.max_capacity = max_capacity
        self.batch_size = batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.inference_delay_s = inference_delay_ms / 1000.0

        # Single-threaded CPU inference keeps results deterministic and makes
        # the "one batch at a time" guarantee easy to reason about.
        torch.set_num_threads(1)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        self._pending: deque[Order] = deque()
        self._orders: dict[str, Order] = {}
        self._inflight = 0  # orders inside the batch currently being computed
        self._work = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._worker: Optional[asyncio.Task] = None

        # Occlusion explanations: separate FIFO, separate (non-order) capacity.
        self._explanations: dict[str, OcclusionJob] = {}
        self._explain_pending: deque[OcclusionJob] = deque()
        self._explain_active: Optional[OcclusionJob] = None
        self.explain_rounds_started = 0
        self.last_explain_round_size = 0

        # Counters for /stats and tests.
        self.batches_started = 0
        self.last_batch_size = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="batch-worker")

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._executor.shutdown(wait=False)

    # ------------------------------------------------------------------ #
    # helpers (event-loop thread only)
    # ------------------------------------------------------------------ #
    def _purge_expired_pending(self) -> None:
        """Mark queued orders whose deadline passed; they never enter the model."""
        now = time.monotonic()
        kept: list[Order] = []
        for o in self._pending:
            if o.state == QUEUED and o.deadline_at <= now:
                o.state = EXPIRED
            else:
                kept.append(o)
        if len(kept) != len(self._pending):
            self._pending.clear()
            self._pending.extend(kept)

    def _refresh_state(self, o: Order) -> None:
        """Lazy expiry when an order is observed."""
        if o.state in (QUEUED, PROCESSING) and o.deadline_at <= time.monotonic():
            if o.state == QUEUED:
                try:
                    self._pending.remove(o)
                except ValueError:
                    pass
            o.state = EXPIRED

    def capacity_used(self) -> int:
        self._purge_expired_pending()
        return len(self._pending) + self._inflight

    def stats(self) -> dict:
        self._purge_expired_pending()
        counts: dict[str, int] = {}
        for o in self._orders.values():
            self._refresh_state(o)
            counts[o.state] = counts.get(o.state, 0) + 1
        return {
            "capacity_used": len(self._pending) + self._inflight,
            "max_capacity": self.max_capacity,
            "queued": len(self._pending),
            "inflight": self._inflight,
            "batches_started": self.batches_started,
            "last_batch_size": self.last_batch_size,
            "explanations_queued": len(self._explain_pending),
            "explanations_active": 1 if self._explain_active is not None else 0,
            "explain_rounds_started": self.explain_rounds_started,
            "last_explain_round_size": self.last_explain_round_size,
            "orders": counts,
        }

    # ------------------------------------------------------------------ #
    # public API used by the HTTP layer
    # ------------------------------------------------------------------ #
    def submit(self, tensor: torch.Tensor, deadline_ms: int) -> Optional[Order]:
        """Accept an image, or return None when at capacity (16 by default)."""
        if self.capacity_used() >= self.max_capacity:
            return None
        order = Order(
            id=uuid.uuid4().hex[:12],
            tensor=tensor,
            deadline_ms=deadline_ms,
            deadline_at=time.monotonic() + deadline_ms / 1000.0,
        )
        self._orders[order.id] = order
        self._pending.append(order)
        self._work.set()
        return order

    def get(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is not None:
            self._refresh_state(o)
        return o

    def cancel(self, order_id: str) -> Optional[Order]:
        o = self._orders.get(order_id)
        if o is None:
            return None
        self._refresh_state(o)
        if o.state == QUEUED:
            try:
                self._pending.remove(o)
            except ValueError:
                pass
            o.state = CANCELLED
        elif o.state == PROCESSING:
            # Mark immediately. The slot stays occupied until the batch
            # compute actually ends; no new batch is started because of it.
            o.state = CANCELLED
        return o

    # ------------------------------------------------------------------ #
    # occlusion explanations (completed orders only)
    # ------------------------------------------------------------------ #
    def request_explanation(self, order_id: str) -> tuple[str, Optional[OcclusionJob]]:
        """Queue an explanation for a completed order.

        Returns (outcome, job) where outcome is one of:
          "created"     new (or retried) job queued
          "existing"    a queued/processing/completed job already exists
          "not_found"   unknown order
          "not_completed" order is not in `completed`
          "full"        explanation queue is at capacity
        """
        o = self._orders.get(order_id)
        if o is None:
            return "not_found", None
        self._refresh_state(o)
        if o.state != COMPLETED or o.result is None:
            return "not_completed", None
        existing = self._explanations.get(order_id)
        if existing is not None and existing.state != FAILED:
            return "existing", existing
        if len(self._explain_pending) >= self.max_capacity:
            return "full", None
        if existing is not None:
            # Failed job: allow an explicit retry, reusing the same job entry.
            existing.state = QUEUED
            existing.heatmap = None
            existing.partial = []
            existing.error = None
            existing.cells_done = 0
            job = existing
        else:
            job = OcclusionJob(
                order_id=order_id,
                tensor=o.tensor,
                predicted_class=o.result["predicted_class"],
                original_prob=o.result["scores"][o.result["predicted_class"]],
            )
            self._explanations[order_id] = job
        self._explain_pending.append(job)
        self._work.set()
        return "created", job

    def get_explanation(self, order_id: str) -> Optional[OcclusionJob]:
        """Explanation job if one exists (never triggers or alters anything)."""
        return self._explanations.get(order_id)

    @staticmethod
    def _next_explain_batch(
        active: Optional[OcclusionJob], pending: deque[OcclusionJob]
    ) -> list[OcclusionJob]:
        """Select the next occlusion round: 1..4 images, always from one job.

        The active job's remaining cells come first, so one order's 16 cells
        are spread over 4 rounds (16 cells, at most 4 per round). Only when no
        job is active is the FIFO head started. The returned list repeats the
        same job object `len(list)` times; the engine turns each entry into one
        masked image.
        """
        if active is not None:
            job = active
        else:
            # Defensive: skip any stale fully-computed entries in the FIFO.
            job = None
            while pending:
                candidate = pending.popleft()
                if candidate.state == QUEUED and candidate.cells_done < GRID * GRID:
                    job = candidate
                    break
            if job is None:
                return []
        take = min(GRID * GRID - job.cells_done, OCCLUSION_PER_ROUND)
        return [job] * take

    @staticmethod
    def _occlusion_batch(tensor: torch.Tensor, start: int, count: int) -> torch.Tensor:
        """Stack `count` copies of `tensor` with cells [start, start+count) masked."""
        images = []
        for idx in range(start, start + count):
            img = tensor.clone()
            r, c = divmod(idx, GRID)
            img[:, r * CELL : (r + 1) * CELL, c * CELL : (c + 1) * CELL] = OCCLUSION_VALUE
            images.append(img)
        return torch.stack(images)

    # ------------------------------------------------------------------ #
    # batching worker (single compute channel for frames AND explanations)
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            if not self._pending and self._explain_active is None and not self._explain_pending:
                # Idle: sleep until someone submits a frame or an explanation.
                self._work.clear()
                await self._work.wait()
                continue

            # Frame priority: drain every queued frame order before starting
            # another explanation round. A running batch is never preempted.
            if self._pending:
                await self._run_frame_batch(loop)
            else:
                await self._run_explain_round(loop)

    async def _run_frame_batch(self, loop: asyncio.AbstractEventLoop) -> None:
        # Batch-formation window: the idle model waits at most
        # `batch_wait_s` for more work, up to `batch_size` images.
        batch: list[Order] = []
        window_end = loop.time() + self.batch_wait_s
        while len(batch) < self.batch_size:
            self._purge_expired_pending()
            while self._pending and len(batch) < self.batch_size:
                batch.append(self._pending.popleft())
            if len(batch) >= self.batch_size:
                break
            remaining = window_end - loop.time()
            if remaining <= 0:
                break
            self._work.clear()
            try:
                await asyncio.wait_for(self._work.wait(), remaining)
            except asyncio.TimeoutError:
                break

        # Final gate: cancelled/expired orders never enter the model.
        now = time.monotonic()
        batch = [o for o in batch if o.state == QUEUED and o.deadline_at > now]
        if not batch:
            return

        batch_id = uuid.uuid4().hex[:8]
        for o in batch:
            o.state = PROCESSING
            o.batch_id = batch_id
        self._inflight = len(batch)
        self.batches_started += 1
        self.last_batch_size = len(batch)
        tensors = torch.stack([o.tensor for o in batch])

        try:
            scores = await loop.run_in_executor(self._executor, self._infer, tensors)
        except Exception:  # noqa: BLE001 - defensive: model failure must not kill the worker
            log.exception("inference failed for batch %s", batch_id)
            for o in batch:
                if o.state == PROCESSING:
                    o.state = FAILED
            self._inflight = 0
            return

        now = time.monotonic()
        for o, sc in zip(batch, scores):
            if o.state != PROCESSING:
                # Cancelled or expired mid-flight: result is discarded and
                # the late result must NOT revive the order.
                continue
            if o.deadline_at <= now:
                o.state = EXPIRED
                continue
            o.state = COMPLETED
            predicted = max(range(len(sc)), key=lambda i: sc[i])
            o.result = {"predicted_class": predicted, "scores": sc}
        self._inflight = 0

    async def _run_explain_round(self, loop: asyncio.AbstractEventLoop) -> None:
        """Compute at most 4 occlusion images on the same executor thread."""
        jobs = self._next_explain_batch(self._explain_active, self._explain_pending)
        if not jobs:
            self._explain_active = None
            return

        # A single round only ever mixes cells of one order (see _next_explain_batch).
        job = jobs[0]
        start = job.cells_done
        count = len(jobs)
        self._explain_active = job
        job.state = PROCESSING
        self.explain_rounds_started += 1
        self.last_explain_round_size = count

        try:
            occluded = self._occlusion_batch(job.tensor, start, count)
            probs = await loop.run_in_executor(self._executor, self._infer, occluded)
            deltas = [
                job.original_prob - float(row[job.predicted_class]) for row in probs
            ]
        except Exception as exc:  # noqa: BLE001 - never surface a partial heatmap
            log.exception("occlusion explanation failed for order %s", job.order_id)
            job.state = FAILED
            job.error = f"{type(exc).__name__}: {exc}"[:200]
            job.partial = []
            job.heatmap = None
            self._explain_active = None
            return

        job.partial.extend(deltas)
        job.cells_done += count
        if job.cells_done >= GRID * GRID:
            # Publish atomically, only when every cell succeeded.
            job.heatmap = [
                job.partial[r * GRID : (r + 1) * GRID] for r in range(GRID)
            ]
            job.partial = []
            job.state = COMPLETED
            self._explain_active = None
        # else: stay active so the next round continues the same order.

    # ------------------------------------------------------------------ #
    # runs on the single executor thread — never touches shared state
    # ------------------------------------------------------------------ #
    def _infer(self, batch_tensor: torch.Tensor) -> list[list[float]]:
        if self.inference_delay_s > 0:
            time.sleep(self.inference_delay_s)  # demo: simulate a slow model
        with torch.inference_mode():
            logits = self.model(batch_tensor)
            probs = torch.softmax(logits, dim=1)
        return [[float(v) for v in row] for row in probs]
