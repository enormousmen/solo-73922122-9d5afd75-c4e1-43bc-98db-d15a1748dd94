"""End-to-end tests for the micro-batch classifier API.

Each test gets a fresh app/engine via the `client` factory. Inference delay
is injected through Settings to simulate a slow model without sleeping the
event loop.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
MODEL = Path(__file__).resolve().parent.parent / "app" / "model.pt"
SAMPLE_NAMES = ["sample_gradient.png", "sample_circles.png", "sample_stripes.png", "sample_noise.png"]


def make_settings(**kw) -> Settings:
    kw.setdefault("model_path", str(MODEL))
    return Settings(**kw)


class Client:
    """httpx client wrapper with helpers."""

    def __init__(self, http: httpx.AsyncClient, app):
        self.http = http
        self.app = app

    async def submit(self, name: str, deadline_ms: int = 60_000):
        data = (SAMPLES / name).read_bytes()
        return await self.http.post(
            "/predict",
            files={"file": (name, data, "image/png")},
            data={"deadline_ms": str(deadline_ms)},
        )

    async def get(self, order_id: str):
        return (await self.http.get(f"/orders/{order_id}")).json()

    async def cancel(self, order_id: str):
        return await self.http.post(f"/orders/{order_id}/cancel")

    async def wait_state(self, order_id: str, states, timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = await self.get(order_id)
            if body["status"] in states:
                return body
            await asyncio.sleep(0.01)
        raise AssertionError(f"order {order_id} never reached {states}: {body}")


@pytest.fixture
async def make_client():
    clients = []

    async def factory(**settings_kw):
        app = create_app(make_settings(**settings_kw))
        cm = LifespanManager(app)
        await cm.__aenter__()
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
        client = Client(http, app)
        clients.append((client, cm))
        return client

    yield factory
    for client, cm in clients:
        await client.http.aclose()
        await cm.__aexit__(None, None, None)


# --------------------------------------------------------------------- #
# basic flow
# --------------------------------------------------------------------- #
async def test_single_prediction_completes(make_client):
    c = await make_client()
    r = await c.submit("sample_gradient.png", deadline_ms=5000)
    assert r.status_code == 201, r.text
    oid = r.json()["order_id"]
    body = await c.wait_state(oid, {"completed"})
    scores = body["result"]["scores"]
    assert len(scores) == 10
    assert abs(sum(scores) - 1.0) < 1e-4
    assert 0 <= body["result"]["predicted_class"] < 10


async def test_validation_and_404(make_client):
    c = await make_client()
    # non-positive deadline rejected
    r = await c.submit("sample_gradient.png", deadline_ms=0)
    assert r.status_code == 400
    # garbage bytes rejected
    r = await c.http.post(
        "/predict",
        files={"file": ("x.bin", b"not an image", "application/octet-stream")},
        data={"deadline_ms": "1000"},
    )
    assert r.status_code == 400
    # unknown order
    assert (await c.http.get("/orders/nope")).status_code == 404
    assert (await c.http.post("/orders/nope/cancel")).status_code == 404


# --------------------------------------------------------------------- #
# batching correctness: batched scores match per-image inference <= 1e-5
# --------------------------------------------------------------------- #
async def test_batch_scores_match_single_inference(make_client):
    c = await make_client()

    # reference: each image inferred alone
    single_scores = {}
    for name in SAMPLE_NAMES:
        r = await c.submit(name)
        body = await c.wait_state(r.json()["order_id"], {"completed"})
        single_scores[name] = body["result"]["scores"]

    # batched: 4 concurrent submissions must ride one batch of 4
    responses = await asyncio.gather(*[c.submit(n) for n in SAMPLE_NAMES])
    ids = [r.json()["order_id"] for r in responses]
    bodies = await asyncio.gather(*[c.wait_state(i, {"completed"}) for i in ids])
    batch_ids = {b["batch_id"] for b in bodies}
    assert len(batch_ids) == 1, f"expected one shared batch, got {batch_ids}"
    assert c.app.state.engine.last_batch_size == 4

    for name, body in zip(SAMPLE_NAMES, bodies):
        for a, b in zip(body["result"]["scores"], single_scores[name]):
            assert abs(a - b) <= 1e-5, f"{name}: {a} vs {b}"


async def test_idle_model_waits_to_fill_batch(make_client):
    c = await make_client(batch_wait_ms=200)
    r1 = await c.submit("sample_gradient.png")
    await asyncio.sleep(0.05)  # inside the batch window
    r2 = await c.submit("sample_circles.png")
    b1 = await c.wait_state(r1.json()["order_id"], {"completed"})
    b2 = await c.wait_state(r2.json()["order_id"], {"completed"})
    assert b1["batch_id"] == b2["batch_id"]
    assert c.app.state.engine.last_batch_size == 2


# --------------------------------------------------------------------- #
# capacity: 16 queued+in-flight, then 429
# --------------------------------------------------------------------- #
async def test_capacity_full_rejects_429(make_client):
    c = await make_client(inference_delay_ms=1500)
    accepted = []
    for i in range(16):
        r = await c.submit(SAMPLE_NAMES[i % 4])
        assert r.status_code == 201, (i, r.text)
        accepted.append(r.json()["order_id"])
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 429, r.text
    stats = (await c.http.get("/stats")).json()
    assert stats["capacity_used"] == 16


# --------------------------------------------------------------------- #
# cancellation
# --------------------------------------------------------------------- #
async def test_cancel_queued_order_never_enters_model(make_client):
    c = await make_client(inference_delay_ms=1200)
    ids = []
    for i in range(5):  # 4 fill the running batch, 5th stays queued
        r = await c.submit(SAMPLE_NAMES[i % 4])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)  # let the first batch start
    victim = ids[4]
    body = await c.get(victim)
    assert body["status"] == "queued"
    r = await c.cancel(victim)
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    # after everything settles it must still be cancelled, with no result/batch
    await asyncio.sleep(1.6)
    body = await c.get(victim)
    assert body["status"] == "cancelled"
    assert body["result"] is None
    assert body["batch_id"] is None


async def test_cancel_processing_marks_now_but_holds_capacity(make_client):
    c = await make_client(inference_delay_ms=1500)
    ids = []
    for i in range(4):  # exactly one batch
        r = await c.submit(SAMPLE_NAMES[i])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)  # batch is now computing
    victim, others = ids[0], ids[1:]
    assert (await c.get(victim))["status"] == "processing"

    r = await c.cancel(victim)
    assert r.json()["status"] == "cancelled"  # marked immediately
    assert (await c.get(victim))["status"] == "cancelled"

    # slot is still occupied while the batch computes: only 12 free slots
    extra = []
    for _ in range(12):
        r = await c.submit("sample_gradient.png")
        assert r.status_code == 201
        extra.append(r.json()["order_id"])
    r = await c.submit("sample_gradient.png")
    assert r.status_code == 429, "cancelled in-flight order must still occupy capacity"

    # no new batch may start while the cancelled one's batch is computing
    assert c.app.state.engine.batches_started == 1

    # after compute ends: victim stays cancelled without result, others complete
    await asyncio.sleep(1.6)
    body = await c.get(victim)
    assert body["status"] == "cancelled" and body["result"] is None
    for oid in others:
        assert (await c.get(oid))["status"] == "completed"


async def test_cancel_completed_order_conflicts(make_client):
    c = await make_client()
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await c.wait_state(oid, {"completed"})
    r = await c.cancel(oid)
    assert r.status_code == 409


# --------------------------------------------------------------------- #
# expiry
# --------------------------------------------------------------------- #
async def test_queued_order_expires_and_never_enters_model(make_client):
    c = await make_client(inference_delay_ms=1200)
    ids = []
    for i in range(4):  # occupy the model with one batch
        r = await c.submit(SAMPLE_NAMES[i])
        ids.append(r.json()["order_id"])
    await asyncio.sleep(0.2)
    r = await c.submit("sample_gradient.png", deadline_ms=300)  # expires while queued
    victim = r.json()["order_id"]
    await asyncio.sleep(0.5)
    body = await c.get(victim)
    assert body["status"] == "expired"
    await asyncio.sleep(1.5)  # let the running batch finish and next batch form
    body = await c.get(victim)
    assert body["status"] == "expired"
    assert body["result"] is None
    assert body["batch_id"] is None  # never entered the model


async def test_processing_order_expires_midflight_no_result(make_client):
    c = await make_client(inference_delay_ms=1000)
    r = await c.submit("sample_gradient.png", deadline_ms=400)
    oid = r.json()["order_id"]
    await asyncio.sleep(1.6)  # batch computed (late), deadline long past
    body = await c.get(oid)
    assert body["status"] == "expired"
    assert body["result"] is None


async def test_late_result_does_not_revive_cancelled_order(make_client):
    c = await make_client(inference_delay_ms=800)
    r = await c.submit("sample_gradient.png", deadline_ms=60_000)
    oid = r.json()["order_id"]
    await asyncio.sleep(0.2)  # in-flight now
    await c.cancel(oid)
    await asyncio.sleep(1.2)  # batch compute finished, result arrived late
    body = await c.get(oid)
    assert body["status"] == "cancelled"
    assert body["result"] is None


# --------------------------------------------------------------------- #
# slow inference must not block submit / query / cancel
# --------------------------------------------------------------------- #
async def test_slow_inference_does_not_block_api(make_client):
    c = await make_client(inference_delay_ms=2000)
    r = await c.submit("sample_gradient.png")
    oid = r.json()["order_id"]
    await asyncio.sleep(0.2)  # batch is computing for ~2s

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    (resp, dt1) = await timed(c.http.get(f"/orders/{oid}"))
    (resp2, dt2) = await timed(c.submit("sample_circles.png"))
    (resp3, dt3) = await timed(c.cancel(resp2.json()["order_id"]))
    (resp4, dt4) = await timed(c.http.get("/stats"))
    for dt in (dt1, dt2, dt3, dt4):
        assert dt < 0.5, f"API blocked for {dt:.3f}s during slow inference"
    assert resp.status_code == 200
    assert resp2.status_code == 201
    assert resp3.status_code == 200
    assert resp4.status_code == 200


# --------------------------------------------------------------------- #
# occlusion explanations
# --------------------------------------------------------------------- #
def _reference_heatmap(name: str, predicted: int) -> list[list[float]]:
    """Independent reference: softmax(model(original))[pred] - softmax(model(masked))[pred]."""
    import torch

    from app.preprocessing import preprocess

    model = torch.jit.load(str(MODEL), map_location="cpu")
    model.eval()
    tensor = preprocess((SAMPLES / name).read_bytes())
    with torch.inference_mode():
        orig = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
        images = []
        for idx in range(16):
            img = tensor.clone()
            r, c = divmod(idx, 4)
            img[:, r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] = 0.0
            images.append(img)
        masked = torch.softmax(model(torch.stack(images)), dim=1)
    return [
        [float(orig[predicted] - masked[r * 4 + c, predicted]) for c in range(4)]
        for r in range(4)
    ]


async def _completed(c: Client, name: str) -> dict:
    r = await c.submit(name)
    return await c.wait_state(r.json()["order_id"], {"completed"})


async def _wait_explanation(c: Client, oid: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = await c.http.get(f"/orders/{oid}/explanation")
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("completed", "failed"):
            return body
        # no partial heatmap while queued/processing
        assert body["heatmap"] is None
        await asyncio.sleep(0.02)
    raise AssertionError(f"explanation {oid} never finished: {body}")


async def test_explanation_matches_reference_and_keeps_order(make_client):
    c = await make_client()
    name = "sample_circles.png"
    order = await _completed(c, name)
    oid = order["order_id"]
    predicted = order["result"]["predicted_class"]
    original_scores = list(order["result"]["scores"])

    r = await c.http.post(f"/orders/{oid}/explanation")
    assert r.status_code == 202, r.text
    assert r.json()["status"] in ("queued", "processing")
    assert r.json()["heatmap"] is None

    body = await _wait_explanation(c, oid)
    assert body["status"] == "completed"
    assert body["grid"] == [4, 4]
    assert body["predicted_class"] == predicted
    assert abs(body["original_prob"] - original_scores[predicted]) < 1e-12
    heatmap = body["heatmap"]
    assert len(heatmap) == 4 and all(len(row) == 4 for row in heatmap)

    ref = _reference_heatmap(name, predicted)
    for r_ in range(4):
        for c_ in range(4):
            assert abs(heatmap[r_][c_] - ref[r_][c_]) <= 1e-5, (r_, c_)
    # negative differences are preserved as-is (they exist for this sample)
    assert any(v < 0 for row in heatmap for v in row)

    # explanation work is split into 4 rounds of at most 4 occlusion images
    engine = c.app.state.engine
    assert engine.explain_rounds_started == 4
    assert engine.last_explain_round_size == 4

    # original order untouched: same state, same scores, cancelling still 409
    order_after = await c.get(oid)
    assert order_after["status"] == "completed"
    assert order_after["result"]["scores"] == original_scores
    assert order_after["result"]["predicted_class"] == predicted
    assert (await c.cancel(oid)).status_code == 409

    # repeated request is idempotent: returns the same finished job
    r = await c.http.post(f"/orders/{oid}/explanation")
    assert r.status_code == 202
    assert r.json()["status"] == "completed"
    assert r.json()["heatmap"] == heatmap
    assert (await c.http.get("/stats")).json()["explain_rounds_started"] == 4


async def test_explanation_only_for_completed_orders(make_client):
    c = await make_client(inference_delay_ms=1200)
    r = await c.submit("sample_gradient.png", deadline_ms=60_000)
    oid = r.json()["order_id"]
    await asyncio.sleep(0.15)  # processing
    assert (await c.get(oid))["status"] == "processing"
    assert (await c.http.post(f"/orders/{oid}/explanation")).status_code == 409
    assert (await c.http.get(f"/orders/{oid}/explanation")).status_code == 409

    await asyncio.sleep(1.4)
    assert (await c.get(oid))["status"] == "completed"

    # queued order (model busy) is rejected too
    r2 = await c.submit("sample_circles.png", deadline_ms=60_000)
    oid2 = r2.json()["order_id"]
    assert (await c.get(oid2))["status"] == "queued"
    assert (await c.http.post(f"/orders/{oid2}/explanation")).status_code == 409
    assert (await c.http.get(f"/orders/{oid2}/explanation")).status_code == 409

    # unknown order
    assert (await c.http.post("/orders/nope/explanation")).status_code == 404
    assert (await c.http.get("/orders/nope/explanation")).status_code == 404

    # no explanation requested yet on a completed order
    assert (await c.http.get(f"/orders/{oid}/explanation")).status_code == 404

    # cancelled and expired orders are not explainable
    r3 = await c.submit("sample_stripes.png", deadline_ms=60_000)
    oid3 = r3.json()["order_id"]
    await c.cancel(oid3)
    assert (await c.http.post(f"/orders/{oid3}/explanation")).status_code == 409
    r4 = await c.submit("sample_noise.png", deadline_ms=100)
    oid4 = r4.json()["order_id"]
    await asyncio.sleep(0.25)
    assert (await c.get(oid4))["status"] == "expired"
    assert (await c.http.post(f"/orders/{oid4}/explanation")).status_code == 409


async def test_no_partial_heatmap_while_computing(make_client):
    c = await make_client(inference_delay_ms=400)
    order = await _completed(c, "sample_noise.png")
    oid = order["order_id"]

    r = await c.http.post(f"/orders/{oid}/explanation")
    assert r.status_code == 202

    # first round done, second round in flight: still no heatmap
    await asyncio.sleep(0.55)
    body = (await c.http.get(f"/orders/{oid}/explanation")).json()
    assert body["status"] in ("processing", "completed")
    if body["status"] == "processing":
        assert body["heatmap"] is None
        assert 0 < body["progress"]["cells_done"] < 16

    body = await _wait_explanation(c, oid)
    assert body["status"] == "completed"
    assert body["heatmap"] is not None


async def test_explanation_failure_returns_no_heatmap_and_keeps_order(make_client):
    c = await make_client(inference_delay_ms=50)
    order = await _completed(c, "sample_gradient.png")
    oid = order["order_id"]
    original = await c.get(oid)
    engine = c.app.state.engine

    def boom(tensor, start, count):
        if start >= 4:  # fail on the second round, after 4 cells succeeded
            raise RuntimeError("simulated occlusion failure")
        return original_occlusion(tensor, start, count)

    original_occlusion = engine._occlusion_batch
    engine._occlusion_batch = boom
    try:
        r = await c.http.post(f"/orders/{oid}/explanation")
        assert r.status_code == 202
        body = await _wait_explanation(c, oid)
    finally:
        engine._occlusion_batch = original_occlusion

    assert body["status"] == "failed"
    assert body["heatmap"] is None
    assert "simulated occlusion failure" in body["error"]

    # order untouched; retry after restoring the engine succeeds
    after = await c.get(oid)
    assert after["status"] == "completed"
    assert after["result"] == original["result"]

    r = await c.http.post(f"/orders/{oid}/explanation")
    assert r.status_code == 202
    body = await _wait_explanation(c, oid)
    assert body["status"] == "completed" and body["heatmap"] is not None


async def test_frames_prioritised_between_explanation_rounds(make_client):
    c = await make_client(inference_delay_ms=400)
    order = await _completed(c, "sample_stripes.png")
    oid = order["order_id"]

    r = await c.http.post(f"/orders/{oid}/explanation")
    assert r.status_code == 202
    await asyncio.sleep(0.15)  # first explanation round is computing

    # a frame submitted during an explanation round is served before the
    # explanation's remaining rounds
    fr = await c.submit("sample_circles.png", deadline_ms=60_000)
    frame_id = fr.json()["order_id"]
    frame_body = await c.wait_state(frame_id, {"completed"}, timeout=5.0)
    frame_done = time.monotonic()

    expl = (await c.http.get(f"/orders/{oid}/explanation")).json()
    assert expl["status"] in ("queued", "processing"), expl
    assert frame_body["status"] == "completed"

    body = await _wait_explanation(c, oid)
    assert body["status"] == "completed"
    assert time.monotonic() > frame_done
    assert expl["heatmap"] is None
    assert body["progress"]["cells_done"] == 16
    engine = c.app.state.engine
    assert engine.last_explain_round_size <= 4
    assert engine.batches_started == 2  # original order + prioritised frame


async def test_multiple_explanations_are_fifo_and_independent(make_client):
    c = await make_client(inference_delay_ms=300)
    first = await _completed(c, "sample_gradient.png")
    second = await _completed(c, "sample_circles.png")

    ids = [first["order_id"], second["order_id"]]
    for oid in ids:
        assert (await c.http.post(f"/orders/{oid}/explanation")).status_code == 202

    states = [await _wait_explanation(c, oid, timeout=15.0) for oid in ids]
    for body, order in zip(states, (first, second)):
        name = "sample_gradient.png" if order is first else "sample_circles.png"
        assert body["status"] == "completed"
        predicted = order["result"]["predicted_class"]
        ref = _reference_heatmap(name, predicted)
        for r_ in range(4):
            for c_ in range(4):
                assert abs(body["heatmap"][r_][c_] - ref[r_][c_]) <= 1e-5

    # FIFO: the first requested explanation was started first and finished first
    engine = c.app.state.engine
    assert engine.explain_rounds_started == 8
    assert engine.last_explain_round_size == 4
    first_job = engine.get_explanation(ids[0])
    second_job = engine.get_explanation(ids[1])
    assert first_job.created_at <= second_job.created_at
    # each round belongs to a single order: 4 rounds per explanation
    assert first_job.cells_done == second_job.cells_done == 16


async def test_slow_explanation_does_not_block_api(make_client):
    c = await make_client(inference_delay_ms=600)
    order = await _completed(c, "sample_gradient.png")
    oid = order["order_id"]
    r = await c.http.post(f"/orders/{oid}/explanation")
    assert r.status_code == 202
    await asyncio.sleep(0.1)  # explanation round computing

    async def timed(coro):
        t0 = time.monotonic()
        res = await coro
        return res, time.monotonic() - t0

    (resp, dt1) = await timed(c.http.get(f"/orders/{oid}"))
    (resp2, dt2) = await timed(c.submit("sample_noise.png"))
    (resp3, dt3) = await timed(c.cancel(resp2.json()["order_id"]))
    (resp4, dt4) = await timed(c.http.get(f"/orders/{oid}/explanation"))
    (resp5, dt5) = await timed(c.http.get("/stats"))
    for dt in (dt1, dt2, dt3, dt4, dt5):
        assert dt < 0.5, f"API blocked for {dt:.3f}s during slow explanation"
    assert resp.status_code == 200
    assert resp2.status_code == 201
    assert resp3.status_code == 200
    assert resp4.status_code == 200
    assert resp5.status_code == 200
    # explanation still completes despite the interruption
    body = await _wait_explanation(c, oid)
    assert body["status"] == "completed"


def test_explanation_queue_capacity_and_idempotency():
    """Engine-level: explanation FIFO cap is 16 and requests are idempotent."""
    import torch

    from app.engine import COMPLETED, BatchEngine, Order

    engine = BatchEngine(model_path=str(MODEL))
    try:
        for i in range(17):
            o = Order(
                id=f"o{i}",
                tensor=torch.zeros(3, 64, 64),
                deadline_ms=1000,
                deadline_at=time.monotonic() + 100,
                state=COMPLETED,
                result={"predicted_class": 0, "scores": [1.0] + [0.0] * 9},
            )
            engine._orders[o.id] = o

        for i in range(16):
            assert engine.request_explanation(f"o{i}")[0] == "created"
        assert engine.request_explanation("o16")[0] == "full"
        assert engine.request_explanation("nope")[0] == "not_found"

        # idempotent: same job returned, queue length unchanged
        outcome, job = engine.request_explanation("o0")
        assert outcome == "existing"
        assert job is engine.get_explanation("o0")
        assert len(engine._explain_pending) == 16

        # a non-completed order cannot be explained
        engine._orders["o16"].state = "queued"
        assert engine.request_explanation("o16")[0] == "not_completed"

        # a round never exceeds 4 occlusion images and finishes the active job first
        engine._explain_active = None
        engine._explain_pending.clear()
        engine._explanations.clear()
        for oid in ("o0", "o1"):
            assert engine.request_explanation(oid)[0] == "created"
        rounds = []
        while engine._explain_pending or engine._explain_active is not None:
            jobs = engine._next_explain_batch(
                engine._explain_active, engine._explain_pending
            )
            assert 0 < len(jobs) <= 4
            job = jobs[0]
            assert all(j is job for j in jobs)
            job.cells_done += len(jobs)
            rounds.append((job.order_id, len(jobs)))
            engine._explain_active = None if job.cells_done >= 16 else job
        assert [n for _, n in rounds] == [4] * 8  # two 16-cell jobs, 4 per round
        assert [oid for oid, _ in rounds] == ["o0"] * 4 + ["o1"] * 4
    finally:
        engine._executor.shutdown(wait=False)
