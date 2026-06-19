"""
Stress + allocation regression tests for leaks in talkingHead/backend code paths.

Uses stdlib ``tracemalloc`` to compare net allocation deltas attributable to stack
frames under ``talkingHead/backend/app`` and (for strategic memory stubs)
``talkingHead/backend/memory``. Optional ``psutil`` RSS probes are marked ``slow``.

Thresholds are intentionally loose versus micro-optimization; spikes usually mean
a missing disconnect, unbounded cache, or closure capturing a growing structure.
"""

from __future__ import annotations

import asyncio
import gc
import json
import sys
import tracemalloc
import wave
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.websocket.connection_manager import ConnectionManager

BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _net_allocation_bytes_under(
    before: tracemalloc.Snapshot,
    after: tracemalloc.Snapshot,
    *path_prefixes: str,
) -> int:
    """Net ``size_diff`` for stats whose **top** frame filename starts with one of ``path_prefixes``."""
    prefixes = tuple(path_prefixes)
    net = 0
    for stat in after.compare_to(before, "lineno"):
        traceback = stat.traceback
        if not traceback:
            continue
        top_path = traceback[0].filename
        try:
            if any(top_path.startswith(p) for p in prefixes):
                net += stat.size_diff
        except AttributeError:
            continue
    return net


def _net_app_allocation_bytes(before: tracemalloc.Snapshot, after: tracemalloc.Snapshot) -> int:
    return _net_allocation_bytes_under(before, after, str(BACKEND_ROOT / "app"))


def _net_memory_pkg_allocation_bytes(
    before: tracemalloc.Snapshot, after: tracemalloc.Snapshot
) -> int:
    return _net_allocation_bytes_under(before, after, str(BACKEND_ROOT / "memory"))


def _warm_gc() -> None:
    gc.collect()
    gc.collect()


def test_connection_manager_burst_net_app_allocation_bounded():
    """Many connect → broadcast → disconnect cycles should not grow app heap without bound."""

    async def _run_rounds(manager: ConnectionManager, rounds: int) -> None:
        for _ in range(rounds):
            ws = MagicMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            await manager.connect(ws)
            await manager.broadcast("ping")
            manager.disconnect(ws)
            assert manager.active_connections == []

    mgr = ConnectionManager()

    gc.disable()
    tracemalloc.start(25)
    try:
        asyncio.run(_run_rounds(mgr, 40))
        _warm_gc()
        snap_before = tracemalloc.take_snapshot()

        asyncio.run(_run_rounds(mgr, 500))
        _warm_gc()
        snap_after = tracemalloc.take_snapshot()

        net = _net_app_allocation_bytes(snap_before, snap_after)
        # Allow modest interpreter / mock churn; regressions blow past ~1 MiB quickly.
        assert net < 1_048_576, (
            f"ConnectionManager stress net growth in app/ was {net} bytes (> 1MiB)."
        )
    finally:
        tracemalloc.stop()
        gc.enable()
        gc.collect()


def test_broadcast_dead_peers_burst_net_app_allocation_bounded():
    """Broadcast that always drops failing sockets must not accumulate app allocations."""

    async def _rounds(manager: ConnectionManager, n: int) -> None:
        for _ in range(n):
            ws_dead = MagicMock()
            ws_dead.accept = AsyncMock()
            ws_dead.send_text = AsyncMock(side_effect=RuntimeError("connection reset"))
            await manager.connect(ws_dead)
            await manager.broadcast("x")
            assert manager.active_connections == []

    mgr = ConnectionManager()

    gc.disable()
    tracemalloc.start(25)
    try:
        asyncio.run(_rounds(mgr, 40))

        _warm_gc()
        snap_before = tracemalloc.take_snapshot()

        asyncio.run(_rounds(mgr, 400))

        _warm_gc()
        snap_after = tracemalloc.take_snapshot()

        net = _net_app_allocation_bytes(snap_before, snap_after)
        assert net < 1_048_576, (
            f"broadcast-dead-peer stress net growth in app/ was {net} bytes (> 1MiB)."
        )
    finally:
        tracemalloc.stop()
        gc.enable()
        gc.collect()


def test_voice_chat_stubbed_burst_net_voice_chat_bounded():
    """Repeated STT routing with mocks should not accumulate under ``voice_chat.py``."""

    async def _one_upload() -> None:
        mock_upload = MagicMock()
        mock_upload.read = AsyncMock(return_value=b"\x1a\x45\xdf\xa3")

        fake_model = MagicMock()
        seg = MagicMock()
        seg.text = "hello"
        fake_model.transcribe.return_value = [seg]

        with (
            patch("app.api.voice_chat._try_load_whisper_model", return_value=fake_model),
            patch("app.api.voice_chat.subprocess.run"),
            patch("app.api.voice_chat.sha256sum", return_value="abcd"),
            patch("app.api.voice_chat.os.remove"),
        ):
            from app.api.voice_chat import voice_input

            await voice_input(file=mock_upload)

    vc_path = str(BACKEND_ROOT / "app" / "api" / "voice_chat.py")

    def _voice_chat_net(prev: tracemalloc.Snapshot, cur: tracemalloc.Snapshot) -> int:
        net = 0
        for stat in cur.compare_to(prev, "lineno"):
            traceback = stat.traceback
            if not traceback:
                continue
            if traceback[0].filename == vc_path:
                net += stat.size_diff
        return net

    async def _batch(n: int) -> None:
        for _ in range(n):
            await _one_upload()

    asyncio.run(_batch(2))

    gc.disable()
    tracemalloc.start(25)
    try:
        asyncio.run(_batch(80))

        _warm_gc()
        snap_before = tracemalloc.take_snapshot()

        asyncio.run(_batch(450))

        _warm_gc()
        snap_after = tracemalloc.take_snapshot()

        net = _voice_chat_net(snap_before, snap_after)
        assert net < 512_000, (
            f"voice_chat.py net growth across stubbed bursts was {net} bytes (> 512KiB)."
        )
    finally:
        tracemalloc.stop()
        gc.enable()
        gc.collect()


def test_openapi_fetch_burst_net_app_allocation_bounded():
    """HTTP stack noise is allowed; allocations tagged to talkingHead ``app`` should plateau."""

    with TestClient(app) as client:
        for _ in range(30):
            r = client.get("/openapi.json")
            assert r.status_code == 200
            del r

        gc.disable()
        tracemalloc.start(25)
        try:
            _warm_gc()
            snap_before = tracemalloc.take_snapshot()

            for _ in range(280):
                r = client.get("/openapi.json")
                assert r.status_code == 200
                del r

            _warm_gc()
            snap_after = tracemalloc.take_snapshot()

            net = _net_app_allocation_bytes(snap_before, snap_after)
            # OpenAPI parses are heavy; tighter bound catches real leaks inside our handlers.
            assert net < 2_097_152, f"OpenAPI polling net growth in app/ was {net} bytes (> 2MiB)."
        finally:
            tracemalloc.stop()
            gc.enable()
            gc.collect()


def _writes_minimal_wav_stub(_text: str, wav_file: wave.Wave_write, **_kwargs: object) -> None:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(22050)
    wav_file.writeframes(b"\x00\x00" * 64)


def test_chat_ws_stubbed_turns_net_app_bounded():
    """Fully mocked ``chat_endpoint`` repetitions — covers prompt assembly / JSON / streaming hooks."""

    from app.websocket.chat_ws import chat_endpoint

    async def _batch(times: int) -> None:
        with (
            patch("app.websocket.chat_ws.manager") as mock_mgr,
            patch("app.websocket.chat_ws.store_message", new_callable=AsyncMock),
            patch(
                "app.websocket.chat_ws.search_relevant_memories",
                new_callable=AsyncMock,
                return_value=[
                    {"similarity": 0.9, "zone": "live", "text": "prior context snippet"},
                ],
            ),
            patch(
                "app.websocket.chat_ws.get_recent_messages",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("app.websocket.chat_ws.chat_processor") as mock_proc,
            patch("app.websocket.chat_ws.store_conversation", new_callable=AsyncMock),
        ):
            mock_mgr.connect = AsyncMock(return_value=True)
            mock_mgr.disconnect = MagicMock()
            mock_proc.stream_response = AsyncMock(return_value="assistant boilerplate reply")
            for _ in range(times):
                ws = MagicMock()
                ws.accept = AsyncMock()
                ws.send_text = AsyncMock()
                ws.close = AsyncMock()
                ws.receive_text = AsyncMock(
                    side_effect=[
                        json.dumps({"message": "repeatable stress turn"}),
                        WebSocketDisconnect(),
                    ]
                )
                await chat_endpoint(ws)

    asyncio.run(_batch(25))

    gc.disable()
    tracemalloc.start(25)
    try:
        _warm_gc()
        snap_before = tracemalloc.take_snapshot()

        asyncio.run(_batch(140))

        _warm_gc()
        snap_after = tracemalloc.take_snapshot()

        net = _net_app_allocation_bytes(snap_before, snap_after)
        assert net < 3_145_728, f"chat_ws mocked turns net growth in app/ was {net} bytes (> 3MiB)."
    finally:
        tracemalloc.stop()
        gc.enable()
        gc.collect()


def test_tts_post_burst_net_app_bounded():
    fake_voice = MagicMock()
    fake_voice.synthesize.side_effect = _writes_minimal_wav_stub

    with TestClient(app) as client, patch("app.api.tts._try_load_voice", return_value=fake_voice):
        for _ in range(30):
            r = client.post("/api/chat/speak", json={"text": "stress phrase"})
            assert r.status_code == 200
            del r

        gc.disable()
        tracemalloc.start(25)
        try:
            _warm_gc()
            snap_before = tracemalloc.take_snapshot()

            for _ in range(220):
                r = client.post("/api/chat/speak", json={"text": "short line"})
                assert r.status_code == 200
                del r

            _warm_gc()
            snap_after = tracemalloc.take_snapshot()

            net = _net_app_allocation_bytes(snap_before, snap_after)
            assert net < 2_097_152, (
                f"TTS synthesize burst net growth in app/ was {net} bytes (> 2MiB)."
            )
        finally:
            tracemalloc.stop()
            gc.enable()
            gc.collect()


def test_docs_html_burst_net_app_bounded():
    with TestClient(app) as client:
        for _ in range(25):
            r = client.get("/docs")
            assert r.status_code == 200
            del r

        gc.disable()
        tracemalloc.start(25)
        try:
            _warm_gc()
            snap_before = tracemalloc.take_snapshot()

            for _ in range(200):
                r = client.get("/docs")
                assert r.status_code == 200
                del r

            _warm_gc()
            snap_after = tracemalloc.take_snapshot()

            net = _net_app_allocation_bytes(snap_before, snap_after)
            assert net < 6_291_456, f"/docs burst net growth in app/ was {net} bytes (> 6MiB)."
        finally:
            tracemalloc.stop()
            gc.enable()
            gc.collect()


def test_health_dual_routes_burst_net_app_bounded():
    """Alternating lightweight health probes (loaders stubbed — no ONNX / GGML)."""

    with (
        TestClient(app) as client,
        patch("app.api.tts._try_load_voice", return_value=MagicMock()),
        patch("app.api.voice_chat._try_load_whisper_model", return_value=None),
    ):
        for _ in range(40):
            assert client.get("/api/chat/speak/health").status_code == 200
            assert client.get("/api/chat/voice/health").status_code == 200

        gc.disable()
        tracemalloc.start(25)
        try:
            _warm_gc()
            snap_before = tracemalloc.take_snapshot()

            for _ in range(350):
                assert client.get("/api/chat/speak/health").status_code == 200
                assert client.get("/api/chat/voice/health").status_code == 200

            _warm_gc()
            snap_after = tracemalloc.take_snapshot()

            net = _net_app_allocation_bytes(snap_before, snap_after)
            assert net < 1_572_864, (
                f"health route alternation net growth in app/ was {net} bytes (> 1.5MiB)."
            )
        finally:
            tracemalloc.stop()
            gc.enable()
            gc.collect()


def test_send_personal_message_burst_net_app_bounded():
    async def _rounds(manager: ConnectionManager, n: int) -> None:
        for _ in range(n):
            ws = MagicMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            await manager.connect(ws)
            await manager.send_personal_message("payload", ws)
            manager.disconnect(ws)

    mgr = ConnectionManager()

    asyncio.run(_rounds(mgr, 40))

    gc.disable()
    tracemalloc.start(25)
    try:
        _warm_gc()
        snap_before = tracemalloc.take_snapshot()

        asyncio.run(_rounds(mgr, 600))

        _warm_gc()
        snap_after = tracemalloc.take_snapshot()

        net = _net_app_allocation_bytes(snap_before, snap_after)
        assert net < 1_048_576, (
            f"send_personal_message burst net growth in app/ was {net} bytes (> 1MiB)."
        )
    finally:
        tracemalloc.stop()
        gc.enable()
        gc.collect()


@pytest.mark.memory
@pytest.mark.slow
def test_strategic_search_mock_to_thread_burst_memory_pkg_bounded():
    """``memory.strategic`` path with MCP enabled + DAG off + mocked ``asyncio.to_thread``."""

    async def batch(n: int) -> None:
        from memory.strategic import search_relevant_memories

        for i in range(n):
            await search_relevant_memories(f"stress query iteration {i}", deep=False)

    with (
        patch("memory.strategic._mcp_memory_available", True),
        patch("memory.strategic._dag_retrieval_available", False),
        patch("memory.strategic.asyncio.to_thread", new_callable=AsyncMock, return_value=[]),
    ):
        asyncio.run(batch(50))

        gc.disable()
        tracemalloc.start(25)
        try:
            _warm_gc()
            snap_before = tracemalloc.take_snapshot()

            asyncio.run(batch(600))

            _warm_gc()
            snap_after = tracemalloc.take_snapshot()

            net = _net_memory_pkg_allocation_bytes(snap_before, snap_after)
            assert net < 1_048_576, (
                f"memory.strategic search burst net growth was {net} bytes (> 1MiB)."
            )
        finally:
            tracemalloc.stop()
            gc.enable()
            gc.collect()


@pytest.mark.memory
@pytest.mark.slow
def test_rss_docs_html_burst_growth_cap():
    proc = pytest.importorskip("psutil").Process()

    def rss() -> int:
        _warm_gc()
        return proc.memory_info().rss

    with TestClient(app) as client:
        for _ in range(50):
            r = client.get("/docs")
            assert r.status_code == 200
            del r

        before = rss()
        for _ in range(260):
            r = client.get("/docs")
            assert r.status_code == 200
            del r
        after = rss()

        growth = after - before
        soft_cap = 120 * 1024 * 1024
        assert growth <= soft_cap, (
            f"RSS grew ~{growth // 1048576} MiB across /docs bursts (threshold {soft_cap // 1048576} MiB)."
        )


@pytest.mark.memory
@pytest.mark.slow
def test_rss_mixed_hot_routes_growth_cap():
    proc = pytest.importorskip("psutil").Process()

    def rss() -> int:
        _warm_gc()
        return proc.memory_info().rss

    with (
        TestClient(app) as client,
        patch("app.api.tts._try_load_voice", return_value=MagicMock()),
        patch("app.api.voice_chat._try_load_whisper_model", return_value=None),
    ):
        for _ in range(60):
            assert client.get("/openapi.json").status_code == 200
            assert client.get("/docs").status_code == 200
            assert client.get("/api/chat/speak/health").status_code == 200
            assert client.get("/api/chat/voice/health").status_code == 200

        before = rss()
        for _ in range(220):
            assert client.get("/openapi.json").status_code == 200
            assert client.get("/docs").status_code == 200
            assert client.get("/api/chat/speak/health").status_code == 200
            assert client.get("/api/chat/voice/health").status_code == 200
        after = rss()

        growth = after - before
        soft_cap = 150 * 1024 * 1024
        assert growth <= soft_cap, (
            f"RSS grew ~{growth // 1048576} MiB on mixed-route burst "
            f"(threshold {soft_cap // 1048576} MiB)."
        )


@pytest.mark.memory
@pytest.mark.slow
def test_rss_openapi_burst_growth_cap():
    """Coarse RSS guard: optional (psutil); skip in minimal environments."""
    proc = pytest.importorskip("psutil").Process()

    def rss() -> int:
        _warm_gc()
        return proc.memory_info().rss

    with TestClient(app) as client:
        for _ in range(80):
            r = client.get("/openapi.json")
            assert r.status_code == 200
            del r

        before = rss()
        for _ in range(400):
            r = client.get("/openapi.json")
            assert r.status_code == 200
            del r
        after = rss()

        growth = after - before
        soft_cap = 80 * 1024 * 1024  # 80 MiB slack for allocator noise + JSON parsing
        assert growth <= soft_cap, (
            f"RSS grew ~{growth // 1048576} MiB under OpenAPI bursts (threshold {soft_cap // 1048576} MiB)."
        )


@pytest.mark.memory
@pytest.mark.slow
def test_rss_connection_manager_burst_growth_cap():
    """RSS guard isolated to websocket manager patterns (Mocks only — stacks small)."""
    proc = pytest.importorskip("psutil").Process()

    def rss() -> int:
        _warm_gc()
        return proc.memory_info().rss

    async def _spam(manager: ConnectionManager, reps: int) -> None:
        for _ in range(reps):
            ws = MagicMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            await manager.connect(ws)
            await manager.broadcast("msg")
            manager.disconnect(ws)

    mgr = ConnectionManager()

    asyncio.run(_spam(mgr, 200))
    before = rss()
    asyncio.run(_spam(mgr, 2_500))
    after = rss()

    growth = after - before
    soft_cap = 40 * 1024 * 1024  # mocks should barely move RSS
    assert growth <= soft_cap, (
        f"RSS grew ~{growth // 1048576} MiB across ConnectionManager burst (threshold {soft_cap // 1048576} MiB)."
    )
