"""
Lifecycle / resource-leak tests inspired by daimonos memory-leak prevention rules.

Verifies that connections, temp files, and sessions are properly cleaned up
so nothing accumulates unboundedly at runtime.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.websocket.connection_manager import ConnectionManager


# ---------------------------------------------------------------------------
# ConnectionManager lifecycle
# ---------------------------------------------------------------------------


def _make_mock_ws():
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect_disconnect_leaves_empty():
    """Connect N websockets, disconnect all → active_connections should be empty."""
    mgr = ConnectionManager()
    sockets = [_make_mock_ws() for _ in range(5)]

    for ws in sockets:
        await mgr.connect(ws)
    assert len(mgr.active_connections) == 5

    for ws in sockets:
        mgr.disconnect(ws)
    assert mgr.active_connections == []


@pytest.mark.asyncio
async def test_connection_list_never_grows_unbounded():
    """Connect/disconnect 100 times → list never exceeds 1 element."""
    mgr = ConnectionManager()

    for _ in range(100):
        ws = _make_mock_ws()
        await mgr.connect(ws)
        assert len(mgr.active_connections) <= 1 + 0  # at most 1 active
        mgr.disconnect(ws)

    assert len(mgr.active_connections) == 0


@pytest.mark.asyncio
async def test_concurrent_connects_all_tracked():
    """Multiple concurrent connects should all be tracked."""
    mgr = ConnectionManager()
    sockets = [_make_mock_ws() for _ in range(10)]
    await asyncio.gather(*(mgr.connect(ws) for ws in sockets))
    assert len(mgr.active_connections) == 10

    for ws in sockets:
        mgr.disconnect(ws)
    assert len(mgr.active_connections) == 0


@pytest.mark.asyncio
async def test_broadcast_reaches_all_connected():
    mgr = ConnectionManager()
    sockets = [_make_mock_ws() for _ in range(3)]
    for ws in sockets:
        await mgr.connect(ws)

    await mgr.broadcast("ping")
    for ws in sockets:
        ws.send_text.assert_awaited_once_with("ping")

    for ws in sockets:
        mgr.disconnect(ws)


@pytest.mark.asyncio
async def test_broadcast_continues_past_dead_connection():
    """CLA-261: A dead connection mid-broadcast must not prevent remaining connections from receiving."""
    mgr = ConnectionManager()
    ws_good_1 = _make_mock_ws()
    ws_dead = _make_mock_ws()
    ws_dead.send_text = AsyncMock(side_effect=RuntimeError("connection reset"))
    ws_good_2 = _make_mock_ws()

    await mgr.connect(ws_good_1)
    await mgr.connect(ws_dead)
    await mgr.connect(ws_good_2)

    await mgr.broadcast("hello")

    ws_good_1.send_text.assert_awaited_once_with("hello")
    ws_good_2.send_text.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_broadcast_removes_dead_connection():
    """CLA-261: A connection that errors during broadcast should be evicted from active_connections."""
    mgr = ConnectionManager()
    ws_good = _make_mock_ws()
    ws_dead = _make_mock_ws()
    ws_dead.send_text = AsyncMock(side_effect=RuntimeError("connection reset"))

    await mgr.connect(ws_good)
    await mgr.connect(ws_dead)
    assert len(mgr.active_connections) == 2

    await mgr.broadcast("hello")

    assert ws_dead not in mgr.active_connections
    assert ws_good in mgr.active_connections
    assert len(mgr.active_connections) == 1


# ---------------------------------------------------------------------------
# Voice chat temp file cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voice_temp_files_cleaned_on_success():
    """After a successful transcription, the .webm temp file is removed via os.remove."""
    fake_path = "/tmp/_test_voice_success.webm"
    fake_wav = "/tmp/_test_voice_success.wav"

    mock_tmp_file = MagicMock()
    mock_tmp_file.name = fake_path
    mock_tmp_file.write = MagicMock()
    mock_tmp_file.flush = MagicMock()

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "hello"
    mock_model.transcribe.return_value = [mock_segment]

    mock_ntf = MagicMock()
    mock_ntf.__enter__ = MagicMock(return_value=mock_tmp_file)
    mock_ntf.__exit__ = MagicMock(return_value=False)

    with (
        patch("app.api.voice_chat._VOICE_ENABLED", True),
        patch("app.api.voice_chat.whisper_model", mock_model),
        patch("app.api.voice_chat.tempfile.NamedTemporaryFile", return_value=mock_ntf),
        patch("subprocess.run"),
        patch("app.api.voice_chat.sha256sum", return_value="abc"),
        patch("app.api.voice_chat.os.remove") as mock_remove,
    ):
        from app.api.voice_chat import voice_input

        mock_upload = MagicMock()
        mock_upload.read = AsyncMock(return_value=b"\x1a\x45\xdf\xa3")

        result = await voice_input(file=mock_upload)

    assert fake_path in [c.args[0] for c in mock_remove.call_args_list]
    assert result == {"transcript": "hello"}


@pytest.mark.asyncio
async def test_voice_webm_cleaned_when_write_fails():
    """If tmp.write() or sha256sum raises inside the `with` block, the .webm must still be removed."""
    fake_path = "/tmp/_test_voice_write_fail.webm"

    mock_tmp_file = MagicMock()
    mock_tmp_file.name = fake_path
    mock_tmp_file.write = MagicMock(side_effect=IOError("disk full"))
    mock_tmp_file.flush = MagicMock()

    mock_ntf = MagicMock()
    mock_ntf.__enter__ = MagicMock(return_value=mock_tmp_file)
    mock_ntf.__exit__ = MagicMock(return_value=False)

    with (
        patch("app.api.voice_chat._VOICE_ENABLED", True),
        patch("app.api.voice_chat.whisper_model", MagicMock()),
        patch("app.api.voice_chat.tempfile.NamedTemporaryFile", return_value=mock_ntf),
        patch("app.api.voice_chat.os.remove") as mock_remove,
    ):
        from app.api.voice_chat import voice_input

        mock_upload = MagicMock()
        mock_upload.read = AsyncMock(return_value=b"\x1a\x45\xdf\xa3")

        resp = await voice_input(file=mock_upload)

    assert fake_path in [c.args[0] for c in mock_remove.call_args_list], \
        f".webm leaked on write error; os.remove called with: {[c.args[0] for c in mock_remove.call_args_list]}"
    assert "error" in resp


@pytest.mark.asyncio
async def test_voice_webm_cleaned_when_read_fails():
    """If file.read() raises, the temp file (already created with delete=False) must be cleaned up."""
    fake_path = "/tmp/_test_voice_read_fail.webm"

    mock_tmp_file = MagicMock()
    mock_tmp_file.name = fake_path
    mock_tmp_file.write = MagicMock()
    mock_tmp_file.flush = MagicMock()

    mock_ntf = MagicMock()
    mock_ntf.__enter__ = MagicMock(return_value=mock_tmp_file)
    mock_ntf.__exit__ = MagicMock(return_value=False)

    with (
        patch("app.api.voice_chat._VOICE_ENABLED", True),
        patch("app.api.voice_chat.whisper_model", MagicMock()),
        patch("app.api.voice_chat.tempfile.NamedTemporaryFile", return_value=mock_ntf),
        patch("app.api.voice_chat.os.remove") as mock_remove,
    ):
        from app.api.voice_chat import voice_input

        mock_upload = MagicMock()
        mock_upload.read = AsyncMock(side_effect=IOError("connection reset"))

        resp = await voice_input(file=mock_upload)

    assert fake_path in [c.args[0] for c in mock_remove.call_args_list], \
        f".webm leaked on read error; os.remove called with: {[c.args[0] for c in mock_remove.call_args_list]}"
    assert "error" in resp


@pytest.mark.asyncio
async def test_voice_wav_file_cleaned_on_success():
    """CLA-260: The .wav file produced by ffmpeg must also be removed after transcription."""
    fake_path = "/tmp/_test_voice_wav_success.webm"
    fake_wav = fake_path.replace(".webm", ".wav")

    mock_tmp_file = MagicMock()
    mock_tmp_file.name = fake_path
    mock_tmp_file.write = MagicMock()
    mock_tmp_file.flush = MagicMock()

    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "hello"
    mock_model.transcribe.return_value = [mock_segment]

    mock_ntf = MagicMock()
    mock_ntf.__enter__ = MagicMock(return_value=mock_tmp_file)
    mock_ntf.__exit__ = MagicMock(return_value=False)

    with (
        patch("app.api.voice_chat._VOICE_ENABLED", True),
        patch("app.api.voice_chat.whisper_model", mock_model),
        patch("app.api.voice_chat.tempfile.NamedTemporaryFile", return_value=mock_ntf),
        patch("subprocess.run"),
        patch("app.api.voice_chat.sha256sum", return_value="abc"),
        patch("app.api.voice_chat.os.remove") as mock_remove,
    ):
        from app.api.voice_chat import voice_input

        mock_upload = MagicMock()
        mock_upload.read = AsyncMock(return_value=b"\x1a\x45\xdf\xa3")

        result = await voice_input(file=mock_upload)

    remove_calls = [c.args[0] for c in mock_remove.call_args_list]
    assert fake_wav in remove_calls, f".wav not cleaned up; os.remove called with: {remove_calls}"
    assert fake_path in remove_calls, f".webm not cleaned up; os.remove called with: {remove_calls}"
    assert result == {"transcript": "hello"}


@pytest.mark.asyncio
async def test_voice_wav_file_cleaned_on_error():
    """CLA-260: Even when transcription raises, both .webm and .wav must be removed."""
    fake_path = "/tmp/_test_voice_wav_error.webm"
    fake_wav = fake_path.replace(".webm", ".wav")

    mock_tmp_file = MagicMock()
    mock_tmp_file.name = fake_path
    mock_tmp_file.write = MagicMock()
    mock_tmp_file.flush = MagicMock()

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("boom")

    mock_ntf = MagicMock()
    mock_ntf.__enter__ = MagicMock(return_value=mock_tmp_file)
    mock_ntf.__exit__ = MagicMock(return_value=False)

    with (
        patch("app.api.voice_chat._VOICE_ENABLED", True),
        patch("app.api.voice_chat.whisper_model", mock_model),
        patch("app.api.voice_chat.tempfile.NamedTemporaryFile", return_value=mock_ntf),
        patch("subprocess.run"),
        patch("app.api.voice_chat.sha256sum", return_value="abc"),
        patch("app.api.voice_chat.os.remove") as mock_remove,
    ):
        from app.api.voice_chat import voice_input

        mock_upload = MagicMock()
        mock_upload.read = AsyncMock(return_value=b"\x1a\x45\xdf\xa3")

        resp = await voice_input(file=mock_upload)

    remove_calls = [c.args[0] for c in mock_remove.call_args_list]
    assert fake_wav in remove_calls, f".wav not cleaned up on error; os.remove called with: {remove_calls}"
    assert fake_path in remove_calls, f".webm not cleaned up on error; os.remove called with: {remove_calls}"
    assert "error" in resp or "STT failed" in str(resp)


@pytest.mark.asyncio
async def test_voice_temp_files_cleaned_on_error():
    """Even when transcription raises, the .webm file should still be removed."""
    fake_path = "/tmp/_test_voice_error.webm"

    mock_tmp_file = MagicMock()
    mock_tmp_file.name = fake_path
    mock_tmp_file.write = MagicMock()
    mock_tmp_file.flush = MagicMock()

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("boom")

    mock_ntf = MagicMock()
    mock_ntf.__enter__ = MagicMock(return_value=mock_tmp_file)
    mock_ntf.__exit__ = MagicMock(return_value=False)

    with (
        patch("app.api.voice_chat._VOICE_ENABLED", True),
        patch("app.api.voice_chat.whisper_model", mock_model),
        patch("app.api.voice_chat.tempfile.NamedTemporaryFile", return_value=mock_ntf),
        patch("subprocess.run"),
        patch("app.api.voice_chat.sha256sum", return_value="abc"),
        patch("app.api.voice_chat.os.remove") as mock_remove,
    ):
        from app.api.voice_chat import voice_input

        mock_upload = MagicMock()
        mock_upload.read = AsyncMock(return_value=b"\x1a\x45\xdf\xa3")

        resp = await voice_input(file=mock_upload)

    assert fake_path in [c.args[0] for c in mock_remove.call_args_list]
    assert "error" in resp or "STT failed" in str(resp)


# ---------------------------------------------------------------------------
# Async session cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_session_closes_after_use():
    """AsyncSessionLocal used as context manager should close the session."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_factory = MagicMock(return_value=mock_session)

    with patch("db.session_async.AsyncSessionLocal", mock_factory):
        from db.session_async import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            pass

    mock_session.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_closes_on_exception():
    """Session __aexit__ should be called even when the body raises."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    mock_factory = MagicMock(return_value=mock_session)

    with patch("db.session_async.AsyncSessionLocal", mock_factory):
        from db.session_async import AsyncSessionLocal

        with pytest.raises(ValueError):
            async with AsyncSessionLocal() as session:
                raise ValueError("intentional")

    mock_session.__aexit__.assert_awaited_once()
