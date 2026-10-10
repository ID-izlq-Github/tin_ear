"""TinEar 测试夹具：合成音频 + 数据目录隔离。

所有测试产物都写在 pytest 的 tmp_path 下；运行时请带 TMPDIR=<项目>/data/tmp
（Makefile 的 make test 已内置），因此不会有任何文件落到系统 temp 或用户目录。
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal as sps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module  # noqa: E402

SR = 48000


# ---------------------------------------------------------------------------
# 合成音频
# ---------------------------------------------------------------------------


def synth_signal(seconds: float, seed: int = 0, sr: int = SR, channels: int = 2) -> np.ndarray:
    """合成"像音乐一样"的立体声测试信号：声部 + 粉噪 + 动态包络 + 打击瞬态。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr

    voice = np.zeros(n)
    for k, (freq, amp) in enumerate(((220.0, 0.5), (277.2, 0.35), (329.6, 0.3))):
        drift = 1.0 + 0.02 * np.sin(2 * np.pi * 0.05 * t + k)
        voice += amp * np.sin(2 * np.pi * freq * np.cumsum(drift) / sr)

    noise = sps.lfilter([1.0], [1.0, -0.95], rng.standard_normal(n))
    period = max(seconds, 1.0)
    env = 0.3 + 0.7 * np.clip(np.sin(2 * np.pi * t / period * 1.5), 0.0, 1.0) ** 1.5
    base = env * (0.7 * voice / (np.max(np.abs(voice)) + 1e-9)
                  + 0.5 * noise / (np.max(np.abs(noise)) + 1e-9))

    # 每 7 秒一串瞬态，让 onset_density 有区分度
    for hit in (np.arange(int(seconds // 7) + 1) * 7 * sr + int(0.3 * sr)):
        hit = int(hit)
        if hit + 400 < n:
            base[hit:hit + 400] += 0.5 * np.exp(-np.arange(400) / 60.0)

    base = base / (np.max(np.abs(base)) + 1e-9) * 0.8
    out = np.zeros((n, channels), dtype=np.float64)
    for channel in range(channels):
        shift = channel  # 右声道延迟 1 个采样，避免完全 dual-mono
        out[shift:, channel] = base[: n - shift]
    return out.astype(np.float32)


def write_source(path: Path, seconds: float, seed: int = 0, subtype: str = "PCM_16") -> Path:
    data = synth_signal(seconds, seed=seed)
    sf.write(str(path), data, SR, subtype=subtype, format="WAV")
    return path


def encode_mp3(src: Path, dst: Path, bitrate: int) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(src),
         *app_module.LAME_EXTRA_ARGS, "-c:a", "libmp3lame", "-b:a", f"{bitrate}k",
         "-f", "mp3", str(dst)],
        check=True, stdin=subprocess.DEVNULL,
    )
    return dst


def decode_f32(path: Path, channels: int = 2, sr: int = SR) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"],
        check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    usable = len(raw) // (4 * channels) * (4 * channels)
    return np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, channels).copy()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("synth")


@pytest.fixture(scope="session")
def long_source(synth_dir: Path) -> Path:
    """65 秒源文件（可切出 seg_0 与 seg_1）。"""
    return write_source(synth_dir / "long.wav", 65.0, seed=1)


@pytest.fixture(scope="session")
def short_source(synth_dir: Path) -> Path:
    """35 秒源文件（只够 seg_0）。"""
    return write_source(synth_dir / "short.wav", 35.0, seed=2)


@pytest.fixture(scope="session")
def segment_pair(synth_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    """15 秒母版片段 + 其 128 kbps MP3 解码版。"""
    master = synth_signal(app_module.SEGMENT_SECONDS, seed=7)
    flac = synth_dir / "seg_master.flac"
    sf.write(str(flac), master, SR, subtype="PCM_16", format="FLAC")
    mp3 = synth_dir / "seg_master.mp3"
    encode_mp3(flac, mp3, 128)
    candidate = decode_f32(mp3)
    return master, candidate, flac


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """把 app 的所有数据目录重定向到本测试自己的临时目录。"""
    root = tmp_path / "data"
    for sub in ("library", "cache", "session", "tmp"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module, "DATA_DIR", root)
    monkeypatch.setattr(app_module, "LIBRARY_DIR", root / "library")
    monkeypatch.setattr(app_module, "CACHE_DIR", root / "cache")
    monkeypatch.setattr(app_module, "SESSION_DIR", root / "session")
    monkeypatch.setattr(app_module, "TMP_DIR", root / "tmp")
    monkeypatch.setattr(app_module, "LIBRARY_INDEX", root / "library" / "index.json")
    previous_tempdir = tempfile.tempdir
    yield root
    app_module.SESSIONS.clear()
    tempfile.tempdir = previous_tempdir


@pytest.fixture
def fast_config(monkeypatch):
    """把题量压到最小，让 API 测试快速跑完（默认值另有单元测试守护）。"""
    monkeypatch.setattr(app_module, "SHORT_TEST_BITRATES", [96])
    monkeypatch.setattr(app_module, "SHORT_TEST_TOTAL", 2)  # 1 真实 + 1 安慰剂
    monkeypatch.setattr(app_module, "DETAILED_TEST_BITRATES", [64, 128, 320])
    monkeypatch.setattr(app_module, "DETAILED_TEST_TOTAL", 5)  # 3 真实 + 2 安慰剂
    return app_module


@pytest.fixture
def client(isolated_data):
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as test_client:
        yield test_client


def upload(client, path: Path) -> dict:
    response = client.post("/api/library/upload", files={"files": (path.name, path.read_bytes())})
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["status"] == "imported", result
    return result


def start_session(client, audio_ids: list[str], mode: str = "short") -> dict:
    response = client.post("/api/start", json={"audio_ids": audio_ids, "mode": mode})
    assert response.status_code == 200, response.text
    return response.json()


def cache_wavs(audio_id: str) -> list[Path]:
    root = app_module._stimuli_dir(audio_id)
    return sorted(root.glob("*.wav"))
