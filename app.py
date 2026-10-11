"""TinEar — 本地音频盲测工具（单文件后端）。

A local blind test for MP3 bitrate perception.

运行：
    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
或：
    python app.py

环境纪律：所有缓存与临时文件都写在项目目录内（data/），除 Python 依赖装在本项目专用
conda 环境 tin_ear 外，不写入任何全局位置。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import secrets
import shutil
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy import ndimage
from scipy import signal as sps
from scipy import stats as sps_stats

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LIBRARY_DIR = DATA_DIR / "library"
CACHE_DIR = DATA_DIR / "cache"
SESSION_DIR = DATA_DIR / "session"
TMP_DIR = DATA_DIR / "tmp"
STATIC_DIR = PROJECT_ROOT / "static"
LIBRARY_INDEX = LIBRARY_DIR / "index.json"

# ---------------------------------------------------------------------------
# 常量（AGENTS.md §4，新增项已标注）
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000
CHANNELS = 2
SEGMENT_SECONDS = 15
SEGMENT_STEP_SECONDS = 5
SEGMENT_SAMPLES = SAMPLE_RATE * SEGMENT_SECONDS  # 720000 = 恰好 625 个 MP3 帧
SEGMENT_FADE_SECONDS = 0.02
SEGMENT_ALGO_VERSION = 1
STIMULI_ALGO_VERSION = 3  # 测试音频缓存版本：任何改变其内容的常量/参数都要 bump
                          # （v2：TARGET_LUFS -23 → -18；v3：STIMULUS_SUBTYPE FLOAT → PCM_24）
#                           （v2：TARGET_LUFS 由 -23 提到 -18 LUFS）

TARGET_LUFS = -18.0
TRUE_PEAK_DBTP = -1.0
MAX_LUFS_DIFF_LU = 0.05  # 测试断言用（协议上限 0.3）
TRUE_PEAK_SLACK_DB = 0.05  # 浮点容差

SHORT_TEST_BITRATES = [48, 96, 160, 224, 320]
DETAILED_TEST_BITRATES = [32, 48, 64, 80, 96, 128, 160, 192, 256, 320]
SHORT_TEST_TOTAL = 6  # 5 真实 + 1 安慰剂（快速模式：每档 1 题）
DETAILED_TEST_TOTAL = 22  # 20 真实 + 2 安慰剂（详细模式：每档 2 题）
SESSION_TTL_SECONDS = 7200

# 新增
ANALYSIS_SR = 16000
ANALYSIS_MAX_SECONDS = 1200  # 只分析前 20 分钟，约束内存
STFT_N = 1024
STFT_HOP = 256
LOUDNESS_BLOCK_MS = 400
LOUDNESS_HOP_MS = 100
SILENCE_ABS_DB = -60.0
SILENCE_REL_DB = 40.0
MIN_SECONDS_FOR_SECOND_SEGMENT = 60.0  # AGENTS.md §6：时长 ≥ 60 秒才取 seg_1
ONSET_MEDIAN_MS = 1000
ONSET_FACTOR = 1.5
ONSET_MIN_SEPARATION_S = 0.05
SEG_WEIGHTS = {
    "loudness_std_db": 1.0,
    "spectral_flux": 1.0,
    "onset_density": 1.0,
    "silence_ratio": 1.0,
}
ALIGN_PROBE_SECONDS = 6
ALIGN_REFINE_SECONDS = 8
ALIGN_MAX_LAG = 8192
ALIGN_REFINE_SPAN = 64
# 交付给浏览器的 WAV 位深。内容不变，只影响磁盘/传输：
#   FLOAT   5,760,088 B/文件，逐位透明，但带一个含时间戳的 PEAK chunk；
#   PCM_24  4,320,044 B（-25%），量化噪声底 ≈ -149 dBFS；
#   PCM_16  2,880,044 B（-50%），量化噪声底 ≈ -101 dBFS（= CD 精度）。
# 对照：32 kbps MP3 的伪影在 -40 dBFS 上下，所以 24 与 16 bit 都不可能成为线索；
# 选 24 而不是 16 是为了不动"参考侧"的精度，也是为了 24-bit WAV 在浏览器里少踩坑。
# 换这个常量必须 bump STIMULI_ALGO_VERSION（内容变了），并重跑 MEASUREMENTS.md 的体积/自检数字。
STIMULUS_SUBTYPE = "PCM_24"
STIMULUS_BITS = {"FLOAT": 32, "PCM_24": 24, "PCM_16": 16}
STIMULI_CONCURRENCY_MAX = 8  # 单题生成实测 ~0.70 s，8 路之后基本饱和（见 MEASUREMENTS.md）


def _detect_cpu_count() -> int:
    """可用核数：优先亲和集（Linux/WSL/容器里比 os.cpu_count() 准），回退 cpu_count。"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows 没有这个函数
        return os.cpu_count() or 4


def _resolve_concurrency() -> int:
    """测试音频生成的并发数：按机器核数自适应，封顶 STIMULI_CONCURRENCY_MAX。

    实测（本机 24 核、detailed 22 题冷启动）：1 路 15.3 s / 4 路 5.0 s / 8 路 3.9 s /
    12 路 3.5 s —— 8 之后收益很小，低核机器取核数即可。
    `TINEAR_CONCURRENCY` 环境变量可覆盖（排查问题与测试用）。
    """
    override = os.environ.get("TINEAR_CONCURRENCY", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)
    return max(1, min(_detect_cpu_count(), STIMULI_CONCURRENCY_MAX))


STIMULI_CONCURRENCY = _resolve_concurrency()
FFMPEG_CONCURRENCY = STIMULI_CONCURRENCY  # ffmpeg 子进程槽位不再比任务槽位更窄
MAX_UPLOAD_BYTES = 2 * 1024**3
LOSSLESS_CODECS = {"flac", "alac", "wavpack", "tta"}
MIN_SOURCE_SAMPLE_RATE = 44100
MIN_SOURCE_SECONDS = 30
LAME_EXTRA_ARGS = [
    "-write_xing", "1",
    "-id3v2_version", "0",
    "-write_id3v1", "0",
    "-map_metadata", "-1",
]
SHUFFLE_ALL_QUESTIONS = True  # 真实题与安慰剂题统一打乱位置，避免用「第几题」预判
PLACEBO_SAME_SEGMENT = True  # 默认：安慰剂 = 同一片段的两份无损拷贝
PREWARM_SHORT_COMPLEMENT = True  # 详细模式顺手生成"快速模式才需要的那几对"（快速 → 详细仍需补 17 个：
#                                  详细是 10 档 × 2 片段，快速只覆盖其中 5 个组合，这部分无法靠预热消除）。
#                                  代价：每首多 1～2 对（约 11 MB），换来"详细 → 快速"也 0.01 s。
RESULT_BRACKETS = [(0, 96, "≤96 kbps"), (128, 192, "128–192 kbps"), (224, 10**9, "≥224 kbps")]

# 统计口径（AGENTS.md §4.1）
CHANCE_LEVEL = 1.0 / 3.0  # 三选一（A/B/听不出）的瞎猜基线，不是 1/2
ALPHA = 0.05  # 单侧精确二项检验的显著性水平
POWER_TARGET = 0.8  # 期望的检验功效（用于算"每档需要多少题"）
P1_FOR_POWER = 0.8  # 计算功效时假定的真实听出率
MIN_N_FOR_SIGNIFICANCE = 3  # 少于 3 题任何结果都无法判定（n=1 的最小 p 就是 1/3）


class AudioToolError(RuntimeError):
    """ffmpeg/音频处理失败。"""


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text("utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    os.replace(tmp, path)


def _dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=20,
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        raise RuntimeError(
            "找不到可用的 ffmpeg，请先安装（本项目只依赖系统已有的 ffmpeg/ffprobe）"
        ) from exc
    first = out.splitlines()[0] if out else "ffmpeg"
    return f"libmp3lame/{first.strip()}"


_ENCODER_TAG = _ffmpeg_version()

_FFMPEG_SEM = asyncio.Semaphore(FFMPEG_CONCURRENCY)
_LIBRARY_LOCK = asyncio.Lock()
# 共享资源的锁按「事件循环 + key」存放，见 _loop_scoped_lock 的说明
_CACHE_INDEX_LOCKS: dict[tuple[object, str], asyncio.Lock] = {}


def _loop_scoped_lock(box: dict[tuple[object, str], asyncio.Lock], key: str) -> asyncio.Lock:
    """取「当前事件循环 + key」对应的一把锁。

    为什么不能直接用模块级 `asyncio.Lock`：它一旦被**争用**（真的有人等）就会绑定到当时的
    事件循环，之后在另一个循环里再争用会抛 `... is bound to a different event loop`。
    生产环境只有一个循环，但 pytest 每个用例都用新的循环跑 TestClient，跨用例复用同一把锁
    就会炸；按循环分桶既保留串行语义，也不会把上一个循环的锁带过来。
    """
    loop = asyncio.get_running_loop()
    full_key = (loop, key)
    lock = box.get(full_key)
    if lock is None:
        lock = asyncio.Lock()
        box[full_key] = lock
    return lock


async def _run_process(args: list[str], capture_stdout: bool = False):
    async with _FFMPEG_SEM:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await proc.communicate()
        except asyncio.CancelledError:
            # 并发准备中途失败会取消其余任务：别把 ffmpeg 子进程留成孤儿
            proc.kill()
            await proc.wait()
            raise
    return proc.returncode, out or b"", err or b""


async def _ffmpeg(args: list[str], capture_stdout: bool = False) -> bytes:
    rc, out, err = await _run_process(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *args], capture_stdout
    )
    if rc != 0:
        detail = err.decode("utf-8", "replace").strip()[:400]
        raise AudioToolError(f"ffmpeg 失败 (rc={rc}): {detail}")
    return out


async def _probe(path: Path) -> dict:
    rc, out, err = await _run_process(
        [
            "ffprobe", "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels,duration:format=duration",
            "-of", "json", str(path),
        ],
        capture_stdout=True,
    )
    if rc != 0:
        raise AudioToolError("无法解析音频流，可能不是音频文件或文件已损坏")
    try:
        data = json.loads(out.decode("utf-8", "replace"))
    except json.JSONDecodeError as exc:
        raise AudioToolError("无法解析音频流信息") from exc
    streams = data.get("streams") or []
    if not streams:
        raise AudioToolError("文件不含音频流")
    stream = streams[0]
    duration = stream.get("duration") or (data.get("format") or {}).get("duration")
    try:
        duration_f = float(duration)
    except (TypeError, ValueError):
        duration_f = 0.0
    return {
        "codec_name": (stream.get("codec_name") or "").lower(),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "duration": duration_f,
    }


async def _ffmpeg_to_array(path: Path, sr: int, channels: int) -> np.ndarray:
    raw = await _ffmpeg(
        ["-i", str(path), "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"],
        capture_stdout=True,
    )
    frame = 4 * channels
    usable = len(raw) // frame * frame
    if usable == 0:
        raise AudioToolError("解码结果为空")
    return np.frombuffer(raw[:usable], dtype="<f4").reshape(-1, channels).copy()


# ---------------------------------------------------------------------------
# 音频基础操作
# ---------------------------------------------------------------------------


def _read_audio(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise AudioToolError(f"内部音频采样率异常：{sr} != {SAMPLE_RATE}")
    return np.ascontiguousarray(data)


def _write_wav(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, SAMPLE_RATE, subtype=STIMULUS_SUBTYPE, format="WAV")


def _write_wav_atomic(path: Path, data: np.ndarray) -> None:
    """先写临时文件再原子替换（`os.replace`）。

    并发生成时，同一片段的多个码率会写**同一个**共享无损文件（见 `_ensure_stimuli`）；
    直接往目标路径写，后到的任务可能截断别人刚写完的文件、或让检查方读到半截数据。
    内容按构造就是逐位相同的，所以谁最后到都一样。
    """
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    _write_wav(tmp, data)
    os.replace(tmp, path)


def _stimulus_lsb() -> float:
    """当前 `STIMULUS_SUBTYPE` 的量化台阶（满刻度 = 1.0；浮点返回 0.0）。"""
    bits = STIMULUS_BITS.get(STIMULUS_SUBTYPE)
    if bits is None:
        raise AudioToolError(f"未知的 STIMULUS_SUBTYPE：{STIMULUS_SUBTYPE!r}")
    return 0.0 if STIMULUS_SUBTYPE == "FLOAT" else 2.0 ** (1 - bits)


def _verify_written_pair(path_ll: Path, path_mp3: Path, out_ll: np.ndarray, out_mp3: np.ndarray) -> None:
    """以**落盘文件**为准做结构自检（整数位深下内存数组 != 交付字节）。

    `_prepare_pair` 的自检测的是内存里的 float 数组（长度、ΔLUFS、真峰值）。改成整数位深后，
    交付的字节是那份数组的量化结果，所以这里补三项关于**文件**的硬事实：

    1. 子类型确实是我们要的（防 libsndfile 静默降级）；
    2. 两路采样数与**字节数**完全相等（验收 #8 的"无体积泄漏"靠的就是这一条）；
    3. 每个采样与已通过自检的数组相差 <= 1.5 LSB（只可能是量化，不可能夹带别的处理）。

    有了 3 就能推出交付文件仍然满足 `_prepare_pair` 的全部结论：24 bit 的 1 LSB = 2^-23 ≈ -138 dBFS，
    经 4× 过采样核（L1 范数略大于 1）放大后仍在 -130 dBFS 量级，比 0.05 dB / -1 dBTP 的容差低三个
    数量级。**这套推断由 `test_stimulus_metadata_matches_written_files` 实测把关**（它把落盘文件
    重新解码后重算 LUFS/真峰值，与索引里的记录比对），所以这里不再重复付一次 ~182 ms/题的测量开销
    （那会让冷启动从 4.8 s 涨到 6.0 s）。
    """
    tolerance = 1.5 * _stimulus_lsb() + 1e-6  # 浮点子类型走 1e-6（实际逐位相等）
    for path, source in ((path_ll, out_ll), (path_mp3, out_mp3)):
        info = sf.info(str(path))
        if info.subtype != STIMULUS_SUBTYPE:
            raise AudioToolError(f"落盘子类型异常：{info.subtype} != {STIMULUS_SUBTYPE}")
        if info.frames != SEGMENT_SAMPLES:
            raise AudioToolError(f"落盘长度异常：{info.frames} != {SEGMENT_SAMPLES}")
        disk, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr != SAMPLE_RATE:
            raise AudioToolError(f"落盘采样率异常：{sr} != {SAMPLE_RATE}")
        reference = np.asarray(_fit_length(source)[0], dtype=np.float64)
        drift = float(np.max(np.abs(np.asarray(disk, dtype=np.float64) - reference)))
        if drift > tolerance:
            raise AudioToolError(
                f"落盘内容与已自检数组相差 {drift:.3e}（容差 {tolerance:.3e}），不只是量化误差"
            )
    if path_ll.stat().st_size != path_mp3.stat().st_size:
        raise AudioToolError(
            f"两路字节数不等（{path_ll.stat().st_size} vs {path_mp3.stat().st_size}），会从体积泄漏答案"
        )


def _integrated_loudness(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    try:
        return float(pyloudnorm.Meter(sr).integrated_loudness(x))
    except Exception:
        # 极安静/接近静音的信号：pyloudnorm 会拒绝，退回 RMS 近似，避免整条流程失败
        rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
        return 20.0 * math.log10(max(rms, 1e-12)) - 0.691


def _true_peak_db(x: np.ndarray) -> float:
    up = sps.resample_poly(x, 4, 1, axis=0)
    return 20.0 * math.log10(max(float(np.max(np.abs(up))), 1e-12))


def _fit_length(x: np.ndarray, n: int = SEGMENT_SAMPLES) -> tuple[np.ndarray, int]:
    if len(x) >= n:
        return x[:n], 0
    pad = n - len(x)
    return np.pad(x, ((0, pad), (0, 0))), pad


def _shifted(x: np.ndarray, lag: int) -> np.ndarray:
    """把 x 平移 lag 个采样（lag>0 表示 x 比参考晚，需要丢弃开头）。"""
    if lag >= 0:
        return x[lag:]
    return np.concatenate([np.zeros((-lag, x.shape[1]), dtype=x.dtype), x], axis=0)


def _estimate_lag(cand: np.ndarray, master: np.ndarray) -> int:
    """估计 cand 相对 master 的延迟（采样数）。先 FFT 互相关粗估，再逐点精修。

    约定：返回值 lag > 0 表示 cand 比 master 晚 lag 个采样，即 cand[i] ≈ master[i - lag]。
    """
    probe = min(int(ALIGN_PROBE_SECONDS * SAMPLE_RATE), len(master), len(cand))
    if probe <= 0:
        return 0
    ref = master[:probe].mean(axis=1).astype(np.float64)
    span = min(len(cand), probe + 2 * ALIGN_MAX_LAG)
    seg = cand[:span].mean(axis=1).astype(np.float64)
    corr = sps.correlate(seg, ref, mode="full", method="fft")
    coarse = int(np.argmax(corr)) - (len(ref) - 1)
    coarse = int(np.clip(coarse, -ALIGN_MAX_LAG, ALIGN_MAX_LAG))

    # 精修：在粗估附近用更长的窗口最小化平均绝对差；正负延迟都要正确覆盖
    n_ref = min(len(master), int(ALIGN_REFINE_SECONDS * SAMPLE_RATE))
    ref_mono = master[:n_ref].mean(axis=1).astype(np.float64)
    cand_mono = cand.mean(axis=1).astype(np.float64)
    min_overlap = max(2048, n_ref // 4)
    best_lag, best_cost = coarse, None
    for lag in range(coarse - ALIGN_REFINE_SPAN, coarse + ALIGN_REFINE_SPAN + 1):
        # 与 _shifted(cand, lag) 一致：aligned[k] = cand[k + lag]，比较 aligned[k] 与 master[k]
        lo = max(0, -lag)
        hi = min(n_ref, len(cand_mono) - lag)
        if hi - lo < min_overlap:
            continue
        cost = float(np.mean(np.abs(cand_mono[lo + lag:hi + lag] - ref_mono[lo:hi])))
        if best_cost is None or cost < best_cost:
            best_cost, best_lag = cost, lag
    return int(np.clip(best_lag, -ALIGN_MAX_LAG, ALIGN_MAX_LAG))


def _prepare_pair(master: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """对齐 + 等响度 + 峰值余量，返回 (无损版, 编码版, 指标)。

    响度策略（对 AGENTS.md §6 的修正）：
      1. 两路各自按 integrated loudness 独立增益，使两者都落在 TARGET_LUFS —— 于是
         LUFS 差**恒为 0**，不留任何响度线索。
      2. 再对**这一对**施加同一个线性衰减 delta ≤ 0，把最坏一路的 True Peak 压到
         TRUE_PEAK_DBTP 以下。delta 是共同线性增益，不改变两者响度差，也**不使用限幅器**
         （限幅是非线性处理，会在低码率引入与 MP3 无关的失真线索）。
    """
    lag = _estimate_lag(cand, master)
    aligned, padding = _fit_length(_shifted(cand, lag))
    master_fit, master_padding = _fit_length(master)

    peak_in = float(np.max(np.abs(master_fit)))
    if peak_in < 1e-6:
        raise AudioToolError("片段是数字静音，无法生成盲测音频（请检查源文件）")

    lufs_ll = _integrated_loudness(master_fit)
    lufs_mp3 = _integrated_loudness(aligned)
    tp_ll_0 = _true_peak_db(master_fit)
    tp_mp3_0 = _true_peak_db(aligned)
    for label, value in (("LUFS(master)", lufs_ll), ("LUFS(mp3)", lufs_mp3),
                         ("TP(master)", tp_ll_0), ("TP(mp3)", tp_mp3_0)):
        if not math.isfinite(value):
            raise AudioToolError(f"{label} 计算得到非有限值（{value}），无法继续")

    g_ll = TARGET_LUFS - lufs_ll
    g_mp3 = TARGET_LUFS - lufs_mp3
    delta = min(0.0, TRUE_PEAK_DBTP - max(tp_ll_0 + g_ll, tp_mp3_0 + g_mp3))

    out_ll = (master_fit * (10.0 ** ((g_ll + delta) / 20.0))).astype(np.float32)
    out_mp3 = (aligned * (10.0 ** ((g_mp3 + delta) / 20.0))).astype(np.float32)

    lu_a = _integrated_loudness(out_ll)
    lu_b = _integrated_loudness(out_mp3)
    tp_a = _true_peak_db(out_ll)
    tp_b = _true_peak_db(out_mp3)

    info = {
        "lag_samples": lag,
        "padding_samples": padding + master_padding,
        "lufs": {"ll": round(lu_a, 4), "mp3": round(lu_b, 4)},
        "lufs_diff": round(abs(lu_a - lu_b), 6),
        "true_peak_db": round(max(tp_a, tp_b), 4),
        "gain_db": {"ll": round(g_ll + delta, 4), "mp3": round(g_mp3 + delta, 4), "common_delta": round(delta, 4)},
        "n_samples": int(len(out_ll)),
    }
    for key, value in (("lufs.ll", lu_a), ("lufs.mp3", lu_b), ("true_peak_db", max(tp_a, tp_b))):
        if not math.isfinite(value):
            raise AudioToolError(f"{key} 非有限值（{value}）")
    if info["n_samples"] != SEGMENT_SAMPLES:
        raise AudioToolError(f"测试音频长度异常：{info['n_samples']} != {SEGMENT_SAMPLES}")
    if info["lufs_diff"] > MAX_LUFS_DIFF_LU:
        raise AudioToolError(f"两路响度差过大：{info['lufs_diff']} LU")
    if info["true_peak_db"] > TRUE_PEAK_DBTP + TRUE_PEAK_SLACK_DB:
        raise AudioToolError(f"峰值超标：{info['true_peak_db']} dBTP")
    return out_ll, out_mp3, info


# ---------------------------------------------------------------------------
# 音频库
# ---------------------------------------------------------------------------


def _library() -> dict:
    data = _read_json(LIBRARY_INDEX, None)
    if not isinstance(data, dict) or "items" not in data:
        data = {"version": 1, "items": []}
    return data


def _library_item(audio_id: str) -> dict:
    for item in _library()["items"]:
        if item["audio_id"] == audio_id:
            return item
    raise HTTPException(status_code=404, detail="音频不存在")


def _audio_path(audio_id: str) -> Path:
    return LIBRARY_DIR / f"{audio_id}.flac"


def _import_reject(codec: str, sample_rate: int, duration: float) -> str | None:
    if codec not in LOSSLESS_CODECS and not codec.startswith("pcm_"):
        return f"有损或不受支持的编码（{codec or '未知'}），请使用 FLAC / WAV / AIFF 等无损格式"
    if sample_rate < MIN_SOURCE_SAMPLE_RATE:
        return f"采样率 {sample_rate} Hz 低于 {MIN_SOURCE_SAMPLE_RATE} Hz"
    if duration < MIN_SOURCE_SECONDS:
        return f"时长 {duration:.1f} 秒，至少需要 {MIN_SOURCE_SECONDS} 秒"
    return None


async def _spool_upload(upload: UploadFile, dest: Path) -> tuple[int, str | None]:
    size = 0
    with dest.open("wb") as fh:
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                fh.close()
                dest.unlink(missing_ok=True)
                return size, f"文件超过 {MAX_UPLOAD_BYTES // (1024**3)} GB 上限"
            fh.write(chunk)
    return size, None


async def _import_one(upload: UploadFile) -> dict:
    raw_name = (upload.filename or "").replace("\\", "/")
    name = Path(raw_name).name or "unnamed"
    result = {"filename": name, "status": "rejected", "audio_id": None, "reason": None}
    suffix = Path(name).suffix[:12].lower() or ".bin"
    src = TMP_DIR / f"upload_{secrets.token_hex(8)}{suffix}"
    try:
        _, too_big = await _spool_upload(upload, src)
        if too_big:
            result["reason"] = too_big
            return result
        if src.stat().st_size == 0:
            result["reason"] = "文件为空"
            return result
        try:
            probe = await _probe(src)
        except AudioToolError as exc:
            result["reason"] = str(exc)
            return result
        reason = _import_reject(probe["codec_name"], probe["sample_rate"], probe["duration"])
        if reason:
            result["reason"] = reason
            return result

        converted = TMP_DIR / f"conv_{secrets.token_hex(8)}.flac"
        try:
            # -map 0:a:0：只取音频流。否则内嵌封面（attached_pic 视频流）会被一起拷进库内 FLAC，
            # 后续切片段/编码 MP3 时 mp3 muxer 会尝试写 ID3v2 APIC 而我们的 ID3v2 是关闭的 → rc=234。
            await _ffmpeg(
                [
                    "-y", "-i", str(src), "-map", "0:a:0", "-map_metadata", "-1",
                    "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
                    "-c:a", "flac", "-compression_level", "5", str(converted),
                ]
            )
            digest = hashlib.sha256(converted.read_bytes()).hexdigest()
            audio_id = digest[:16]

            async with _LIBRARY_LOCK:
                lib = _library()
                if any(
                    item["sha256"] == digest or item["audio_id"] == audio_id
                    for item in lib["items"]
                ):
                    result["status"] = "duplicate"
                    result["audio_id"] = audio_id
                    result["reason"] = "同一音频已在库中（按内容哈希去重）"
                    return result
                item = {
                    "audio_id": audio_id,
                    "sha256": digest,
                    "display_name": name,
                    "duration": round(probe["duration"], 3),
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                    "size_bytes": converted.stat().st_size,
                    "source_codec": probe["codec_name"],
                    "source_sample_rate": probe["sample_rate"],
                    "source_channels": probe["channels"],
                    "added_at": _now_iso(),
                }
                os.replace(converted, _audio_path(audio_id))
                lib["items"].append(item)
                _write_json(LIBRARY_INDEX, lib)
        finally:
            converted.unlink(missing_ok=True)
        result["status"] = "imported"
        result["audio_id"] = audio_id
        return result
    except AudioToolError as exc:
        result["reason"] = str(exc)
        return result
    finally:
        src.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 片段选择
# ---------------------------------------------------------------------------


def _frames_in_window(start: float, end: float, hop_s: float, frame_s: float, limit: int) -> tuple[int, int]:
    first = max(0, int(math.ceil(start / hop_s - 1e-9)))
    last = min(limit - 1, int(math.floor((end - frame_s) / hop_s + 1e-9)))
    if last < first:
        last = first
    return first, last


def _block_rms_db(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    block = int(ANALYSIS_SR * LOUDNESS_BLOCK_MS / 1000)
    hop = int(ANALYSIS_SR * LOUDNESS_HOP_MS / 1000)
    ratio = block // hop
    assert ratio * hop == block, "LOUDNESS_BLOCK_MS 必须是 LOUDNESS_HOP_MS 的整数倍"
    n_full = len(x) // hop
    if n_full < ratio:
        return np.array([-120.0]), hop / ANALYSIS_SR, block / ANALYSIS_SR
    usable = x[: n_full * hop]
    squared = np.square(usable, dtype=np.float32)
    hop_energy = np.add.reduceat(squared, np.arange(0, usable.size, hop)).astype(np.float64)
    energy = np.convolve(hop_energy, np.ones(ratio), mode="valid")
    rms = np.sqrt(np.maximum(energy / block, 1e-24))
    return 20.0 * np.log10(rms), hop / ANALYSIS_SR, block / ANALYSIS_SR


def _spectral_flux(x: np.ndarray) -> np.ndarray:
    win = np.hanning(STFT_N).astype(np.float32)
    if len(x) < STFT_N:
        return np.zeros(1, dtype=np.float32)
    n_frames = 1 + (len(x) - STFT_N) // STFT_HOP
    view = np.lib.stride_tricks.sliding_window_view(x, STFT_N)[::STFT_HOP][:n_frames]
    flux = np.zeros(n_frames, dtype=np.float32)
    chunk = 2048
    prev_log: np.ndarray | None = None
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        frames = view[start:stop] * win
        spec = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
        logspec = np.log1p(spec)
        if prev_log is not None:
            flux[start] = float(np.maximum(logspec[0] - prev_log, 0.0).sum())
        if stop - start > 1:
            diff = logspec[1:] - logspec[:-1]
            np.maximum(diff, 0.0, out=diff)
            flux[start + 1:stop] = diff.sum(axis=1)
        prev_log = logspec[-1].copy()
    return flux


def _onset_flags(flux: np.ndarray, hop_s: float) -> np.ndarray:
    if flux.size < 3:
        return np.zeros_like(flux, dtype=bool)
    window = int(round(ONSET_MEDIAN_MS / 1000.0 / hop_s)) | 1
    base = ndimage.median_filter(flux, size=max(3, window), mode="nearest")
    threshold = base * ONSET_FACTOR + 1e-6
    peaks = (flux > threshold) & (flux >= np.roll(flux, 1)) & (flux > np.roll(flux, -1))
    flags = np.zeros_like(peaks)
    min_sep = max(1, int(round(ONSET_MIN_SEPARATION_S / hop_s)))
    last = -10**9
    for idx in np.flatnonzero(peaks):
        if idx - last >= min_sep:
            flags[idx] = True
            last = idx
    return flags


def _analyse_windows(mono: np.ndarray) -> dict:
    duration = len(mono) / ANALYSIS_SR
    n_windows = 1 + int(math.floor((duration - SEGMENT_SECONDS) / SEGMENT_STEP_SECONDS + 1e-9))
    if n_windows < 1:
        raise AudioToolError(f"可分析时长 {duration:.1f} 秒，不足以切出 {SEGMENT_SECONDS} 秒片段")

    rms_db, rms_hop_s, block_s = _block_rms_db(mono)
    flux = _spectral_flux(mono)
    hop_s = STFT_HOP / ANALYSIS_SR
    frame_s = STFT_N / ANALYSIS_SR
    onsets = _onset_flags(flux, hop_s)
    ref_db = float(np.percentile(rms_db, 95))
    silence_threshold = max(SILENCE_ABS_DB, ref_db - SILENCE_REL_DB)

    windows: list[dict] = []
    for i in range(n_windows):
        start = i * SEGMENT_STEP_SECONDS
        end = start + SEGMENT_SECONDS
        b0, b1 = _frames_in_window(start, end, rms_hop_s, block_s, rms_db.size)
        seg_rms = rms_db[b0:b1 + 1]
        f0, f1 = _frames_in_window(start, end, hop_s, frame_s, flux.size)
        seg_flux = flux[f0:f1 + 1]
        seg_onsets = onsets[f0:f1 + 1]
        metrics = {
            "loudness_std_db": round(float(np.std(seg_rms)) if seg_rms.size > 1 else 0.0, 4),
            "spectral_flux": round(float(np.mean(seg_flux)) if seg_flux.size else 0.0, 6),
            "onset_density": round(float(np.count_nonzero(seg_onsets)) / SEGMENT_SECONDS, 4),
            "silence_ratio": round(float(np.mean(seg_rms < silence_threshold)) if seg_rms.size else 1.0, 4),
        }
        windows.append({"start": round(start, 3), "end": round(end, 3), "metrics": metrics})

    keys = list(SEG_WEIGHTS)
    zscores: dict[str, np.ndarray] = {}
    for key in keys:
        values = np.array([w["metrics"][key] for w in windows], dtype=np.float64)
        std = float(values.std())
        zscores[key] = (values - values.mean()) / std if std > 1e-12 else np.zeros_like(values)
    score = np.zeros(n_windows, dtype=np.float64)
    for key, weight in SEG_WEIGHTS.items():
        sign = -1.0 if key == "silence_ratio" else 1.0
        score += sign * weight * zscores[key]
    for w, s in zip(windows, score):
        w["score"] = round(float(s), 6)

    best0 = int(np.argmax(score))
    selected = [windows[best0]]
    if duration >= MIN_SECONDS_FOR_SECOND_SEGMENT and n_windows > 1:
        candidates = [
            (i, windows[i]) for i in range(n_windows)
            if abs(windows[i]["start"] - windows[best0]["start"]) >= SEGMENT_SECONDS - 1e-9
        ]
        if candidates:
            best1 = max(candidates, key=lambda pair: (score[pair[0]], -pair[0]))[1]
            selected.append(best1)

    return {
        "windows": windows,
        "selected": selected,
        "analysis": {
            "duration": round(duration, 3),
            "analyzed_seconds": round(min(duration, ANALYSIS_MAX_SECONDS), 3),
            "sample_rate": ANALYSIS_SR,
            "silence_threshold_db": round(silence_threshold, 3),
            "weights": dict(SEG_WEIGHTS),
            "window_count": n_windows,
        },
    }


def _segments_meta_path(audio_id: str) -> Path:
    return CACHE_DIR / audio_id / "meta.json"


# 每个 audio_id 一把锁：并发准备测试音频时，多个任务会同时走到 _ensure_segments，
# 没有锁的话两边都可能判定"没有片段缓存"，然后各自 rmtree + 重算，互相删掉对方的产物。
_SEGMENT_LOCKS: dict[tuple[object, str], asyncio.Lock] = {}


def _segment_lock(audio_id: str) -> asyncio.Lock:
    return _loop_scoped_lock(_SEGMENT_LOCKS, audio_id)


async def _ensure_segments(audio_id: str) -> dict:
    """命中片段缓存就直接返回，否则分析 + 切片（同一个 audio_id 同时只允许一个任务在算）。"""
    async with _segment_lock(audio_id):
        return await _ensure_segments_locked(audio_id)


async def _ensure_segments_locked(audio_id: str) -> dict:
    item = _library_item(audio_id)
    flac = _audio_path(audio_id)
    if not flac.exists():
        raise HTTPException(status_code=500, detail="库内音频文件缺失，请重新导入")
    meta_path = _segments_meta_path(audio_id)
    meta = _read_json(meta_path, None)
    if (
        isinstance(meta, dict)
        and meta.get("segment_algo_version") == SEGMENT_ALGO_VERSION
        and meta.get("segments")
        and all((meta_path.parent / seg["file"]).exists() for seg in meta["segments"])
    ):
        return meta

    cache_dir = CACHE_DIR / audio_id
    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw = await _ffmpeg(
        [
            "-i", str(flac), "-t", str(ANALYSIS_MAX_SECONDS), "-map", "0:a:0",
            "-ac", "1", "-ar", str(ANALYSIS_SR), "-f", "f32le", "-",
        ],
        capture_stdout=True,
    )
    usable = len(raw) // 4 * 4
    mono = np.frombuffer(raw[:usable], dtype="<f4").copy()
    if mono.size == 0:
        raise HTTPException(status_code=500, detail="音频分析解码为空")
    analysis = await asyncio.to_thread(_analyse_windows, mono)

    segments = []
    for index, window in enumerate(analysis["selected"]):
        out = cache_dir / f"seg_{index}.flac"
        await _cut_segment(flac, out, window["start"])
        segments.append(
            {
                "index": index,
                "start": window["start"],
                "end": window["end"],
                "file": out.name,
                "score": window["score"],
            }
        )
    meta = {
        "audio_id": audio_id,
        "segment_algo_version": SEGMENT_ALGO_VERSION,
        "encoder": _ENCODER_TAG,
        "display_name": item["display_name"],
        "analysis": analysis["analysis"],
        "windows": analysis["windows"],
        "segments": segments,
    }
    _write_json(meta_path, meta)
    return meta


async def _cut_segment(flac: Path, out: Path, start: float) -> None:
    """按起点切出 15 秒片段（首尾各 20 ms 淡变）。

    注意：-ss 必须放在 -i 之前（输入 seek）。本机 ffmpeg 6.1.1 上，把 -ss 放到 -i 之后
    的“输出 seek”在 start 较大时（实测 40 s）会输出整段数字静音，因此这里固定用输入 seek；
    ffmpeg 默认 -accurate_seek 会解码到精确采样点，采样级精度由测试守护。

    -map 0:a:0 也是必需的：带内嵌封面的源在 start=0 时会把封面帧（attached_pic 视频流）
    一起写进片段，进而让 MP3 编码步骤以 rc=234 失败（见 _import_one 的说明）。
    """
    fade_out_start = SEGMENT_SECONDS - SEGMENT_FADE_SECONDS
    await _ffmpeg(
        [
            "-y", "-ss", f"{start:.6f}", "-i", str(flac), "-t", str(SEGMENT_SECONDS),
            "-map", "0:a:0",
            "-af",
            f"afade=t=in:st=0:d={SEGMENT_FADE_SECONDS},"
            f"afade=t=out:st={fade_out_start}:d={SEGMENT_FADE_SECONDS}",
            "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
            "-c:a", "flac", "-compression_level", "5", str(out),
        ]
    )


# ---------------------------------------------------------------------------
# 测试音频生成（持久缓存）
# ---------------------------------------------------------------------------


def _stimuli_dir(audio_id: str) -> Path:
    return CACHE_DIR / audio_id / "stimuli" / f"v{STIMULI_ALGO_VERSION}"


def _stimuli_key(seg_index: int, bitrate: int | None, alt_seg: int | None = None) -> str:
    if bitrate is None:
        return f"seg{seg_index}_p{alt_seg if alt_seg is not None else seg_index}"
    return f"seg{seg_index}_{bitrate}k"


async def _ensure_stimuli(
    audio_id: str, seg_index: int, bitrate: int | None, alt_seg: int | None = None
) -> tuple[dict, bool]:
    """返回 (测试音频条目, 是否命中缓存)。命中缓存只是读 index.json，不重算。"""
    meta = await _ensure_segments(audio_id)
    seg_files = {seg["index"]: seg["file"] for seg in meta["segments"]}
    if seg_index not in seg_files:
        raise HTTPException(status_code=400, detail=f"片段 {seg_index} 不存在")
    cache_root = _stimuli_dir(audio_id)
    index_path = cache_root / "index.json"
    index = _read_json(index_path, None)
    if not isinstance(index, dict) or index.get("stimuli_algo_version") != STIMULI_ALGO_VERSION:
        index = {"stimuli_algo_version": STIMULI_ALGO_VERSION, "encoder": _ENCODER_TAG, "entries": {}}
    key = _stimuli_key(seg_index, bitrate, alt_seg)
    entry = (index.get("entries") or {}).get(key)
    if entry and all((cache_root / rel).exists() for rel in entry["files"].values()):
        return entry, True

    seg_path = CACHE_DIR / audio_id / seg_files[seg_index]
    master, _ = _fit_length(_read_audio(seg_path))
    if bitrate is None:
        if PLACEBO_SAME_SEGMENT or alt_seg is None:
            cand = master
        else:
            alt_path = CACHE_DIR / audio_id / seg_files[alt_seg]
            cand = _read_audio(alt_path)
    else:
        tmp_mp3 = TMP_DIR / f"enc_{secrets.token_hex(8)}.mp3"
        try:
            await _ffmpeg(
                [
                    "-y", "-i", str(seg_path), "-map", "0:a:0", *LAME_EXTRA_ARGS,
                    "-c:a", "libmp3lame", "-b:a", f"{bitrate}k", "-f", "mp3", str(tmp_mp3),
                ]
            )
            cand = await _ffmpeg_to_array(tmp_mp3, SAMPLE_RATE, CHANNELS)
        finally:
            tmp_mp3.unlink(missing_ok=True)

    out_ll, out_mp3, info = await asyncio.to_thread(_prepare_pair, master, cand)

    cache_root.mkdir(parents=True, exist_ok=True)
    # 无损侧的内容只由 (片段, common_delta) 决定 —— `master_fit` 与 `g_ll` 都只依赖片段，
    # 所以 delta 相同的两个码率算出来逐位相同。delta == 0（默认情况：-18 LUFS 下真峰值
    # 还有 4 dB 余量）时让整个片段共用一份 `seg{S}_ll.wav`，无损侧从 N 份降到 1 份。
    # delta != 0（把 TARGET_LUFS 调得很高才会发生）时退回每档一份，保证自检结论不被共享破坏：
    # 共用只在"这份文件就是 delta=0 的那份"时成立，而 delta=0 的判据与内容一一对应。
    # 注：delta 存的是四舍五入到 1e-4 dB 的值，被归入 0 的偏差 <= 5e-4 dB，
    # 远小于自检容差 TRUE_PEAK_SLACK_DB = 0.05，既不会让峰值超标也不影响 ΔLUFS。
    shared_ll = info["gain_db"]["common_delta"] == 0.0
    ll_name = f"seg{seg_index}_ll.wav" if shared_ll else f"{key}_ll.wav"

    if bitrate is None and (PLACEBO_SAME_SEGMENT or alt_seg is None):
        # 安慰剂：A/B 是同一份文件。delta == 0 时它和无损侧本来就是同一份内容，直接共用。
        if shared_ll:
            path_ll = path_mp3 = cache_root / ll_name
            await asyncio.to_thread(_write_wav_atomic, path_ll, out_ll)
            files = {"ll": ll_name, "mp3": ll_name}
        else:
            path_ll = path_mp3 = cache_root / f"{key}.wav"
            await asyncio.to_thread(_write_wav_atomic, path_ll, out_ll)
            files = {"ll": path_ll.name, "mp3": path_ll.name}
    else:
        path_ll = cache_root / ll_name
        path_mp3 = cache_root / f"{key}_mp3.wav"
        await asyncio.to_thread(_write_wav_atomic, path_ll, out_ll)
        await asyncio.to_thread(_write_wav_atomic, path_mp3, out_mp3)
        files = {"ll": path_ll.name, "mp3": path_mp3.name}

    # 整数位深下"内存数组"和"交付字节"不是一回事：这里以落盘文件为准补一次结构自检，
    # 证明交付内容只是那份数组的量化（详见 `_verify_written_pair` 的说明）。
    await asyncio.to_thread(_verify_written_pair, path_ll, path_mp3, out_ll, out_mp3)

    entry = {
        "key": key,
        "bitrate": bitrate,
        "seg": seg_index,
        "alt_seg": alt_seg,
        "files": files,
        "encoder": _ENCODER_TAG,
        **info,
    }
    # 并发生成时多个任务会同时读改写 index.json：必须在锁内**重新读一次**再合并，
    # 否则后写的会覆盖掉别人刚登记的条目（文件在、索引丢 → 下次白重算一遍）。
    async with _loop_scoped_lock(_CACHE_INDEX_LOCKS, audio_id):
        current = _read_json(index_path, None)
        if (
            not isinstance(current, dict)
            or current.get("stimuli_algo_version") != STIMULI_ALGO_VERSION
        ):
            current = {
                "stimuli_algo_version": STIMULI_ALGO_VERSION,
                "encoder": _ENCODER_TAG,
                "entries": {},
            }
        current.setdefault("entries", {})[key] = entry
        _write_json(index_path, current)
    return entry, False


# ---------------------------------------------------------------------------
# 出题
# ---------------------------------------------------------------------------


@dataclass
class Question:
    index: int
    question_id: str
    audio_id: str
    seg_index: int
    bitrate: int | None
    lossless_is_a: bool
    stimulus: dict
    alt_seg: int | None = None


# 一次测试音频请求 = 缓存键 = (音频, 片段, 码率)；安慰剂题的 bitrate 为 None
StimulusKey = tuple[str, int, int | None, int | None]


def _stimulus_key_of(question: Question) -> StimulusKey:
    return (question.audio_id, question.seg_index, question.bitrate, question.alt_seg)


def _unique_requests(keys: list[StimulusKey]) -> list[StimulusKey]:
    """去重且保持顺序（同一条测试音频在一轮里只该生成一次）。"""
    return list(dict.fromkeys(keys))


@dataclass
class SessionState:
    session_id: str
    created_at: float
    mode: str
    audio_ids: list[str]
    questions: list[Question]
    answers: dict[str, dict] = field(default_factory=dict)
    manifest: dict[str, str] = field(default_factory=dict)  # token -> 绝对路径
    tokens: dict[str, dict[str, str]] = field(default_factory=dict)  # question_id -> {role: token}
    dir: Path = field(default_factory=Path)

    def to_public(self) -> dict:
        return {"session_id": self.session_id, "total": len(self.questions), "mode": self.mode}


SESSIONS: dict[str, SessionState] = {}


def _real_and_placebo_counts(mode: str) -> tuple[int, int]:
    if mode == "short":
        return SHORT_TEST_TOTAL - 1, 1
    return DETAILED_TEST_TOTAL - 2, 2


def _build_questions(mode: str, audio_ids: list[str], segment_indices: dict[str, list[int]]) -> list[Question]:
    ladder = SHORT_TEST_BITRATES if mode == "short" else DETAILED_TEST_BITRATES
    n_real, n_placebo = _real_and_placebo_counts(mode)
    bitrates = [ladder[i % len(ladder)] for i in range(n_real)]
    rng = random.Random(secrets.randbits(64))

    usage = {aid: 0 for aid in audio_ids}
    used_pairs: set[tuple[str, int, int]] = set()
    questions: list[Question] = []

    def pick_seg(audio_id: str, bitrate: int) -> int:
        segs = segment_indices[audio_id]
        k = usage[audio_id]
        usage[audio_id] += 1
        ordered = [segs[k % len(segs)]] + [s for s in segs if s != segs[k % len(segs)]]
        for seg in ordered:
            if (audio_id, seg, bitrate) not in used_pairs:
                used_pairs.add((audio_id, seg, bitrate))
                return seg
        return ordered[0]

    for i, bitrate in enumerate(bitrates):
        audio_id = audio_ids[i % len(audio_ids)]
        questions.append(
            Question(
                index=len(questions),
                question_id=f"q{len(questions)}",
                audio_id=audio_id,
                seg_index=pick_seg(audio_id, bitrate),
                bitrate=bitrate,
                lossless_is_a=rng.random() < 0.5,
                stimulus={},
            )
        )

    for j in range(n_placebo):
        audio_id = audio_ids[j % len(audio_ids)]
        segs = segment_indices[audio_id]
        seg_index = segs[usage[audio_id] % len(segs)]
        usage[audio_id] += 1
        alt_seg = None
        if not PLACEBO_SAME_SEGMENT and len(segs) > 1:
            alt_seg = next(s for s in segs if s != seg_index)
        questions.append(
            Question(
                index=len(questions),
                question_id=f"q{len(questions)}",
                audio_id=audio_id,
                seg_index=seg_index,
                bitrate=None,
                lossless_is_a=rng.random() < 0.5,
                stimulus={},
                alt_seg=alt_seg,
            )
        )

    if SHUFFLE_ALL_QUESTIONS:
        # 真实题与安慰剂题一起打乱：这道题是第几题不再携带任何信息
        # （既不能靠"最后一题必是安慰剂"直接按听不出，也不能靠阶梯顺序推码率）。
        rng.shuffle(questions)
    for position, question in enumerate(questions):
        question.index = position
        question.question_id = f"q{position}"
    return questions


# ---------------------------------------------------------------------------
# 准备进度（内存态，仅用于界面进度条）
# ---------------------------------------------------------------------------

PROGRESS: dict[str, dict] = {}
PROGRESS_TTL_SECONDS = 300


def _progress_begin(token: str | None, *, phase: str, total: int) -> None:
    if not token:
        return
    PROGRESS[token] = {
        "phase": phase,
        "done": 0,
        "total": max(0, total),
        "finished": False,
        "error": None,
        "updated_at": time.time(),
    }


def _progress_update(
    token: str | None, *, phase: str | None = None, done: int | None = None, total: int | None = None
) -> None:
    entry = PROGRESS.get(token) if token else None
    if entry is None:
        return
    if phase is not None:
        entry["phase"] = phase
    if done is not None:
        entry["done"] = done
    if total is not None:
        entry["total"] = total
    entry["updated_at"] = time.time()


def _progress_finish(token: str | None, error: str | None = None) -> None:
    entry = PROGRESS.get(token) if token else None
    if entry is None:
        return
    entry["finished"] = True
    entry["error"] = error
    if error is None:
        entry["done"] = entry["total"]
    entry["updated_at"] = time.time()


def _progress_purge() -> None:
    now = time.time()
    for token, entry in list(PROGRESS.items()):
        if now - entry["updated_at"] > PROGRESS_TTL_SECONDS:
            PROGRESS.pop(token, None)


async def _prepare_session(
    questions: list[Question],
    progress_token: str | None = None,
    prewarm: list["StimulusKey"] | None = None,
) -> dict:
    """为每道题准备测试音频（命中缓存则复用），返回本次新生成/复用/预热的数量。

    并发执行：单题约 0.70 s（响度+对齐 0.46 s、ffmpeg 编解码 0.25 s），逐题串行时
    22 题要 15 s 以上；按核数并发（默认封顶 8）实测 3.9 s。
    `STIMULI_CONCURRENCY` 同时决定任务槽位与 ffmpeg 子进程槽位。

    `prewarm` 是额外的请求（本轮题目用不到，但下次换模式会用到，见 `PREWARM_SHORT_COMPLEMENT`）：
    它们只生成缓存、不挂到题目上，计数单独报在 `prewarmed` 里，免得用户看到"22 题却新生成 23 段"。
    """
    requests = _unique_requests([_stimulus_key_of(q) for q in questions])
    question_keys = set(requests)
    extra = [key for key in _unique_requests(prewarm or []) if key not in question_keys]
    total = len(requests) + len(extra)
    semaphore = asyncio.Semaphore(STIMULI_CONCURRENCY)
    entries: dict[StimulusKey, dict] = {}
    counters = {"generated": 0, "reused": 0, "prewarmed": 0, "done": 0}

    async def prepare_one(key: StimulusKey) -> None:
        audio_id, seg_index, bitrate, alt_seg = key
        async with semaphore:
            entry, from_cache = await _ensure_stimuli(audio_id, seg_index, bitrate, alt_seg)
        entries[key] = entry
        if key in question_keys:
            counters["reused" if from_cache else "generated"] += 1
        elif not from_cache:
            counters["prewarmed"] += 1
        counters["done"] += 1
        # 单事件循环里读改写之间没有 await，计数不会互相覆盖
        _progress_update(progress_token, done=counters["done"], total=total, phase="stimuli")

    tasks = [asyncio.create_task(prepare_one(key)) for key in requests + extra]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        # 任何一题失败都要收掉其余任务，否则接口已经返回 500 了后台还在跑 ffmpeg
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for question in questions:
        question.stimulus = entries[_stimulus_key_of(question)]
    return {
        "generated": counters["generated"],
        "reused": counters["reused"],
        "prewarmed": counters["prewarmed"],
    }


def _reveal(question: Question) -> dict:
    """本题两侧的真实身份（无损 / MP3 码率）。只在作答之后的响应里出现。"""
    lossless = {"kind": "lossless", "bitrate": None}
    coded = {"kind": "mp3", "bitrate": question.bitrate}
    files = question.stimulus.get("files", {}) if question.stimulus else {}
    if question.bitrate is None:
        # 安慰剂题：两侧都是无损（默认还是同一片段的两份拷贝）
        sides = {"a": dict(lossless), "b": dict(lossless)}
    else:
        sides = {
            "a": dict(lossless if question.lossless_is_a else coded),
            "b": dict(coded if question.lossless_is_a else lossless),
        }
    return {
        "placebo": question.bitrate is None,
        "correct_side": None if question.bitrate is None else ("a" if question.lossless_is_a else "b"),
        "same_segment": bool(files.get("ll")) and files.get("ll") == files.get("mp3"),
        "sides": sides,
    }


def _destroy_session(session_id: str) -> None:
    state = SESSIONS.pop(session_id, None)
    if state is None:
        return
    shutil.rmtree(state.dir, ignore_errors=True)


def _cleanup_sessions() -> None:
    now = time.time()
    for session_id, state in list(SESSIONS.items()):
        if now - state.created_at > SESSION_TTL_SECONDS:
            _destroy_session(session_id)
    if SESSION_DIR.exists():
        for path in SESSION_DIR.iterdir():
            if path.is_dir() and path.name not in SESSIONS:
                shutil.rmtree(path, ignore_errors=True)


def _require_session(session_id: str) -> SessionState:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新开始测试")
    return state


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _binom_p_value(successes: int, total: int, p0: float = CHANCE_LEVEL) -> float | None:
    """单侧精确二项检验 P(X >= successes | p = p0)，p0 默认 = 三选一瞎猜基线 1/3。"""
    if total <= 0:
        return None
    return float(sps_stats.binomtest(successes, total, p0, alternative="greater").pvalue)


def _required_n(
    p0: float = CHANCE_LEVEL,
    p1: float = P1_FOR_POWER,
    alpha: float = ALPHA,
    power: float = POWER_TARGET,
) -> int:
    """达到指定功效所需的题数（单侧精确二项检验）。找不到则返回 -1。"""
    for n in range(1, 500):
        critical = next(
            (k for k in range(n + 1) if sps_stats.binom.sf(k - 1, n, p0) <= alpha), None
        )
        if critical is None:
            continue
        if sps_stats.binom.sf(critical - 1, n, p1) >= power:
            return n
    return -1


# 当前口径下"每档要看多少题"才能在 80% 功效下做出判断（默认：7 题）
POWER_REQUIRED_N = _required_n()


def _row_stats(correct: int, total: int) -> dict:
    """一行统计：计数 + 描述性 Wilson 区间 + 单侧精确二项检验（H0 = 1/3）。

    `status` 是给前端做视觉分级的稳定枚举：
    `insufficient`（样本不足）/ `significant`（显著高于瞎猜）/ `not_significant`（无显著差异）。
    """
    if total <= 0:
        return {
            "n": 0,
            "correct": 0,
            "accuracy": None,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": None,
            "significant": False,
            "status": "insufficient",
            "note": "没有作答",
        }
    low, high = _wilson_interval(correct, total)
    p_value = _binom_p_value(correct, total)
    significant = total >= MIN_N_FOR_SIGNIFICANCE and p_value is not None and p_value < ALPHA
    if total < MIN_N_FOR_SIGNIFICANCE:
        status = "insufficient"
        note = f"样本不足（n<{MIN_N_FOR_SIGNIFICANCE}，无法判定）"
    elif significant:
        status = "significant"
        note = "显著高于瞎猜基线"
    else:
        status = "not_significant"
        note = "与瞎猜基线无显著差异"
    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        # 不四舍五入：20/20 的精确 p 是 2.9e-10，舍入到 4 位会变成误导性的 0.0
        "p_value": p_value,
        "significant": significant,
        "status": status,
        "note": note,
    }


def _is_correct(question: Question, choice: str) -> bool | None:
    if choice == "tie":
        return question.bitrate is None
    if question.bitrate is None:
        return False
    chose_lossless = (choice == "a") == question.lossless_is_a
    return chose_lossless


def _build_result(state: SessionState) -> dict:
    answered = [q for q in state.questions if q.question_id in state.answers]
    real = [q for q in answered if q.bitrate is not None]
    placebo = [q for q in answered if q.bitrate is None]

    def correct_of(items: list[Question]) -> int:
        return sum(1 for q in items if _is_correct(q, state.answers[q.question_id]["choice"]))

    by_bitrate = []
    for bitrate in (SHORT_TEST_BITRATES if state.mode == "short" else DETAILED_TEST_BITRATES):
        items = [q for q in real if q.bitrate == bitrate]
        if not items:
            continue
        row = {"bitrate": bitrate, **_row_stats(correct_of(items), len(items))}
        times = [state.answers[q.question_id]["reaction_ms"] for q in items
                 if state.answers[q.question_id].get("reaction_ms") is not None]
        row["mean_reaction_ms"] = round(sum(times) / len(times)) if times else None
        by_bitrate.append(row)

    by_bracket = []
    for low, high, label in RESULT_BRACKETS:
        items = [q for q in real if low <= q.bitrate <= high]
        if not items:
            continue
        by_bracket.append({"label": label, "range": [low, high], **_row_stats(correct_of(items), len(items))})

    placebo_ties = sum(1 for q in placebo if state.answers[q.question_id]["choice"] == "tie")
    placebo_rate = round(placebo_ties / len(placebo), 4) if placebo else None
    placebo_row = {
        "n": len(placebo),
        "tie": placebo_ties,
        "tie_rate": placebo_rate,
        # 在"其实没有差异"的题上仍听出更好的一边 → 提示瞎猜倾向
        "warning": bool(placebo) and placebo_rate is not None and placebo_rate < 0.5,
    }

    overall = _row_stats(correct_of(real), len(real))
    max_per_bitrate = max((row["n"] for row in by_bitrate), default=0)
    return {
        "mode": state.mode,
        "total": len(state.questions),
        "answered": len(answered),
        "real_total": len([q for q in state.questions if q.bitrate is not None]),
        "real_answered": len(real),
        "real_correct": overall["correct"],
        "placebo_total": len([q for q in state.questions if q.bitrate is None]),
        "overall": overall,
        "by_bitrate": by_bitrate,
        "by_bracket": by_bracket,
        "placebo": placebo_row,
        "chance_level": round(CHANCE_LEVEL, 4),
        "alpha": ALPHA,
        "power_required_n": POWER_REQUIRED_N,
        # 给前端渲染"样本量说明"用的结构化信息（避免前端去解析中文字符串）
        "power": {
            "required_n": POWER_REQUIRED_N,
            "max_per_bitrate": max_per_bitrate,
            "target_power": POWER_TARGET,
            "assumed_rate": P1_FOR_POWER,
        },
        "conclusion": _conclusion(overall, by_bracket, by_bitrate, placebo_row),
    }


def _p_text(p_value: float | None) -> str:
    if p_value is None:
        return "p 值缺失"
    return "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"


def _verdict(row: dict) -> str:
    if row["n"] < MIN_N_FOR_SIGNIFICANCE:
        return f"样本不足（n={row['n']} < {MIN_N_FOR_SIGNIFICANCE}），无法判定"
    if row["significant"]:
        return f"显著高于瞎猜基线 1/3（{_p_text(row['p_value'])}）"
    return f"与瞎猜基线无显著差异（{_p_text(row['p_value'])}）"


def _conclusion(overall: dict, by_bracket: list[dict], by_bitrate: list[dict], placebo: dict) -> str:
    if not overall["n"] and not placebo["n"]:
        return "还没有作答，无法给出结论。"
    sentences: list[str] = []
    if overall["n"]:
        sentences.append(
            f"真实题总体 {overall['correct']}/{overall['n']}"
            f"（{round(overall['accuracy'] * 100)}%）：{_verdict(overall)}"
        )
    for bracket in by_bracket:
        sentences.append(
            f"{bracket['label']} {bracket['correct']}/{bracket['n']}：{_verdict(bracket)}"
        )
    if not sentences:
        sentences.append("没有真实题作答，无法判断")
    if by_bitrate:
        max_n = max(row["n"] for row in by_bitrate)
        sentences.append(
            f"要让每档都达到 {round(POWER_TARGET * 100)}% 检验功效"
            f"（假定真听出率 {P1_FOR_POWER}），约需每档 {POWER_REQUIRED_N} 题，本次每档最多 {max_n} 题"
        )
    if placebo["n"] and placebo["tie_rate"] is not None and placebo["tie_rate"] < 0.5:
        sentences.append("安慰剂题上你多次声称听出差异，提示存在瞎猜倾向")
    return "；".join(sentences) + "。"


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


def _bootstrap() -> None:
    for path in (LIBRARY_DIR, CACHE_DIR, SESSION_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(TMP_DIR)
    # 启动清理：session 与 tmp 全部清空；library 与 cache 保留
    for path in (SESSION_DIR, TMP_DIR):
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    SESSIONS.clear()
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"缺少必需的外部程序：{', '.join(missing)}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    _bootstrap()
    yield


app = FastAPI(title="TinEar", lifespan=lifespan)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def _revalidate_static_assets(request, call_next):
    """HTML / JS / CSS 一律要求浏览器重验证。

    否则浏览器可能继续用缓存里的旧 app.js 去调新 API —— 症状是"点查看答案按钮消失、
    什么都不显示"（旧脚本读 data.correct_side，而新接口已改成 reveal，取属性时抛异常）。
    本项目是本地工具，静态资源很小，重验证（命中 ETag 时是 304）没有任何负担。
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static/") or path == "/index.html":
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


class StartRequest(BaseModel):
    audio_ids: list[str]
    mode: str
    progress_token: str | None = None


class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    choice: str
    reaction_ms: int | None = None


class EndRequest(BaseModel):
    session_id: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/library")
async def api_library() -> dict:
    # cache_bytes = 界面「清理测试音频缓存」按钮会释放的量（data/cache：片段 + 测试音频）
    # tmp_bytes   = 草稿区占用（data/tmp：上传落盘 / ffmpeg 中间文件），后端每次启动自动清空，
    #               不由那个按钮负责 —— 分开报，界面才不会让人误以为按钮没生效。
    return {
        "items": _library()["items"],
        "cache_bytes": _dir_size(CACHE_DIR),
        "tmp_bytes": _dir_size(TMP_DIR),
    }


@app.get("/api/progress")
async def api_progress(token: str = Query(...)) -> dict:
    """准备进度（前端进度条用）：token 由前端生成并随 /api/start 一起提交。

    只读内存字典，命不中就返回 found=false —— 前端因此不需要额外处理 404。
    """
    _progress_purge()
    entry = PROGRESS.get(token)
    if entry is None:
        return {"found": False}
    return {"found": True, **entry}


@app.post("/api/library/upload")
async def api_upload(files: list[UploadFile] = File(...)) -> dict:
    results = []
    for upload in files:
        results.append(await _import_one(upload))
    return {"results": results}


@app.delete("/api/library/{audio_id}")
async def api_delete_library(audio_id: str) -> dict:
    item = _library_item(audio_id)
    # 只有**尚未完成**的测试才算占用：答完全部题目后即不再阻止删除
    in_use = [
        session_id for session_id, state in SESSIONS.items()
        if len(state.answers) < len(state.questions)
        and any(q.audio_id == audio_id for q in state.questions)
    ]
    if in_use:
        raise HTTPException(status_code=409, detail="该音频正被进行中的测试引用，请先结束测试")
    async with _LIBRARY_LOCK:
        lib = _library()
        lib["items"] = [entry for entry in lib["items"] if entry["audio_id"] != audio_id]
        _write_json(LIBRARY_INDEX, lib)
        _audio_path(audio_id).unlink(missing_ok=True)
        freed = _dir_size(CACHE_DIR / audio_id)
        shutil.rmtree(CACHE_DIR / audio_id, ignore_errors=True)
    return {"deleted": True, "freed_bytes": freed, "display_name": item["display_name"]}


@app.post("/api/start")
async def api_start(request: StartRequest) -> dict:
    _cleanup_sessions()
    _progress_purge()
    if request.mode not in ("short", "detailed"):
        raise HTTPException(status_code=400, detail="mode 必须是 short 或 detailed")
    # 固定顺序：同一批素材必须给出同一张题表，否则 (片段, 码率) 组合会变、缓存命中率退化。
    # 前端是按**勾选顺序**发送的（Set 的迭代顺序），所以这里统一按库内顺序排序。
    library_order = {item["audio_id"]: i for i, item in enumerate(_library()["items"])}
    audio_ids = sorted(
        dict.fromkeys(request.audio_ids),
        key=lambda audio_id: (library_order.get(audio_id, 1 << 30), audio_id),
    )
    if not 1 <= len(audio_ids) <= 3:
        raise HTTPException(status_code=400, detail="请选择 1～3 个音频")
    for audio_id in audio_ids:
        _library_item(audio_id)

    progress_token = request.progress_token
    _progress_begin(progress_token, phase="segments", total=len(audio_ids))
    segment_indices: dict[str, list[int]] = {}
    for position, audio_id in enumerate(audio_ids):
        meta = await _ensure_segments(audio_id)
        segment_indices[audio_id] = [seg["index"] for seg in meta["segments"]]
        _progress_update(progress_token, done=position + 1)

    questions = _build_questions(request.mode, audio_ids, segment_indices)
    prewarm: list[StimulusKey] = []
    if request.mode == "detailed" and PREWARM_SHORT_COMPLEMENT:
        # 详细模式顺手把"快速模式才用到的那几对"也生成掉，下次切快速就是 0.01 s。
        # 这里用 _build_questions 算出快速模式实际会请求哪些组合（而不是"224 kbps × 所有片段"），
        # 阶梯以后怎么改都自动对齐；两套阶梯的差集通常只有 1～2 对。
        short_requests = _unique_requests(
            [_stimulus_key_of(q) for q in _build_questions("short", audio_ids, segment_indices)]
        )
        detailed_keys = set(_stimulus_key_of(q) for q in questions)
        prewarm = [key for key in short_requests if key not in detailed_keys]
    _progress_update(
        progress_token, phase="stimuli", done=0, total=len(questions) + len(prewarm)
    )
    try:
        stimuli = await _prepare_session(questions, progress_token, prewarm)
    except AudioToolError as exc:
        detail = f"生成测试音频失败：{exc}"
        _progress_finish(progress_token, error=detail)
        raise HTTPException(status_code=500, detail=detail) from exc

    session_id = secrets.token_hex(8)
    session_dir = SESSION_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    state = SessionState(
        session_id=session_id,
        created_at=time.time(),
        mode=request.mode,
        audio_ids=audio_ids,
        questions=questions,
        dir=session_dir,
    )
    for question in questions:
        files = question.stimulus["files"]
        role_tokens: dict[str, str] = {}
        for role in ("ll", "mp3"):
            source = _stimuli_dir(question.audio_id) / files[role]
            audio_token = secrets.token_hex(8)
            target = session_dir / f"{audio_token}.wav"
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            state.manifest[audio_token] = str(target)
            role_tokens[role] = audio_token
        state.tokens[question.question_id] = role_tokens
    SESSIONS[session_id] = state
    _progress_finish(progress_token)
    return {**state.to_public(), "stimuli": stimuli, "target_lufs": TARGET_LUFS}


@app.get("/api/question")
async def api_question(session_id: str = Query(...), index: int = Query(...)) -> dict:
    state = _require_session(session_id)
    if index < 0 or index >= len(state.questions):
        raise HTTPException(status_code=400, detail="题号越界")
    question = state.questions[index]
    role_tokens = state.tokens.get(question.question_id)
    if not role_tokens:
        raise HTTPException(status_code=500, detail="题目音频缺失")
    url_for: dict[str, str] = {}
    for question_role, role in (("a", "ll" if question.lossless_is_a else "mp3"),
                                ("b", "mp3" if question.lossless_is_a else "ll")):
        token = role_tokens.get(role)
        if token is None:
            raise HTTPException(status_code=500, detail="题目音频缺失")
        url_for[question_role] = f"/api/audio/{session_id}/{token}"
    return {
        "index": index,
        "total": len(state.questions),
        "question_id": question.question_id,
        "url_a": url_for["a"],
        "url_b": url_for["b"],
    }


@app.get("/api/audio/{session_id}/{token}")
async def api_audio(session_id: str, token: str) -> FileResponse:
    state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    raw = state.manifest.get(token)
    if raw is None:
        raise HTTPException(status_code=404, detail="音频不存在")
    path = Path(raw).resolve()
    try:
        path.relative_to(state.dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="音频不存在") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="音频不存在")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/answer")
async def api_answer(request: AnswerRequest) -> dict:
    state = _require_session(request.session_id)
    if request.choice not in ("a", "b", "tie"):
        raise HTTPException(status_code=400, detail="choice 必须是 a / b / tie")
    question = next((q for q in state.questions if q.question_id == request.question_id), None)
    if question is None:
        raise HTTPException(status_code=400, detail="题目不属于该会话")
    reaction = request.reaction_ms
    if reaction is not None and reaction < 0:
        raise HTTPException(status_code=400, detail="reaction_ms 不能为负")

    stored = state.answers.get(question.question_id)
    if stored is None:
        state.answers[question.question_id] = {"choice": request.choice, "reaction_ms": reaction}
        accepted = True
    else:
        # 首答即最终：同样的选项重复提交按幂等回放处理；改了选项则忽略（否则可以先看答案再改分）。
        accepted = stored["choice"] == request.choice
    done = len(state.answers) >= len(state.questions)
    # reveal（本题两侧各是无损还是 MP3 几 kbps）只在**首答/幂等重放**时返回，
    # 未作答的题目（/api/question）与音频接口始终不泄漏任何码率信息。
    return {
        "ok": True,
        "index": question.index,
        "next_index": question.index + 1,
        "done": done,
        "accepted": accepted,
        "reveal": _reveal(question) if accepted else None,
    }


@app.get("/api/result")
async def api_result(session_id: str = Query(...)) -> dict:
    state = _require_session(session_id)
    return _build_result(state)


@app.post("/api/end")
async def api_end(request: EndRequest) -> dict:
    _require_session(request.session_id)
    _destroy_session(request.session_id)
    return {"ok": True}


@app.delete("/api/cache")
async def api_clear_cache() -> dict:
    freed = _dir_size(CACHE_DIR)
    if CACHE_DIR.exists():
        for child in CACHE_DIR.iterdir():
            shutil.rmtree(child, ignore_errors=True)
    return {"freed_bytes": freed}


# ---------------------------------------------------------------------------
# 命令行入口（`python app.py`）
# ---------------------------------------------------------------------------


def _drop_uvicorn_bind_banner(record) -> bool:
    """过滤掉 uvicorn 的 "Uvicorn running on …" 那一行。

    那行打的是**绑定地址**（默认写成 `0.0.0.0`，而它并不是一个能访问的地址），与我们要展示的
    "打开 http://localhost:<port>" 并排出现只会让人犹豫该开哪个。`logger.addFilter` 接受任意
    可调用对象，所以这里用函数而不是 `logging.Filter` 子类。
    """
    return "Uvicorn running on" not in record.getMessage()


def _startup_lines(host: str, port: int) -> list[str]:
    """`python app.py` 启动时要打印的行：访问地址 + （仅监听所有网卡时的）安全提示。

    `0.0.0.0` / `::` 表示监听所有网卡，用户该打开的是 localhost；绑了具体网卡就照实显示那个地址
    （那时 localhost 反而打不开，所以不能一律写 localhost）。
    """
    display = "localhost" if host in {"0.0.0.0", "::", "127.0.0.1", "localhost"} else host
    lines = [f"TinEar 启动中：http://{display}:{port}    （按 Ctrl+C 停止）"]
    if host in {"0.0.0.0", "::"}:
        lines.append(
            "提示：服务监听所有网卡，局域网内其他设备也能打开，而界面没有认证；"
            "只想本机访问请用 HOST=127.0.0.1 启动。"
        )
    return lines


if __name__ == "__main__":  # pragma: no cover
    # 最省事的启动入口：`python app.py`（等价于 uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1）。
    # 这里直接传 app 对象、不用 "app:app" 字符串，省掉一次模块导入；会话在内存里，所以固定单进程。
    # HOST / PORT 环境变量可覆盖，与 Makefile 的约定一致。
    import logging
    import sys

    import uvicorn

    _host = os.environ.get("HOST", "0.0.0.0")
    _port = int(os.environ.get("PORT", "8000"))
    # 走 stderr 且 flush：uvicorn 的日志也在 stderr，两者顺序才正确；而且 stdout 被重定向时是块缓冲的，
    # 那行提示会一直压到进程退出才出现（被 Ctrl+C 打断时甚至看不到）—— 见 MEASUREMENTS.md 的实测记录。
    for _line in _startup_lines(_host, _port):
        print(_line, file=sys.stderr, flush=True)

    # 自己建 Config/Server 而不是调 uvicorn.run()：uvicorn 在 `Config.__init__` 里做 dictConfig，
    # 在这之后装过滤器才不会被它覆盖（用 uvicorn.run() 就得赌过滤器能活过 dictConfig）。
    _config = uvicorn.Config(app, host=_host, port=_port, workers=1)
    logging.getLogger("uvicorn.error").addFilter(_drop_uvicorn_bind_banner)
    uvicorn.Server(_config).run()
