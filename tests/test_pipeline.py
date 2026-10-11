"""音频管线测试：常量守护、对齐、等响度、线性增益、片段选择、刺激缓存。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import pytest
import soundfile as sf

import app as app_module
from conftest import (
    SR,
    attach_cover_flac,
    cache_wavs,
    decode_f32,
    encode_mp3,
    start_session,
    stream_types,
    synth_signal,
    upload,
    write_source,
)


# ---------------------------------------------------------------------------
# 常量守护（与 AGENTS.md §4 对齐）
# ---------------------------------------------------------------------------


def test_spec_constants_match_agents_md():
    assert app_module.SAMPLE_RATE == 48000
    assert app_module.CHANNELS == 2
    assert app_module.SEGMENT_SECONDS == 15
    assert app_module.TARGET_LUFS == -18.0
    assert app_module.TRUE_PEAK_DBTP == -1.0
    assert app_module.SHORT_TEST_BITRATES == [48, 96, 160, 224, 320]
    assert app_module.DETAILED_TEST_BITRATES == [32, 48, 64, 80, 96, 128, 160, 192, 256, 320]
    assert app_module.SHORT_TEST_TOTAL == 6
    assert app_module.DETAILED_TEST_TOTAL == 22  # 每档 2 题，给统计一点功效
    assert app_module.SESSION_TTL_SECONDS == 7200
    assert app_module.SHUFFLE_ALL_QUESTIONS is True  # 真实题与安慰剂题一起打乱
    # 三选一的瞎猜基线是 1/3，不是 1/2
    assert app_module.CHANCE_LEVEL == pytest.approx(1 / 3)
    assert app_module.MIN_N_FOR_SIGNIFICANCE == 3
    # short 的档位并非 detailed 的子集（224 kbps 只在 short 里），两条阶梯的并集共 11 档
    union = set(app_module.SHORT_TEST_BITRATES) | set(app_module.DETAILED_TEST_BITRATES)
    assert union == {32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320}
    assert 224 in app_module.SHORT_TEST_BITRATES and 224 not in app_module.DETAILED_TEST_BITRATES
    assert app_module._real_and_placebo_counts("short") == (5, 1)
    assert app_module._real_and_placebo_counts("detailed") == (20, 2)
    # 默认题量下每档题数：快速 1 题、详细 2 题
    assert app_module._real_and_placebo_counts("short")[0] / len(app_module.SHORT_TEST_BITRATES) == 1
    assert app_module._real_and_placebo_counts("detailed")[0] / len(app_module.DETAILED_TEST_BITRATES) == 2


def test_segment_geometry_is_mp3_frame_aligned():
    """15 s @48 kHz = 720000 采样 = 恰好 625 个 MPEG-1 Layer III 帧。"""
    assert app_module.SEGMENT_SAMPLES == 720000
    assert app_module.SEGMENT_SAMPLES % 1152 == 0
    assert app_module.SEGMENT_SAMPLES // 1152 == 625


# ---------------------------------------------------------------------------
# 对齐
# ---------------------------------------------------------------------------


def test_align_sign_convention():
    """对齐器必须能还原已知的整数延迟，正负两个方向都要正确。"""
    master = synth_signal(3.0, seed=11)
    assert app_module._estimate_lag(master, master) == 0

    for shift in (137, 1107, -233):
        if shift > 0:
            candidate = np.concatenate(
                [np.zeros((shift, master.shape[1]), dtype=master.dtype), master[:-shift]]
            )
        else:
            candidate = np.concatenate(
                [master[-shift:], np.zeros((-shift, master.shape[1]), dtype=master.dtype)]
            )
        assert app_module._estimate_lag(candidate, master) == shift, f"shift={shift}"


def test_mp3_roundtrip_is_recovered(segment_pair):
    """真实的 MP3 编解码往返：对齐后与母版高度相关，长度精确 720000。"""
    master, candidate, _ = segment_pair
    lag = app_module._estimate_lag(candidate, master)
    aligned, _ = app_module._fit_length(app_module._shifted(candidate, lag))
    assert len(aligned) == app_module.SEGMENT_SAMPLES

    a = aligned.mean(axis=1)
    b = master.mean(axis=1)
    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert corr > 0.95, f"对齐后相关性只有 {corr:.4f}（lag={lag}）"
    assert abs(app_module._estimate_lag(aligned, master)) <= 1


# ---------------------------------------------------------------------------
# 等响度 / 峰值 / 线性
# ---------------------------------------------------------------------------


def test_prepare_pair_loudness_and_peak(segment_pair):
    master, candidate, _ = segment_pair
    out_ll, out_mp3, info = app_module._prepare_pair(master, candidate)

    assert len(out_ll) == app_module.SEGMENT_SAMPLES
    assert len(out_mp3) == app_module.SEGMENT_SAMPLES
    assert info["lufs_diff"] <= app_module.MAX_LUFS_DIFF_LU, info
    assert info["true_peak_db"] <= app_module.TRUE_PEAK_DBTP + app_module.TRUE_PEAK_SLACK_DB, info
    for value in info["lufs"].values():
        # 紧贴上界：说明响度目标真的达到了（没有被共同峰值衰减吃掉）
        assert app_module.TARGET_LUFS - 0.5 <= value <= app_module.TARGET_LUFS + 1e-6, info
    assert info["gain_db"]["common_delta"] <= 0.0
    # 我们的合成素材波峰因数约 12.7 dB，-18 LUFS 下真峰值约 -5.3 dBTP：
    # 距离 -1 dBTP 上限还应有 ≥2 dB 余量，否则说明响度定得太高、动态段落会被压回。
    assert info["true_peak_db"] <= app_module.TRUE_PEAK_DBTP - 2.0, info


def test_gain_is_pure_linear_no_limiter(segment_pair):
    """输出必须是输入的常数倍（防止日后塞进限幅器/压缩器，那会引入额外失真线索）。"""
    master, candidate, _ = segment_pair
    out_ll, out_mp3, _ = app_module._prepare_pair(master, candidate)
    for source, result in ((master, out_ll), (candidate, out_mp3)):
        fit = app_module._fit_length(source)[0]
        mask = np.abs(fit) > 1e-4
        ratios = (result[mask] / fit[mask]).astype(np.float64)
        assert np.std(ratios) < 1e-3, f"增益不是常数，std={np.std(ratios):.6f}"
        assert np.allclose(ratios, np.mean(ratios), atol=2e-3)


def test_prepare_pair_identical_input_is_identical_output():
    """安慰剂：同一片段两份无损拷贝 → 输出完全一致。"""
    master = synth_signal(app_module.SEGMENT_SECONDS, seed=13)
    out_ll, out_mp3, info = app_module._prepare_pair(master, master)
    assert info["lufs_diff"] <= 1e-9
    assert np.array_equal(out_ll, out_mp3)


def test_loudness_match_across_bitrates(segment_pair, tmp_path):
    master, _, flac = segment_pair
    for bitrate in (32, 96, 320):
        mp3 = encode_mp3(flac, tmp_path / f"seg_{bitrate}.mp3", bitrate)
        _, _, info = app_module._prepare_pair(master, decode_f32(mp3))
        assert info["lufs_diff"] <= app_module.MAX_LUFS_DIFF_LU, (bitrate, info)
        assert info["true_peak_db"] <= app_module.TRUE_PEAK_DBTP + app_module.TRUE_PEAK_SLACK_DB
        assert 0 <= info["lag_samples"] <= app_module.ALIGN_MAX_LAG


# ---------------------------------------------------------------------------
# 片段选择（走真实 API 路径）
# ---------------------------------------------------------------------------


def _segments_meta(audio_id: str) -> dict:
    return json.loads((app_module.CACHE_DIR / audio_id / "meta.json").read_text("utf-8"))


def test_segment_selection_is_deterministic_and_non_overlapping(client, fast_config, long_source):
    item = upload(client, long_source)
    start_session(client, [item["audio_id"]])
    first = _segments_meta(item["audio_id"])

    start_session(client, [item["audio_id"]])
    second = _segments_meta(item["audio_id"])

    assert first["segments"] == second["segments"], "重复出题不应改变片段选择"
    assert [seg["index"] for seg in first["segments"]] == [0, 1], "65 秒素材应切出 seg_0 与 seg_1"
    gap = abs(first["segments"][1]["start"] - first["segments"][0]["start"])
    assert gap >= app_module.SEGMENT_SECONDS, "seg_0 与 seg_1 不允许重叠"
    assert first["analysis"]["window_count"] == 11  # 1 + floor((65-15)/5)
    assert len(first["windows"]) == 11
    for seg in first["segments"]:
        assert (app_module.CACHE_DIR / item["audio_id"] / seg["file"]).exists()


def test_source_under_60s_has_single_segment(client, fast_config, short_source):
    item = upload(client, short_source)
    start_session(client, [item["audio_id"]])
    meta = _segments_meta(item["audio_id"])
    assert [seg["index"] for seg in meta["segments"]] == [0]


def test_segment_algo_version_invalidates_cache(client, fast_config, long_source, monkeypatch):
    item = upload(client, long_source)
    start_session(client, [item["audio_id"]])
    original = app_module.SEGMENT_ALGO_VERSION
    assert _segments_meta(item["audio_id"])["segment_algo_version"] == original

    monkeypatch.setattr(app_module, "SEGMENT_ALGO_VERSION", original + 1)
    start_session(client, [item["audio_id"]])
    assert _segments_meta(item["audio_id"])["segment_algo_version"] == original + 1


def test_stimuli_algo_version_moves_cache_dir(monkeypatch):
    audio_id = "deadbeefdeadbeef"
    before = app_module._stimuli_dir(audio_id)
    monkeypatch.setattr(app_module, "STIMULI_ALGO_VERSION", app_module.STIMULI_ALGO_VERSION + 1)
    after = app_module._stimuli_dir(audio_id)
    assert before != after
    assert f"v{app_module.STIMULI_ALGO_VERSION}" in str(after)


# ---------------------------------------------------------------------------
# 片段切片精度（防止 -ss 位置变化导致静音/错位）
# ---------------------------------------------------------------------------


def test_cut_segment_is_sample_accurate(isolated_data, tmp_path):
    full_wav = write_source(tmp_path / "full.wav", 65.0, seed=21)
    flac = tmp_path / "full.flac"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(full_wav),
         "-map_metadata", "-1", "-ar", "48000", "-ac", "2", "-c:a", "flac", str(flac)],
        check=True, stdin=subprocess.DEVNULL,
    )
    start_sample = 592560  # 12.345 s
    out = tmp_path / "seg.flac"
    asyncio.run(app_module._cut_segment(flac, out, start_sample / app_module.SAMPLE_RATE))

    cut = app_module._read_audio(out)
    full = app_module._read_audio(flac)
    reference = full[start_sample:start_sample + app_module.SEGMENT_SAMPLES]

    assert cut.shape == (app_module.SEGMENT_SAMPLES, 2)
    assert float(np.max(np.abs(cut))) > 1e-3, "切片不应该是静音"
    # 首尾各 20 ms 淡变，跳过 50 ms 后应逐采样一致
    edge = int(0.05 * app_module.SAMPLE_RATE)
    middle = slice(edge, app_module.SEGMENT_SAMPLES - edge)
    assert float(np.max(np.abs(cut[middle] - reference[middle]))) < 1e-6


def test_cut_segment_drops_attached_picture(isolated_data, tmp_path):
    """带内嵌封面的源在 start=0 时也必须只切出音频流。

    回归：封面（attached_pic 视频流）一旦混进片段，MP3 编码就会以
    rc=234「Attached pictures were requested, but the ID3v2 header is disabled」失败。
    起点必须正好是 0——起点大于 0 时 ffmpeg 本来就会漏掉封面帧，那样测不出这个 bug。
    """
    base = tmp_path / "base.flac"
    sf.write(str(base), synth_signal(35.0, seed=11), SR, subtype="PCM_16", format="FLAC")
    flac = attach_cover_flac(base, tmp_path / "withart.flac")
    assert stream_types(flac) == ["audio", "video"], "前置条件：源文件应当带封面流"

    out = tmp_path / "seg.flac"
    asyncio.run(app_module._cut_segment(flac, out, 0.0))

    assert stream_types(out) == ["audio"], "片段里不允许残留封面流"
    assert sf.info(str(out)).frames == app_module.SEGMENT_SAMPLES


def test_import_strips_attached_picture(client, art_source):
    """导入时只映射音频流：库内母版不应再带封面（否则等于给下游埋雷）。"""
    item = upload(client, art_source)
    master = app_module.LIBRARY_DIR / f"{item['audio_id']}.flac"
    assert stream_types(master) == ["audio"]
    entry = next(e for e in app_module._library()["items"] if e["audio_id"] == item["audio_id"])
    assert entry["source_codec"] == "flac"


# ---------------------------------------------------------------------------
# 刺激缓存
# ---------------------------------------------------------------------------


def test_stimuli_cached_and_reusable(client, fast_config, long_source):
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    start_session(client, [audio_id])
    files = cache_wavs(audio_id)
    assert files, "应生成刺激缓存"
    mtimes = {path.name: path.stat().st_mtime_ns for path in files}

    second = start_session(client, [audio_id])
    assert {path.name: path.stat().st_mtime_ns for path in cache_wavs(audio_id)} == mtimes, (
        "命中缓存不应重写文件"
    )
    assert second["stimuli"]["generated"] == 0, "第二轮不应重新生成任何刺激"
    assert second["stimuli"]["reused"] == second["total"]

    index = json.loads((app_module._stimuli_dir(audio_id) / "index.json").read_text("utf-8"))
    assert index["stimuli_algo_version"] == app_module.STIMULI_ALGO_VERSION
    coded = index["entries"]["seg0_96k"]
    assert coded["n_samples"] == app_module.SEGMENT_SAMPLES
    assert coded["lufs_diff"] <= app_module.MAX_LUFS_DIFF_LU
    assert coded["true_peak_db"] <= app_module.TRUE_PEAK_DBTP + app_module.TRUE_PEAK_SLACK_DB


def test_placebo_pair_is_single_identical_file(client, fast_config, long_source):
    item = upload(client, long_source)
    start_session(client, [item["audio_id"]])
    index = json.loads(
        (app_module._stimuli_dir(item["audio_id"]) / "index.json").read_text("utf-8")
    )
    placebo_entries = [e for e in index["entries"].values() if e["bitrate"] is None]
    assert placebo_entries, "安慰剂刺激应被缓存"
    for placebo in placebo_entries:
        assert placebo["files"]["ll"] == placebo["files"]["mp3"], "安慰剂只应写一份文件"
        assert placebo["lufs_diff"] <= 1e-9
        assert placebo["true_peak_db"] <= app_module.TRUE_PEAK_DBTP + app_module.TRUE_PEAK_SLACK_DB


def test_stimulus_wav_headers_are_canonical(client, fast_config, long_source):
    item = upload(client, long_source)
    start_session(client, [item["audio_id"]])
    for path in cache_wavs(item["audio_id"]):
        info = sf.info(str(path))
        assert info.samplerate == app_module.SAMPLE_RATE
        assert info.channels == app_module.CHANNELS
        assert info.frames == app_module.SEGMENT_SAMPLES
        assert info.subtype == "FLOAT"
        assert info.format == "WAV"


# ---------------------------------------------------------------------------
# 选曲顺序 / 并发生成 / 准备进度
# ---------------------------------------------------------------------------


def test_selection_order_does_not_change_stimulus_set(
    client, fast_config, long_source, short_source
):
    """前端按**勾选顺序**发送 audio_ids；题表与缓存命中不能因此改变。

    否则同一批素材、只是勾选先后不同，第二轮就会"部分复用、部分重生成"。
    """
    a = upload(client, long_source)["audio_id"]
    b = upload(client, short_source)["audio_id"]
    first = start_session(client, [a, b])
    assert first["stimuli"]["generated"] == first["total"]

    second = start_session(client, [b, a])
    assert second["stimuli"]["generated"] == 0
    assert second["stimuli"]["reused"] == second["total"]


def test_concurrent_stimuli_generation_is_identical_to_serial(client, monkeypatch, long_source):
    """并发生成：索引条目不丢（读改写竞态守护），解码后的样本与串行完全一致。

    注意比较的是**解码后的样本**而不是文件字节：libsndfile 会写一个带时间戳的 `PEAK` chunk，
    所以同一段数据两次写出的 WAV 文件字节必然不同（音频内容仍然完全一致）。
    """
    monkeypatch.setattr(app_module, "SHORT_TEST_BITRATES", [64, 128, 320])
    monkeypatch.setattr(app_module, "SHORT_TEST_TOTAL", 5)  # 4 真实 + 1 安慰剂
    audio_id = upload(client, long_source)["audio_id"]

    def run(workers: int) -> tuple[set, dict]:
        client.delete("/api/cache")
        monkeypatch.setattr(app_module, "STIMULI_CONCURRENCY", workers)
        started = start_session(client, [audio_id])
        root = app_module._stimuli_dir(audio_id)
        index = json.loads((root / "index.json").read_text("utf-8"))
        expected = started["total"] + started["stimuli"]["prewarmed"]
        assert len(index["entries"]) == expected, "并发下索引条目不能丢"
        digests = {}
        for entry in index["entries"].values():
            for name in entry["files"].values():
                samples, _ = sf.read(str(root / name), dtype="float32")
                digests[name] = hashlib.sha256(samples.tobytes()).hexdigest()
        return set(index["entries"]), digests

    keys_parallel, digests_parallel = run(4)
    keys_serial, digests_serial = run(1)
    assert keys_parallel == keys_serial
    assert digests_parallel == digests_serial


def test_concurrency_autodetect(monkeypatch):
    monkeypatch.delenv("TINEAR_CONCURRENCY", raising=False)
    detected = app_module._resolve_concurrency()
    assert 1 <= detected <= app_module.STIMULI_CONCURRENCY_MAX
    assert app_module.FFMPEG_CONCURRENCY >= 1

    monkeypatch.setenv("TINEAR_CONCURRENCY", "3")
    assert app_module._resolve_concurrency() == 3
    monkeypatch.setenv("TINEAR_CONCURRENCY", "0")  # 非法值 → 回退自动探测
    assert app_module._resolve_concurrency() == detected


def test_shared_locks_survive_multiple_event_loops(client, long_source):
    """回归：模块级 `asyncio.Lock` 一旦被**争用**就绑定当时的事件循环，跨循环再争用会抛
    `is bound to a different event loop`。pytest 每个用例都是新循环，所以共享资源的锁
    必须按「事件循环 + key」分桶（`_loop_scoped_lock`）。"""
    audio_id = upload(client, long_source)["audio_id"]
    cache_root = app_module.CACHE_DIR / audio_id

    async def cold_segments():
        # 两个任务同时要算片段：冷缓存 → 后到的那个必然在锁上等一次（这就是"争用"）
        await asyncio.gather(
            app_module._ensure_segments(audio_id),
            app_module._ensure_segments(audio_id),
        )

    for _ in range(2):  # 两轮各跑在一个全新的 asyncio.run 循环里
        shutil.rmtree(cache_root, ignore_errors=True)
        asyncio.run(cold_segments())
    assert (cache_root / "meta.json").exists()


def test_detailed_round_prewarms_short_complement(client, fast_config, long_source):
    """详细模式顺手生成"快速模式才用到的那几对" → 之后切快速是 0 新生成。

    快速 → 详细那 17 个组合的缺口是阶梯覆盖问题（详细 10 档 × 2 片段，快速只覆盖其中 5 个），
    预热消不掉；但"详细 → 快速"这一侧的缺口可以被预热补平。
    """
    audio_id = upload(client, long_source)["audio_id"]

    detailed = start_session(client, [audio_id], "detailed")
    assert detailed["stimuli"]["generated"] == detailed["total"]
    assert detailed["stimuli"]["prewarmed"] >= 1, "冷缓存下详细模式应当顺手预热快速模式的补集"

    short = start_session(client, [audio_id], "short")
    assert short["stimuli"]["generated"] == 0, "预热之后切快速不应再新生成"
    assert short["stimuli"]["reused"] == short["total"]


def test_prewarm_is_detailed_only_and_switchable(client, fast_config, long_source, monkeypatch):
    audio_id = upload(client, long_source)["audio_id"]

    short = start_session(client, [audio_id], "short")
    assert short["stimuli"]["prewarmed"] == 0, "快速模式没有可预热的补集"

    monkeypatch.setattr(app_module, "PREWARM_SHORT_COMPLEMENT", False)
    client.delete("/api/cache")
    detailed = start_session(client, [audio_id], "detailed")
    assert detailed["stimuli"]["prewarmed"] == 0, "关掉开关就不该预热"

    short = start_session(client, [audio_id], "short")
    assert short["stimuli"]["generated"] == 1, "关掉预热后，切快速仍要补那 1 个组合"


def test_progress_tracking_lifecycle():
    token = "t-expired"
    app_module._progress_begin(token, phase="stimuli", total=3)
    app_module.PROGRESS[token]["updated_at"] = time.time() - app_module.PROGRESS_TTL_SECONDS - 1
    app_module._progress_purge()
    assert token not in app_module.PROGRESS

    token = "t-fresh"
    app_module._progress_begin(token, phase="segments", total=2)
    app_module._progress_update(token, done=1)
    assert (app_module.PROGRESS[token]["done"], app_module.PROGRESS[token]["phase"]) == (1, "segments")
    app_module._progress_update(token, phase="stimuli", done=0, total=5)
    app_module._progress_update(token, done=5)
    app_module._progress_finish(token)
    entry = app_module.PROGRESS[token]
    assert entry["finished"] is True and entry["error"] is None and entry["done"] == 5
    app_module._progress_purge()  # 未过期 → 保留
    assert token in app_module.PROGRESS
    app_module.PROGRESS.pop(token, None)


# ---------------------------------------------------------------------------
# 出题（乱序 / 覆盖面）
# ---------------------------------------------------------------------------


def _questions(spy_seed: int) -> list:
    """用固定随机种子调 _build_questions，结果可复现。"""
    import secrets as secrets_module

    original = secrets_module.randbits
    secrets_module.randbits = lambda _bits: spy_seed
    try:
        return app_module._build_questions("detailed", ["audio1"], {"audio1": [0, 1]})
    finally:
        secrets_module.randbits = original


def test_questions_shuffle_but_keep_identical_stimulus_set(fast_config):
    """安慰剂与真实题一起打乱：顺序变、用到的 (片段, 码率) 集合不变。"""
    combos = set()
    orders = set()
    placebo_positions = set()
    for seed in range(12):
        questions = _questions(seed)
        combos.add(tuple(sorted((str(q.bitrate), q.seg_index) for q in questions)))
        orders.add(tuple(q.bitrate for q in questions))
        placebo_positions.add(tuple(i for i, q in enumerate(questions) if q.bitrate is None))
        assert [q.index for q in questions] == list(range(len(questions)))
        assert [q.question_id for q in questions] == [f"q{i}" for i in range(len(questions))]
    assert len(combos) == 1, "每轮用到的刺激集合必须完全一致，否则缓存复用会退化"
    assert len(orders) > 1, "顺序应当被打乱"
    assert len(placebo_positions) > 1, "安慰剂题的位置也应当被打乱（否则可以被预判）"


def test_questions_cover_each_bitrate_evenly(fast_config, monkeypatch):
    monkeypatch.setattr(app_module, "DETAILED_TEST_BITRATES", [32, 64, 128])
    monkeypatch.setattr(app_module, "DETAILED_TEST_TOTAL", 8)  # 6 真实 + 2 安慰剂 → 每档 2 题
    questions = app_module._build_questions("detailed", ["audio1"], {"audio1": [0, 1]})
    assert len(questions) == 8
    assert sum(1 for q in questions if q.bitrate is None) == 2
    for bitrate in (32, 64, 128):
        assert sum(1 for q in questions if q.bitrate == bitrate) == 2


# ---------------------------------------------------------------------------
# 统计口径（H0 = 三选一瞎猜基线 1/3）
# ---------------------------------------------------------------------------


def test_binom_p_value_and_required_n():
    """逐档 n=1 是结构性无法判定；每档 7 题才能在 80% 功效下做出判断。"""
    assert app_module._binom_p_value(0, 0) is None
    assert app_module._binom_p_value(1, 1) == pytest.approx(1 / 3)  # n=1 永远不可能显著
    assert app_module._binom_p_value(2, 2) == pytest.approx(1 / 9)
    assert app_module._binom_p_value(3, 3) == pytest.approx(1 / 27)
    assert app_module._binom_p_value(4, 5) == pytest.approx(0.0453, abs=1e-4)
    assert app_module._binom_p_value(7, 10) == pytest.approx(0.0197, abs=1e-4)
    assert app_module.POWER_REQUIRED_N == 7
    assert app_module._required_n(p1=0.5) == 58  # 越接近基线越难判


def test_row_stats_marks_small_samples_as_undecidable():
    tiny = app_module._row_stats(1, 1)
    assert tiny["p_value"] == pytest.approx(1 / 3, abs=1e-4)
    assert tiny["significant"] is False
    assert tiny["status"] == "insufficient"
    assert "样本不足" in tiny["note"]

    clear = app_module._row_stats(5, 5)
    assert clear["significant"] is True
    assert clear["status"] == "significant"
    assert "显著高于瞎猜基线" in clear["note"]

    noise = app_module._row_stats(1, 4)
    assert noise["significant"] is False
    assert noise["status"] == "not_significant"
    assert "无显著差异" in noise["note"]

    empty = app_module._row_stats(0, 0)
    assert empty["p_value"] is None and empty["accuracy"] is None
    assert empty["significant"] is False
    assert empty["status"] == "insufficient"
