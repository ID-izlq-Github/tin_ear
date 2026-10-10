"""音频管线测试：常量守护、对齐、等响度、线性增益、片段选择、刺激缓存。"""

from __future__ import annotations

import asyncio
import json
import subprocess

import numpy as np
import soundfile as sf

import app as app_module
from conftest import (
    cache_wavs,
    decode_f32,
    encode_mp3,
    start_session,
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
    assert app_module.TARGET_LUFS == -23.0
    assert app_module.TRUE_PEAK_DBTP == -1.0
    assert app_module.SHORT_TEST_BITRATES == [48, 96, 160, 224, 320]
    assert app_module.DETAILED_TEST_BITRATES == [32, 48, 64, 80, 96, 128, 160, 192, 256, 320]
    assert app_module.SHORT_TEST_TOTAL == 6
    assert app_module.DETAILED_TEST_TOTAL == 12
    assert app_module.SESSION_TTL_SECONDS == 7200
    # short 的档位并非 detailed 的子集（224 kbps 只在 short 里），两条阶梯的并集共 11 档
    union = set(app_module.SHORT_TEST_BITRATES) | set(app_module.DETAILED_TEST_BITRATES)
    assert union == {32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 320}
    assert 224 in app_module.SHORT_TEST_BITRATES and 224 not in app_module.DETAILED_TEST_BITRATES
    assert app_module._real_and_placebo_counts("short") == (5, 1)
    assert app_module._real_and_placebo_counts("detailed") == (10, 2)


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
        assert app_module.TARGET_LUFS - 6.0 <= value <= app_module.TARGET_LUFS + 1e-6, info
    assert info["gain_db"]["common_delta"] <= 0.0


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

    start_session(client, [audio_id])
    assert {path.name: path.stat().st_mtime_ns for path in cache_wavs(audio_id)} == mtimes, (
        "命中缓存不应重写文件"
    )

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
