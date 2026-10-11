"""音频管线测试：常量守护、对齐、等响度、线性增益、片段选择、刺激缓存。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import threading
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


def test_lossless_side_is_bit_identical_across_bitrates(segment_pair, tmp_path):
    """无损侧共享的**前提**：它的内容只由 (片段, common_delta) 决定。

    `master_fit` 与 `g_ll` 都只依赖片段，所以只要 `common_delta == 0`（默认情况：-18 LUFS
    下真峰值还有 4 dB 余量），不同码率算出的无损侧必须**逐位相同** —— `_ensure_stimuli`
    正是靠这一条才敢让整个片段共用一份 `seg{S}_ll.wav`。前提不成立时代码会退回每档一份
    （结果仍然正确，只是空间优化失效），所以这条挂了说明"优化失效"，不是"算错了"。
    """
    master, _, flac = segment_pair
    canonical: list[np.ndarray] = []
    for bitrate in (64, 128, 320):
        mp3 = encode_mp3(flac, tmp_path / f"share_{bitrate}.mp3", bitrate)
        out_ll, _, info = app_module._prepare_pair(master, decode_f32(mp3))
        if info["gain_db"]["common_delta"] == 0.0:
            canonical.append(out_ll)
    assert len(canonical) >= 2, "本素材在 -18 LUFS 下应有多档 delta == 0，才能验证共享前提"
    for out_ll in canonical[1:]:
        assert np.array_equal(canonical[0], out_ll), "delta == 0 时无损侧必须逐位相同"


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


def test_lossless_reference_is_shared_across_bitrates(client, monkeypatch, long_source):
    """同一个片段的所有码率共用一份无损侧（delta == 0 时的确定性去重）。

    细测 20 对真实测试音频的无损侧因此从 20 个文件降到"每片段 1 个"（2 个）。
    安慰剂 A/B 本来就同文件，delta == 0 时它和无损侧是同一份内容，一并共用。
    """
    monkeypatch.setattr(app_module, "SHORT_TEST_BITRATES", [64, 128, 320])
    monkeypatch.setattr(app_module, "SHORT_TEST_TOTAL", 7)  # 6 真实 + 1 安慰剂 → 每片段 3 档
    audio_id = upload(client, long_source)["audio_id"]
    start_session(client, [audio_id])

    root = app_module._stimuli_dir(audio_id)
    index = json.loads((root / "index.json").read_text("utf-8"))
    entries = index["entries"]
    real = [e for e in entries.values() if e["bitrate"] is not None]
    segs = sorted({e["seg"] for e in real})
    assert segs == [0, 1], "65 秒素材应切出两个片段"
    assert len(real) == 6 and len(entries) == 7

    # 每个片段一份无损侧，该片段的所有题目（真实 + 安慰剂）都指向它
    for seg in segs:
        consumers = [e for e in entries.values() if e["seg"] == seg]
        assert {e["files"]["ll"] for e in consumers} == {f"seg{seg}_ll.wav"}
        for entry in consumers:
            assert entry["gain_db"]["common_delta"] == 0.0, entry["key"]
            assert entry["lufs_diff"] <= app_module.MAX_LUFS_DIFF_LU
            assert entry["true_peak_db"] <= app_module.TRUE_PEAK_DBTP + app_module.TRUE_PEAK_SLACK_DB

    # 磁盘上正好是：每片段 1 个无损侧 + 6 个压缩侧（安慰剂不再单独占文件）
    expected = {f"seg{s}_ll.wav" for s in segs} | {
        f"seg{e['seg']}_{e['bitrate']}k_mp3.wav" for e in real
    }
    assert len(expected) == 8, "6 对真实测试音频：2 个共享无损侧 + 6 个压缩侧"
    assert {path.name for path in cache_wavs(audio_id)} == expected


def test_sharing_falls_back_when_peak_attenuation_is_needed(
    client, fast_config, long_source, monkeypatch
):
    """把 TARGET_LUFS 调高到触发共同峰值衰减时，无损侧必须退回每档一份。

    无损侧的内容由 (片段, common_delta) 共同决定，所以共享的判据只能是
    `common_delta == 0`：一旦某对题的真峰值顶到 -1 dBTP（这里 -6 LUFS 必然发生），
    那份 `_ll` 就只属于它自己，不能拿 `seg{S}_ll.wav` 去顶。
    """
    monkeypatch.setattr(app_module, "TARGET_LUFS", -6.0)
    audio_id = upload(client, long_source)["audio_id"]
    start_session(client, [audio_id])

    root = app_module._stimuli_dir(audio_id)
    index = json.loads((root / "index.json").read_text("utf-8"))
    entries = index["entries"]
    shared_names = {f"seg{seg}_ll.wav" for seg in {e["seg"] for e in entries.values()}}
    fell_back: list[str] = []

    for key, entry in entries.items():
        assert entry["lufs_diff"] <= app_module.MAX_LUFS_DIFF_LU
        assert entry["true_peak_db"] <= app_module.TRUE_PEAK_DBTP + app_module.TRUE_PEAK_SLACK_DB
        shared = entry["gain_db"]["common_delta"] == 0.0
        if entry["bitrate"] is None:
            # 安慰剂 A/B 是同一份文件；delta == 0 时它就是该片段的无损侧
            assert entry["files"]["ll"] == entry["files"]["mp3"]
        if shared:
            expected = f"seg{entry['seg']}_ll.wav"
        else:
            expected = f"{key}.wav" if entry["bitrate"] is None else f"{key}_ll.wav"
            fell_back.append(key)
        assert entry["files"]["ll"] == expected, (key, entry["files"], entry["gain_db"])
        assert shared or entry["files"]["ll"] not in shared_names, "delta < 0 的题目不能吃共享的无损侧"

    assert fell_back, "这个响度目标必须至少触发一次峰值衰减，否则测不到回退分支"
    assert not any((root / name).exists() for name in set(fell_back) & shared_names)


def test_stimulus_wav_headers_are_canonical(client, fast_config, long_source):
    item = upload(client, long_source)
    start_session(client, [item["audio_id"]])
    for path in cache_wavs(item["audio_id"]):
        info = sf.info(str(path))
        assert info.samplerate == app_module.SAMPLE_RATE
        assert info.channels == app_module.CHANNELS
        assert info.frames == app_module.SEGMENT_SAMPLES
        assert info.subtype == app_module.STIMULUS_SUBTYPE
        assert info.format == "WAV"
        # 固定位深 → 字节数只由长度与位深决定、与内容无关（验收 #8 的"两路等长"靠这一条）。
        # 整数位深还顺带没有 libsndfile 的 PEAK chunk（那个 chunk 带时间戳），所以整个文件
        # 就是 44 字节规范头 + 采样数据，连字节级可比。
        width = app_module.STIMULUS_BITS[app_module.STIMULUS_SUBTYPE] // 8
        payload = app_module.SEGMENT_SAMPLES * app_module.CHANNELS * width
        header = path.stat().st_size - payload
        assert header == (88 if app_module.STIMULUS_SUBTYPE == "FLOAT" else 44), (path.name, header)


def test_stimulus_subtype_lsb_mapping(monkeypatch):
    """位深常量与量化台阶必须对得上（`STIMULUS_BITS` 是唯一的换算来源）。"""
    assert app_module.STIMULUS_BITS[app_module.STIMULUS_SUBTYPE] == 24
    assert app_module._stimulus_lsb() == 2.0 ** -23
    monkeypatch.setattr(app_module, "STIMULUS_SUBTYPE", "PCM_16")
    assert app_module._stimulus_lsb() == 2.0 ** -15
    monkeypatch.setattr(app_module, "STIMULUS_SUBTYPE", "FLOAT")
    assert app_module._stimulus_lsb() == 0.0
    monkeypatch.setattr(app_module, "STIMULUS_SUBTYPE", "MP3")
    with pytest.raises(app_module.AudioToolError):
        app_module._stimulus_lsb()


def test_written_pair_verification_rejects_mismatch(tmp_path):
    """落盘自检必须真的会拦 —— 否则它只是一段装饰代码。"""
    seconds = app_module.SEGMENT_SECONDS
    source = synth_signal(seconds, seed=9)
    path = tmp_path / "pair.wav"
    sf.write(str(path), source, SR, subtype=app_module.STIMULUS_SUBTYPE, format="WAV")

    app_module._verify_written_pair(path, path, source, source)  # 正常路径不抛

    tampered = source.copy()
    tampered[1000] += 0.5  # 相当于"落盘的不只是量化"
    with pytest.raises(app_module.AudioToolError, match="不只是量化误差"):
        app_module._verify_written_pair(path, path, tampered, tampered)

    short = tmp_path / "short.wav"
    half = app_module.SEGMENT_SAMPLES // 2
    sf.write(str(short), np.zeros((half, app_module.CHANNELS), dtype=np.float32), SR,
             subtype=app_module.STIMULUS_SUBTYPE, format="WAV")
    with pytest.raises(app_module.AudioToolError, match="落盘长度异常"):
        app_module._verify_written_pair(path, short, source, source[:half])


def test_stimulus_metadata_matches_written_files(client, fast_config, long_source):
    """索引里的 LUFS / 真峰值必须描述**落盘文件**，而不只是内存里的 float 数组。

    改成整数位深（`STIMULUS_SUBTYPE = "PCM_24"`）之后，交付的字节是那份数组的量化结果，
    所以 `_prepare_pair` 的自检（只证明内存数组合规）不够；`_ensure_stimuli` 补的是
    "落盘内容只是量化"这条结构事实，这里则**实测**这条推断的结论：把文件重新解码、重算
    LUFS 与真峰值，与索引里的记录逐项比对。量化带来的偏差应 ≤ 1e-3 dB，而容差是 0.05 dB。
    """
    audio_id = upload(client, long_source)["audio_id"]
    start_session(client, [audio_id])
    root = app_module._stimuli_dir(audio_id)
    index = json.loads((root / "index.json").read_text("utf-8"))
    assert index["stimuli_algo_version"] == app_module.STIMULI_ALGO_VERSION
    assert index["entries"], "应生成测试音频缓存"

    measured: dict[str, tuple] = {}
    for entry in index["entries"].values():
        assert entry["lufs_diff"] <= app_module.MAX_LUFS_DIFF_LU, entry["key"]
        for role, name in entry["files"].items():
            if name not in measured:
                path = root / name
                data, sr = sf.read(str(path), dtype="float32", always_2d=True)
                assert sr == app_module.SAMPLE_RATE
                assert len(data) == app_module.SEGMENT_SAMPLES
                assert sf.info(str(path)).subtype == app_module.STIMULUS_SUBTYPE
                measured[name] = (
                    app_module._integrated_loudness(data),  # 实测，不是索引里的值
                    app_module._true_peak_db(data),
                    path.stat().st_size,
                )
            loud, _, _ = measured[name]
            assert abs(loud - entry["lufs"][role]) <= 1e-3, (name, loud, entry["lufs"][role])
        # 索引里的 true_peak_db 记的是**这一对的最坏一路**（`max(tp_a, tp_b)`），
        # 而不是无损侧自己的峰值，所以按对比较。
        pair_peak = max(measured[entry["files"]["ll"]][1], measured[entry["files"]["mp3"]][1])
        assert abs(pair_peak - entry["true_peak_db"]) <= 1e-3, (entry["key"], pair_peak)
        assert pair_peak <= app_module.TRUE_PEAK_DBTP + app_module.TRUE_PEAK_SLACK_DB, entry["key"]
        # 两路字节数必须相等，否则体积本身就是答案
        assert measured[entry["files"]["ll"]][2] == measured[entry["files"]["mp3"]][2]


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


def test_write_wav_atomic_retries_windows_sharing_violation(tmp_path, monkeypatch):
    """Windows 回归：目标文件被别的句柄占着时 `os.replace` 抛 WinError 5（PermissionError），
    必须有限重试，而不是把 500 抛给用户（真机报错见 MEASUREMENTS.md 的记录）。

    模拟方式：前两次替换抛 `PermissionError`，第三次放行。
    """
    target = tmp_path / "seg0_ll.wav"
    data = np.zeros((2048, 2), dtype=np.float32)
    real_replace = os.replace
    attempts = {"n": 0}

    def flaky_replace(src, dst):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError(13, "拒绝访问。")
        real_replace(src, dst)

    monkeypatch.setattr(app_module, "_REPLACE_RETRY_DELAYS", (0.0, 0.0, 0.0, 0.0))
    monkeypatch.setattr(app_module.os, "replace", flaky_replace)

    app_module._write_wav_atomic(target, data)

    assert attempts["n"] == 3, "被占用时应当重试"
    written, sr = sf.read(str(target), dtype="float32")
    assert sr == app_module.SAMPLE_RATE
    assert written.shape == data.shape
    assert not list(tmp_path.glob(".*.tmp")), "成功后不应留下临时文件"


def test_write_wav_atomic_gives_up_and_cleans_tmp(tmp_path, monkeypatch):
    """占用一直不释放时要把异常抛出去（不吞真错误），并且不留半成品临时文件。"""

    def always_busy(src, dst):
        raise PermissionError(5, "拒绝访问。")

    monkeypatch.setattr(app_module, "_REPLACE_RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(app_module.os, "replace", always_busy)

    with pytest.raises(PermissionError):
        app_module._write_wav_atomic(tmp_path / "seg0_ll.wav", np.zeros((256, 2), dtype=np.float32))
    assert not (tmp_path / "seg0_ll.wav").exists()
    assert list(tmp_path.rglob(".*.tmp")) == [], "重试失败后临时文件不该留下"


def test_replace_with_retry_does_not_swallow_real_errors(tmp_path, monkeypatch):
    """只有"被占用"这一类才重试；真正的参数/权限错误必须一次就抛。"""
    attempts = {"n": 0}

    def broken(src, dst):
        attempts["n"] += 1
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(app_module, "_REPLACE_RETRY_DELAYS", (0.0, 0.0, 0.0))
    monkeypatch.setattr(app_module.os, "replace", broken)

    with pytest.raises(OSError):
        app_module._replace_with_retry(tmp_path / "a", tmp_path / "b")
    assert attempts["n"] == 1


def test_shared_stimulus_write_and_verify_are_serialized(client, monkeypatch, long_source):
    """根因守护：同一共享无损文件的「写」不能与它的「落盘自检读」重叠。

    Windows 上 `os.replace` 的目标只要还有读句柄开着就抛 WinError 5 —— 并发准备时，
    一个码率任务正在 `_verify_written_pair` 里读 `seg{S}_ll.wav`，另一个码率任务去替换它，
    `/api/start` 就 500（真机 traceback）。这里把自检的读窗口人为拉长，若写/读没有按路径
    串行（修复前），并发任务必然被抓到重叠；修复后 `overlaps` 必须为空。
    """
    monkeypatch.setattr(app_module, "SHORT_TEST_BITRATES", [64, 96, 128, 160, 320])
    monkeypatch.setattr(app_module, "SHORT_TEST_TOTAL", 6)  # 5 真实 + 1 安慰剂
    audio_id = upload(client, long_source)["audio_id"]

    guard = threading.Lock()
    reading: set[str] = set()
    writes: dict[str, int] = {}
    overlaps: list[str] = []

    real_verify = app_module._verify_written_pair
    real_write = app_module._write_wav_atomic

    def slow_verify(path_ll, path_mp3, out_ll, out_mp3):
        names = {path_ll.name, path_mp3.name}
        with guard:
            reading.update(names)
        try:
            time.sleep(0.05)  # 拉长"读句柄开着"的窗口
            return real_verify(path_ll, path_mp3, out_ll, out_mp3)
        finally:
            with guard:
                reading.difference_update(names)

    def watching_write(path, data):
        with guard:
            writes[path.name] = writes.get(path.name, 0) + 1
            if path.name in reading:
                overlaps.append(path.name)
        return real_write(path, data)

    monkeypatch.setattr(app_module, "_verify_written_pair", slow_verify)
    monkeypatch.setattr(app_module, "_write_wav_atomic", watching_write)

    start_session(client, [audio_id])  # 冷缓存：同一片段的每个码率都要写同一个 seg{S}_ll.wav

    shared_writes = sum(n for name, n in writes.items() if name.endswith("_ll.wav"))
    assert shared_writes >= 2, "本用例必须真的出现「多个任务写同一份共享无损文件」才有意义"
    assert overlaps == [], f"写入 {overlaps} 时该文件正被自检读取（Windows 上会 WinError 5）"


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
