"""API 测试：导入/去重/拒绝、删除、题目防泄漏、作答与统计、清理、路径穿越。"""

from __future__ import annotations

import os
import subprocess

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app as app_module
from conftest import (
    cache_wavs,
    start_session,
    stream_types,
    upload,
    write_source,
)


@pytest.fixture(scope="session")
def encoded_sources(synth_dir):
    """构造各种编码的源文件，用于验收「能导入什么 / 该拒绝什么」。"""
    base = write_source(synth_dir / "base.wav", 35.0, seed=31)
    out: dict[str, object] = {"wav": base, "flac_source": base}

    specs = {
        "mp3": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
        "aac": ["-c:a", "aac", "-f", "adts"],
        "ogg": ["-c:a", "libvorbis", "-f", "ogg"],
        "opus": ["-c:a", "libopus", "-f", "opus"],
    }
    for name, args in specs.items():
        path = synth_dir / f"base.{name}"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(base), *args, str(path)],
            check=True, stdin=subprocess.DEVNULL,
        )
        out[name] = path

    aiff = synth_dir / "base.aiff"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(base),
         "-c:a", "pcm_s16be", "-f", "aiff", str(aiff)],
        check=True, stdin=subprocess.DEVNULL,
    )
    out["aiff"] = aiff

    alt = write_source(synth_dir / "alt.wav", 35.0, seed=34)
    aiff_alt = synth_dir / "alt.aiff"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(alt),
         "-c:a", "pcm_s16be", "-f", "aiff", str(aiff_alt)],
        check=True, stdin=subprocess.DEVNULL,
    )
    out["wav_alt"] = alt
    out["aiff_alt"] = aiff_alt

    low_rate = synth_dir / "low_rate.wav"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(base),
         "-ar", "22050", "-c:a", "pcm_s16le", str(low_rate)],
        check=True, stdin=subprocess.DEVNULL,
    )
    out["low_rate"] = low_rate
    out["too_short"] = write_source(synth_dir / "too_short.wav", 12.0, seed=33)
    return out


# ---------------------------------------------------------------------------
# 导入 / 去重 / 拒绝
# ---------------------------------------------------------------------------


def test_import_lossless_formats_converts_to_48k_stereo_flac(client, encoded_sources):
    response = client.post(
        "/api/library/upload",
        files=[
            ("files", ("base.wav", encoded_sources["wav"].read_bytes())),
            ("files", ("alt.aiff", encoded_sources["aiff_alt"].read_bytes())),
        ],
    )
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert [r["status"] for r in results] == ["imported", "imported"], results

    for result in results:
        recorded = next(
            item for item in client.get("/api/library").json()["items"]
            if item["audio_id"] == result["audio_id"]
        )
        assert recorded["sample_rate"] == 48000
        assert recorded["channels"] == 2
        assert recorded["source_codec"] in ("pcm_s16le", "pcm_s16be")
        info = sf.info(str(app_module.LIBRARY_DIR / f"{result['audio_id']}.flac"))
        assert info.samplerate == 48000 and info.channels == 2


def test_same_audio_in_two_lossless_containers_dedupes(client, encoded_sources):
    """同一段音频的 WAV 与 AIFF 是同一内容 → 只应入库一次。"""
    response = client.post(
        "/api/library/upload",
        files=[
            ("files", ("base.wav", encoded_sources["wav"].read_bytes())),
            ("files", ("base.aiff", encoded_sources["aiff"].read_bytes())),
        ],
    )
    results = response.json()["results"]
    assert [r["status"] for r in results] == ["imported", "duplicate"], results
    assert results[0]["audio_id"] == results[1]["audio_id"]
    assert len(client.get("/api/library").json()["items"]) == 1


def test_import_accepts_flac_source(client, encoded_sources):
    response = client.post(
        "/api/library/upload",
        files={"files": ("base.aiff", encoded_sources["aiff"].read_bytes())},
    )
    assert response.json()["results"][0]["status"] == "imported"


@pytest.mark.parametrize("kind", ["mp3", "aac", "ogg", "opus"])
def test_lossy_formats_are_rejected(client, encoded_sources, kind):
    response = client.post(
        "/api/library/upload",
        files={"files": (f"base.{kind}", encoded_sources[kind].read_bytes())},
    )
    result = response.json()["results"][0]
    assert result["status"] == "rejected", result
    assert result["audio_id"] is None
    assert "有损" in result["reason"], result


def test_low_sample_rate_and_short_duration_are_rejected(client, encoded_sources):
    response = client.post(
        "/api/library/upload",
        files=[
            ("files", ("low_rate.wav", encoded_sources["low_rate"].read_bytes())),
            ("files", ("too_short.wav", encoded_sources["too_short"].read_bytes())),
        ],
    )
    results = response.json()["results"]
    assert [r["status"] for r in results] == ["rejected", "rejected"]
    assert "采样率" in results[0]["reason"]
    assert "至少需要" in results[1]["reason"]
    assert client.get("/api/library").json()["items"] == []


def test_not_audio_is_rejected(client, tmp_path):
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"this is definitely not audio" * 100)
    response = client.post("/api/library/upload", files={"files": ("broken.wav", broken.read_bytes())})
    result = response.json()["results"][0]
    assert result["status"] == "rejected"
    assert result["reason"]


def test_batch_upload_mixed_results(client, encoded_sources):
    response = client.post(
        "/api/library/upload",
        files=[
            ("files", ("good.wav", encoded_sources["wav"].read_bytes())),
            ("files", ("bad.mp3", encoded_sources["mp3"].read_bytes())),
        ],
    )
    results = response.json()["results"]
    assert [r["status"] for r in results] == ["imported", "rejected"]
    assert len(client.get("/api/library").json()["items"]) == 1


def test_duplicate_upload_is_deduplicated(client, encoded_sources):
    first = upload(client, encoded_sources["wav"])
    response = client.post(
        "/api/library/upload",
        files={"files": ("renamed-copy.wav", encoded_sources["wav"].read_bytes())},
    )
    second = response.json()["results"][0]
    assert second["status"] == "duplicate"
    assert second["audio_id"] == first["audio_id"]
    items = client.get("/api/library").json()["items"]
    assert len(items) == 1, "同内容不同文件名只应入库一次"


def test_filename_with_path_traversal_is_sanitized(client, encoded_sources):
    response = client.post(
        "/api/library/upload",
        files={"files": ("../../evil/../base.wav", encoded_sources["wav"].read_bytes())},
    )
    result = response.json()["results"][0]
    assert result["status"] == "imported"
    assert result["filename"] == "base.wav"
    assert not (app_module.PROJECT_ROOT.parent / "evil.wav").exists()


# ---------------------------------------------------------------------------
# 删除
# ---------------------------------------------------------------------------


def test_delete_removes_library_file_and_cache(client, fast_config, long_source):
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    session_id = start_session(client, [audio_id])["session_id"]
    assert cache_wavs(audio_id)
    client.post("/api/end", json={"session_id": session_id})

    response = client.delete(f"/api/library/{audio_id}")
    assert response.status_code == 200, response.text
    assert response.json()["deleted"] is True
    assert not (app_module.LIBRARY_DIR / f"{audio_id}.flac").exists()
    assert not (app_module.CACHE_DIR / audio_id).exists()
    assert client.get("/api/library").json()["items"] == []
    assert client.delete(f"/api/library/{audio_id}").status_code == 404


def test_delete_is_refused_while_session_running(client, fast_config, long_source):
    item = upload(client, long_source)
    start_session(client, [item["audio_id"]])
    response = client.delete(f"/api/library/{item['audio_id']}")
    assert response.status_code == 409
    assert "进行中的测试" in response.json()["detail"]


def test_delete_allowed_once_all_questions_are_answered(client, fast_config, long_source):
    """回归：答完全部题目后不该再被当成"进行中的测试"而拒绝删除。"""
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    session_id = start_session(client, [audio_id])["session_id"]
    for question in app_module.SESSIONS[session_id].questions:
        assert _answer(client, session_id, question, _correct_choice(question)).status_code == 200

    response = client.delete(f"/api/library/{audio_id}")
    assert response.status_code == 200, response.text
    assert not (app_module.LIBRARY_DIR / f"{audio_id}.flac").exists()
    assert not (app_module.CACHE_DIR / audio_id).exists()


def test_end_then_delete_works(client, fast_config, long_source):
    """回归：结束测试后回到主页面应当可以删除音频。"""
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    session_id = start_session(client, [audio_id])["session_id"]
    client.post("/api/end", json={"session_id": session_id})
    assert client.delete(f"/api/library/{audio_id}").status_code == 200


# ---------------------------------------------------------------------------
# 出题：参数校验 + 防泄漏
# ---------------------------------------------------------------------------

FORBIDDEN_SUBSTRINGS = ("mp3", "flac", "kbps", "ll.wav", "mp3.wav", "seg", "placebo", "lossless", "bitrate")


def test_start_validates_parameters(client, fast_config, long_source):
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    assert client.post("/api/start", json={"audio_ids": [], "mode": "short"}).status_code == 400
    too_many = [f"{index:016x}" for index in range(4)]
    assert client.post(
        "/api/start", json={"audio_ids": too_many, "mode": "short"}
    ).status_code == 400
    assert client.post(
        "/api/start", json={"audio_ids": [audio_id], "mode": "quick"}
    ).status_code == 400
    assert client.post(
        "/api/start", json={"audio_ids": ["0011223344556677"], "mode": "short"}
    ).status_code == 404


def test_start_deduplicates_repeated_audio_ids(client, fast_config, long_source):
    item = upload(client, long_source)
    session = start_session(client, [item["audio_id"]] * 3)
    assert session["total"] == 2  # 快速模式 1 真实 + 1 安慰剂


def test_question_does_not_leak_bitrate_or_codec(client, fast_config, long_source):
    item = upload(client, long_source)
    session = start_session(client, [item["audio_id"]])
    session_id = session["session_id"]
    questions = app_module.SESSIONS[session_id].questions

    for index, question in enumerate(questions):
        response = client.get("/api/question", params={"session_id": session_id, "index": index})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert set(payload) == {"index", "total", "question_id", "url_a", "url_b"}, payload
        body = response.text.lower()
        for needle in FORBIDDEN_SUBSTRINGS:
            assert needle not in body, f"响应泄漏了 {needle}: {response.text}"
        if question.bitrate is not None:
            assert f"{question.bitrate}k" not in body
            assert str(question.bitrate) not in payload["url_a"].split("/")[-1]
        assert payload["url_a"] != payload["url_b"] or question.bitrate is None

        audio = [client.get(payload[key]) for key in ("url_a", "url_b")]
        assert [r.status_code for r in audio] == [200, 200]
        assert all(r.headers["content-type"] == "audio/wav" for r in audio)
        assert "content-disposition" not in {k.lower() for k in audio[0].headers}
        assert len(audio[0].content) == len(audio[1].content), "两路体积必须一致，避免体积泄漏"


def test_question_index_out_of_range_and_unknown_session(client, fast_config, long_source):
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    assert client.get("/api/question", params={"session_id": session_id, "index": 99}).status_code == 400
    assert client.get("/api/question", params={"session_id": session_id, "index": -1}).status_code == 400
    assert client.get(
        "/api/question", params={"session_id": "ffffffffffffffff", "index": 0}
    ).status_code == 404


def test_start_with_cover_art_source_builds_stimuli(client, fast_config, art_source, monkeypatch):
    """端到端回归（rc=234）：带内嵌封面的源 + 起点为 0 的片段，以前 /api/start 会 500。

    这里把片段选择固定到 0 秒：起点 > 0 时 ffmpeg 本来就会漏掉封面帧，测不出这个 bug。
    """

    def fake_analysis(mono):
        return {
            "selected": [{"index": 0, "start": 0.0, "end": 15.0, "score": 1.0}],
            "analysis": {
                "analysis_sr": app_module.ANALYSIS_SR,
                "samples": int(mono.size),
                "duration": float(mono.size / app_module.ANALYSIS_SR),
                "silence_threshold_db": -60.0,
                "weights": dict(app_module.SEG_WEIGHTS),
                "window_count": 1,
            },
            "windows": [],
        }

    monkeypatch.setattr(app_module, "_analyse_windows", fake_analysis)
    item = upload(client, art_source)
    start_session(client, [item["audio_id"]])  # 修复前这里会以 500 + rc=234 失败

    segment = app_module.CACHE_DIR / item["audio_id"] / "seg_0.flac"
    assert stream_types(segment) == ["audio"]
    assert cache_wavs(item["audio_id"]), "刺激应当已生成"


def test_placebo_question_serves_identical_audio(client, fast_config, long_source):
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    placebo_index = next(
        i for i, q in enumerate(app_module.SESSIONS[session_id].questions) if q.bitrate is None
    )
    payload = client.get(
        "/api/question", params={"session_id": session_id, "index": placebo_index}
    ).json()
    left = client.get(payload["url_a"]).content
    right = client.get(payload["url_b"]).content
    assert left == right, "安慰剂题的 A/B 必须是同一份音频"


# ---------------------------------------------------------------------------
# 作答与统计
# ---------------------------------------------------------------------------


def _answer(client, session_id: str, question, choice: str, reaction_ms: int | None = 1200):
    return client.post(
        "/api/answer",
        json={
            "session_id": session_id,
            "question_id": question.question_id,
            "choice": choice,
            "reaction_ms": reaction_ms,
        },
    )


def _correct_choice(question) -> str:
    if question.bitrate is None:
        return "tie"
    return "a" if question.lossless_is_a else "b"


def _wrong_choice(question) -> str:
    if question.bitrate is None:
        return "a"  # 在安慰剂上声称听出差异
    return "b" if question.lossless_is_a else "a"


def test_answer_validation_and_idempotency(client, fast_config, long_source):
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    question = app_module.SESSIONS[session_id].questions[0]

    assert _answer(client, session_id, question, "c").status_code == 400
    assert client.post(
        "/api/answer",
        json={"session_id": session_id, "question_id": "nope", "choice": "a"},
    ).status_code == 400
    assert client.post(
        "/api/answer",
        json={"session_id": "ffffffffffffffff", "question_id": question.question_id, "choice": "a"},
    ).status_code == 404
    assert client.post(
        "/api/answer",
        json={
            "session_id": session_id,
            "question_id": question.question_id,
            "choice": "a",
            "reaction_ms": -5,
        },
    ).status_code == 400

    assert _answer(client, session_id, question, "a").status_code == 200
    assert _answer(client, session_id, question, "a").status_code == 200  # 同选项幂等重放
    assert _answer(client, session_id, question, "tie").status_code == 200  # 首答即最终：忽略修改
    answers = app_module.SESSIONS[session_id].answers
    assert answers[question.question_id]["choice"] == "a"
    assert len(answers) == 1


def test_result_counts_correct_answers(client, monkeypatch, long_source):
    """5 道 96 kbps 全答对 → 该档显著高于瞎猜基线；安慰剂全答「听不出」。"""
    monkeypatch.setattr(app_module, "SHORT_TEST_BITRATES", [96])
    monkeypatch.setattr(app_module, "SHORT_TEST_TOTAL", 6)  # 5 真实 + 1 安慰剂
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    questions = app_module.SESSIONS[session_id].questions
    assert len(questions) == 6

    for question in questions:
        assert _answer(client, session_id, question, _correct_choice(question)).status_code == 200

    result = client.get("/api/result", params={"session_id": session_id}).json()
    assert result["total"] == 6
    assert result["real_total"] == 5 and result["real_correct"] == 5
    assert result["placebo_total"] == 1
    expected = {
        "n": 5,
        "correct": 5,
        "accuracy": 1.0,
        "ci_low": pytest.approx(0.5655, abs=1e-3),
        "ci_high": 1.0,
        "p_value": pytest.approx(0.0041, abs=1e-4),
        "significant": True,
        "status": "significant",
        "note": "显著高于瞎猜基线",
    }
    assert result["by_bitrate"] == [{"bitrate": 96, **expected, "mean_reaction_ms": 1200}]
    assert {k: v for k, v in result["by_bracket"][0].items() if k not in ("label", "range")} == expected
    assert result["by_bracket"][0]["label"] == "≤96 kbps"
    assert result["overall"] == expected
    assert result["chance_level"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["alpha"] == 0.05
    assert result["power_required_n"] == 7
    assert result["power"] == {
        "required_n": 7,
        "max_per_bitrate": 5,
        "target_power": 0.8,
        "assumed_rate": 0.8,
    }
    assert result["placebo"] == {"n": 1, "tie": 1, "tie_rate": 1.0, "warning": False}
    assert "显著高于瞎猜基线" in result["conclusion"]
    assert "瞎猜倾向" not in result["conclusion"]
    assert "每档 7 题" in result["conclusion"]


def test_result_flags_guessing(client, monkeypatch, long_source):
    """全部答错 + 安慰剂上声称听出差异 → 结论应指出无法区分与瞎猜倾向。"""
    monkeypatch.setattr(app_module, "SHORT_TEST_BITRATES", [96])
    monkeypatch.setattr(app_module, "SHORT_TEST_TOTAL", 6)
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    questions = app_module.SESSIONS[session_id].questions

    for question in questions:
        assert _answer(client, session_id, question, _wrong_choice(question)).status_code == 200

    result = client.get("/api/result", params={"session_id": session_id}).json()
    assert result["real_correct"] == 0
    assert result["overall"]["significant"] is False
    assert result["overall"]["status"] == "not_significant"
    assert result["placebo"]["tie_rate"] == 0.0
    assert result["placebo"]["warning"] is True
    assert "与瞎猜基线无显著差异" in result["conclusion"]
    assert "瞎猜倾向" in result["conclusion"]


def test_result_statistics_payload(client, monkeypatch, long_source):
    """未作答与单题两种情况下的统计字段：p 值缺失/样本不足都要如实反映。"""
    monkeypatch.setattr(app_module, "SHORT_TEST_BITRATES", [96])
    monkeypatch.setattr(app_module, "SHORT_TEST_TOTAL", 6)
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]

    empty = client.get("/api/result", params={"session_id": session_id}).json()
    assert empty["overall"]["n"] == 0 and empty["overall"]["p_value"] is None
    assert empty["overall"]["status"] == "insufficient"
    assert empty["by_bitrate"] == [] and empty["placebo"]["n"] == 0
    assert empty["power"]["max_per_bitrate"] == 0
    assert empty["conclusion"]

    # 只答一道（真实题）→ 逐档 n=1，任何结果都不可能显著（p = 1/3）
    real = next(q for q in app_module.SESSIONS[session_id].questions if q.bitrate is not None)
    assert _answer(client, session_id, real, _correct_choice(real)).status_code == 200
    one = client.get("/api/result", params={"session_id": session_id}).json()
    assert one["by_bitrate"][0]["n"] == 1
    assert one["by_bitrate"][0]["p_value"] == pytest.approx(1 / 3, abs=1e-4)
    assert one["by_bitrate"][0]["significant"] is False
    assert one["by_bitrate"][0]["status"] == "insufficient"
    assert one["power"]["max_per_bitrate"] == 1
    assert "样本不足" in one["by_bitrate"][0]["note"]
    assert "样本不足" in one["conclusion"]


def test_static_assets_are_revalidated(client):
    """HTML/JS/CSS 必须要求浏览器重验证。

    回归：浏览器用缓存里的旧 app.js 调新接口时，旧脚本读不到新字段就抛异常，
    "查看答案"只把按钮藏了、什么都不显示（很难排查）。带 ?v= 的 URL + no-cache 双重兜底。
    """
    index = client.get("/")
    assert index.status_code == 200
    assert "no-cache" in index.headers.get("cache-control", "")

    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert "no-cache" in script.headers.get("cache-control", "")

    # 页面里引用静态资源时带版本号，改前端后只需 bump 版本即可绕开任何中间缓存
    assert "/static/app.js?v=" in index.text
    assert "/static/style.css?v=" in index.text

    # 前端不再依赖旧的顶层 correct_side / correctSide 字段（揭晓改成 reveal 之前的契约）
    assert "data.correct_side" not in script.text
    assert "correctSide" not in script.text
    assert "reveal.sides" in script.text


def test_result_handles_unanswered_questions(client, fast_config, long_source):
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    result = client.get("/api/result", params={"session_id": session_id}).json()
    assert result["answered"] == 0
    assert result["by_bitrate"] == []
    assert result["placebo"]["n"] == 0
    assert result["conclusion"]


def test_answer_done_flag(client, fast_config, long_source):
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    questions = app_module.SESSIONS[session_id].questions
    last = len(questions) - 1
    for index, question in enumerate(questions):
        response = _answer(client, session_id, question, _correct_choice(question))
        payload = response.json()
        assert payload["index"] == index
        assert payload["next_index"] == index + 1
        assert payload["done"] is (index == last)


def test_answer_reveals_both_sides_only_after_answering(client, fast_config, long_source):
    """作答后才揭晓本题两侧身份（无损 / MP3 码率）；未作答的任何响应都不含码率。"""
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    questions = app_module.SESSIONS[session_id].questions
    assert any(q.bitrate is not None for q in questions)
    assert any(q.bitrate is None for q in questions)

    for index, question in enumerate(questions):
        # 作答前：题目接口只有 5 个键，且不含任何码率/编码器字样
        before = client.get("/api/question", params={"session_id": session_id, "index": index})
        assert set(before.json()) == {"index", "total", "question_id", "url_a", "url_b"}
        for needle in FORBIDDEN_SUBSTRINGS:
            assert needle not in before.text.lower()

        payload = _answer(client, session_id, question, "tie").json()
        assert set(payload) == {"ok", "index", "next_index", "done", "accepted", "reveal"}
        assert payload["accepted"] is True
        reveal = payload["reveal"]
        assert set(reveal) == {"placebo", "correct_side", "same_segment", "sides"}

        if question.bitrate is None:
            assert reveal["placebo"] is True
            assert reveal["correct_side"] is None
            assert reveal["sides"]["a"] == reveal["sides"]["b"] == {"kind": "lossless", "bitrate": None}
        else:
            assert reveal["placebo"] is False
            assert reveal["correct_side"] == ("a" if question.lossless_is_a else "b")
            sides = reveal["sides"]
            kinds = {side: info["kind"] for side, info in sides.items()}
            assert sorted(kinds.values()) == ["lossless", "mp3"]
            coded = "b" if question.lossless_is_a else "a"
            assert sides[coded] == {"kind": "mp3", "bitrate": question.bitrate}


def test_answer_is_final_and_replay_is_idempotent(client, fast_config, long_source):
    """首答即最终：改了选项只回 accepted=false 且不泄漏答案；同选项重放仍返回同样的揭晓。"""
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    question = next(q for q in app_module.SESSIONS[session_id].questions if q.bitrate is not None)

    first = _answer(client, session_id, question, "tie").json()
    assert first["accepted"] is True and first["reveal"] is not None

    replay = _answer(client, session_id, question, "tie").json()
    assert replay["accepted"] is True
    assert replay["reveal"] == first["reveal"], "同选项重放应当幂等"

    changed = _answer(client, session_id, question, _correct_choice(question)).json()
    assert changed["accepted"] is False
    assert changed["reveal"] is None, "想先看答案再改分：不允许"
    assert app_module.SESSIONS[session_id].answers[question.question_id]["choice"] == "tie"

    result = client.get("/api/result", params={"session_id": session_id}).json()
    assert result["real_correct"] == 0, "被忽略的修改不得计分"


# ---------------------------------------------------------------------------
# 清理与路径安全
# ---------------------------------------------------------------------------


def test_end_removes_session_dir_but_keeps_cache(client, fast_config, long_source):
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    session_id = start_session(client, [audio_id])["session_id"]
    session_dir = app_module.SESSION_DIR / session_id
    assert session_dir.is_dir() and any(session_dir.iterdir())

    # 硬链接语义：session 里的文件与缓存文件必须是同一 inode
    session_file = next(session_dir.iterdir())
    cache_inodes = {os.stat(path).st_ino for path in cache_wavs(audio_id)}
    assert os.stat(session_file).st_ino in cache_inodes

    response = client.post("/api/end", json={"session_id": session_id})
    assert response.status_code == 200
    assert not session_dir.exists()
    assert session_id not in app_module.SESSIONS
    assert cache_wavs(audio_id), "/api/end 不应删除刺激缓存"
    assert client.get("/api/result", params={"session_id": session_id}).status_code == 404


def test_bootstrap_clears_session_and_tmp_but_keeps_library_and_cache(client, fast_config, long_source):
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    start_session(client, [audio_id])
    stray_session = app_module.SESSION_DIR / "stray"
    stray_session.mkdir()
    (stray_session / "leftover.wav").write_bytes(b"junk")
    stray_tmp = app_module.TMP_DIR / "leftover.tmp"
    stray_tmp.write_bytes(b"junk")

    with TestClient(app_module.app):
        assert not stray_session.exists(), "启动应清空 data/session/"
        assert not stray_tmp.exists(), "启动应清空 data/tmp/"
        assert (app_module.LIBRARY_DIR / f"{audio_id}.flac").exists(), "不应删除库文件"
        assert cache_wavs(audio_id), "不应删除刺激缓存"
        assert client.get("/api/library").json()["items"], "不应丢失库索引"


def test_expired_session_is_cleaned_on_next_start(client, fast_config, long_source):
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    stale_id = start_session(client, [audio_id])["session_id"]
    app_module.SESSIONS[stale_id].created_at -= app_module.SESSION_TTL_SECONDS + 1

    start_session(client, [audio_id])
    assert stale_id not in app_module.SESSIONS
    assert not (app_module.SESSION_DIR / stale_id).exists()


def test_audio_endpoint_rejects_unknown_and_foreign_tokens(client, fast_config, long_source):
    item = upload(client, long_source)
    session_id = start_session(client, [item["audio_id"]])["session_id"]
    other_id = start_session(client, [item["audio_id"]])["session_id"]
    payload = client.get("/api/question", params={"session_id": session_id, "index": 0}).json()
    good_token = payload["url_a"].rsplit("/", 1)[-1]

    assert client.get(f"/api/audio/{session_id}/deadbeefdeadbeef").status_code == 404
    assert client.get(f"/api/audio/ffffffffffffffff/{good_token}").status_code == 404
    assert client.get(f"/api/audio/{other_id}/{good_token}").status_code == 404
    assert client.get("/api/audio/%s/..%%2F..%%2Fapp.py" % session_id).status_code == 404
    assert client.get(f"/api/audio/{session_id}/{good_token}").status_code == 200


def test_clear_cache_endpoint(client, fast_config, long_source):
    item = upload(client, long_source)
    audio_id = item["audio_id"]
    start_session(client, [audio_id])
    assert cache_wavs(audio_id)

    response = client.delete("/api/cache")
    assert response.status_code == 200
    assert response.json()["freed_bytes"] > 0
    assert not app_module.CACHE_DIR.joinpath(audio_id).exists()
    # 库文件不受影响，重新出题会重算片段
    assert (app_module.LIBRARY_DIR / f"{audio_id}.flac").exists()
    start_session(client, [audio_id])
    assert cache_wavs(audio_id)


def test_static_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
