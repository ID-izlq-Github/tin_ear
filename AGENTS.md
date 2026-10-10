# AGENTS.md — TinEar

## 1. 项目

TinEar — 本地音频盲测工具。用户管理本地音频库，选 1～3 段开始测试，程序自动截取 15 秒片段，生成 MP3 压缩版，按难度出题，用户判断无损 FLAC 与不同码率 MP3 哪个更好。

一句话简介：A local blind test for MP3 bitrate perception.

定位：个人玩一玩，无多用户、无持久历史。

## 2. 技术栈

- Python 3.12 + FastAPI + Uvicorn
- ffmpeg / ffprobe + numpy + scipy + pyloudnorm
- 原生 HTML/CSS/JS，无框架、无构建
- Web Audio API，AudioContext 固定 48000 Hz
- 无数据库。库信息用 JSON，会话状态用内存 dict
- WSL Ubuntu 跑后端，Windows 浏览器访问 `http://localhost:8000`

禁止：Node.js、JDK、Docker、Electron、SQLite、librosa、React、Vue、Vite、Webpack。

## 3. 目录结构

```
tinear/
  app.py                  # 后端全部逻辑，单文件
  static/
    index.html
    app.js
    style.css
  tests/                  # pytest：conftest.py / test_pipeline.py / test_api.py
  data/                   # 全部 gitignore，运行时自建
    library/              # 持久：48k FLAC 原音频
      index.json
      <audio_id>.flac
    cache/                # 持久：片段与刺激缓存
      <audio_id>/
        meta.json         # 候选窗口、分数、片段起止
        seg_0.flac
        seg_1.flac        # 仅当源时长 ≥ 60 秒
        stimuli/
          v1/             # v{STIMULI_ALGO_VERSION}，版本变化即重算
            index.json    # 每对刺激的 LUFS / True Peak / lag / 增益
            seg0_96k_ll.wav + seg0_96k_mp3.wav
            seg0_p0.wav   # 安慰剂（A/B 同一份文件）
    session/              # 临时：当前测试刺激的硬链接，测完即删
      <session_id>/
    tmp/                  # 临时：上传落盘与 ffmpeg 中间文件，启动清空
  .pip-cache/             # pip 下载缓存（项目内，gitignore）
  requirements.txt
  requirements-dev.txt
  requirements.lock.txt
  Makefile                # install / run / test / clean-cache
  README.md
```

## 4. 配置常量（app.py 顶部）

```python
SAMPLE_RATE = 48000
CHANNELS = 2
SEGMENT_SECONDS = 15
TARGET_LUFS = -23.0
TRUE_PEAK_DBTP = -1.0

SHORT_TEST_BITRATES = [48, 96, 160, 224, 320]
DETAILED_TEST_BITRATES = [32, 48, 64, 80, 96, 128, 160, 192, 256, 320]

SHORT_TEST_TOTAL = 6       # 5 真实 + 1 安慰剂
DETAILED_TEST_TOTAL = 12   # 10 真实 + 2 安慰剂

SEGMENT_ALGO_VERSION = 1
SESSION_TTL_SECONDS = 7200
```

### 4.1 实现新增常量

```python
SEGMENT_STEP_SECONDS = 5
SEGMENT_SAMPLES = 720000           # = 625 个 MPEG-1 Layer III 帧，与采样率精确对齐
SEGMENT_FADE_SECONDS = 0.02
STIMULI_ALGO_VERSION = 1           # 刺激缓存版本，与 SEGMENT_ALGO_VERSION 同步 bump
MAX_LUFS_DIFF_LU = 0.05            # 测试断言阈值（协议上限仍是 0.3）
MIN_SECONDS_FOR_SECOND_SEGMENT = 60
ANALYSIS_SR = 16000                # 片段打分用的分析解码（单声道）
ANALYSIS_MAX_SECONDS = 1200        # 只分析前 20 分钟，约束内存
STFT_N, STFT_HOP = 1024, 256
LOUDNESS_BLOCK_MS, LOUDNESS_HOP_MS = 400, 100
SILENCE_ABS_DB, SILENCE_REL_DB = -60.0, 40.0
ONSET_MEDIAN_MS, ONSET_FACTOR, ONSET_MIN_SEPARATION_S = 1000, 1.5, 0.05
SEG_WEIGHTS = {                    # 四分量 z-score 加权和（silence_ratio 取负）
    "loudness_std_db": 1.0, "spectral_flux": 1.0,
    "onset_density": 1.0, "silence_ratio": 1.0,
}
ALIGN_PROBE_SECONDS, ALIGN_REFINE_SECONDS = 6, 8
ALIGN_MAX_LAG, ALIGN_REFINE_SPAN = 8192, 64
STIMULUS_SUBTYPE = "FLOAT"         # 回退项 "PCM_24"
FFMPEG_CONCURRENCY = 2
MIN_SOURCE_SAMPLE_RATE, MIN_SOURCE_SECONDS = 44100, 30
SHUFFLE_QUESTIONS = False          # 默认保留「码率从低到高」
PLACEBO_SAME_SEGMENT = True        # 默认：安慰剂 = 同一片段的两份无损拷贝
RESULT_BRACKETS = [("≤96 kbps"), ("128–192 kbps"), ("≥224 kbps")]
```

注意：`SHORT_TEST_BITRATES` 并非 `DETAILED_TEST_BITRATES` 的子集（**224 kbps 只在 short 档**），
两条阶梯并集共 11 档，这也是刺激缓存的档位上限。

## 5. 存储规则

- `library/` 持久。启动不清理。
- `cache/` 持久。`meta.json` 中 `SEGMENT_ALGO_VERSION` 不匹配时重算片段；`stimuli/v{STIMULI_ALGO_VERSION}/`
  作为刺激缓存，版本不匹配即整目录重算。
- `session/` 与 `tmp/` 临时。后端启动清空（`data/library/`、`data/cache/` 保留）。
  `/api/start` 时清理超 `SESSION_TTL_SECONDS` 的旧目录与无主目录。`/api/end` 删除当前 session 目录。
- `session/` 内是 `cache/` 刺激文件的**硬链接**（跨文件系统时回退为复制），因此
  `/api/end` 只删链接，刺激缓存仍在，重复测试几乎瞬时开始。
- `tempfile.tempdir` 指向 `data/tmp/`：不写系统 temp 或用户目录。
- `SESSIONS: dict[str, SessionState]` 保存活动会话。重启即丢。

## 6. 音频处理

### 导入到库

1. `ffprobe` 检查：必须无损编码（`pcm_*`、`flac`、`alac`、`wavpack`、`tta`），采样率 ≥ 44100 Hz，时长 ≥ 30 秒。
2. 转 48 kHz / 立体声 / FLAC：
   ```
   ffmpeg -y -i input -ar 48000 -ac 2 -c:a flac output.flac
   ```
3. 对**转换后的 FLAC** 取 SHA-256 去重（而非源文件）：同一段音频的 WAV 与 AIFF 会被识别为同一内容。
   `audio_id = sha256[:16]`。
4. 写 `data/library/<audio_id>.flac`，更新 `index.json`（原子替换：写 tmp 再 rename）。条目字段：
   `audio_id / sha256 / display_name / duration / sample_rate / channels / size_bytes /
   source_codec / source_sample_rate / source_channels / added_at`
   —— "保留原音频"落在这些源信息上：库内母版是规范化的 48k FLAC，源文件本身不保留。
5. 上传先流式落盘 `data/tmp/`，只取 basename（防路径穿越），单文件 > 2 GB 直接拒绝。

### 片段选择

1. 分析解码一次：`ffmpeg -i <flac> -t 1200 -map 0:a:0 -ac 1 -ar 16000 -f f32le -`（16 kHz 单声道，控制内存；
   只分析前 20 分钟）。
2. 15 秒窗、5 秒步长滑动，候选窗数 `n = 1 + floor((duration-15)/5)`。
3. 每个窗口聚合四个分量：`loudness_std_db`（400 ms/100 ms 块 RMS 的 dB 标准差）、`spectral_flux`
   （n_fft=1024、hop=256 的 log1p 幅度谱正增量均值）、`onset_density`（自适应阈值 + 局部极大 + 50 ms
   最小间隔的每秒瞬态数）、`silence_ratio`（低于 `max(-60 dBFS, 全局 95 百分位 - 40 dB)` 的块占比）。
4. 对**本文件所有候选窗**做 z-score（std=0 记 0），按 §4.1 权重加权求和（`silence_ratio` 取负），
   `np.argmax` 保证确定性（并列取最早）。
5. 取最高分窗为 `seg_0`；时长 ≥ 60 秒时，再取与 `seg_0` 不重叠（`|Δstart| ≥ 15` 秒）的最高分窗为 `seg_1`。
6. 切片：20 ms 淡入淡出，48 kHz 立体声 FLAC；起止与全部候选窗分数写入 `cache/<audio_id>/meta.json`。
7. 片段固定 15 秒，不动态调整。

> ⚠️ 实测坑（已修）：本机 ffmpeg 6.1.1 把 `-ss` 放在 `-i` **之后**（输出 seek）时，`start=40s` 会输出
> 整段数字静音；放在 `-i` 之前（输入 seek，默认 `-accurate_seek`）则逐采样精确。由
> `test_cut_segment_is_sample_accurate` 守护。

### 生成刺激

- 缓存在 `cache/<audio_id>/stimuli/v{STIMULI_ALGO_VERSION}/`，键 = `seg<i>_<bitrate>k`（安慰剂为 `seg<i>_p<seg>`）；
  命中即复用，`session/<session_id>/` 只放硬链接。**不再每轮重算**（原 §6 写 session 目录与 §3 的 `stimuli/` 自相矛盾）。
- 无损版：片段 FLAC 解码为 48 kHz float32 母版。
- 压缩版（CBR，且写文件以便 LAME/Xing gapless 头生效）：
  ```
  ffmpeg -y -i seg.flac -map_metadata -1 -id3v2_version 0 -write_id3v1 0 -write_xing 1 \
         -c:a libmp3lame -b:a <bitrate>k temp.mp3
  ffmpeg -v error -i temp.mp3 -f f32le -ac 2 -ar 48000 -      # 直接管道进 numpy
  ```
- 对齐：FFT 互相关粗估（前 6 秒，`|lag| ≤ 8192`）+ 逐点平均绝对差精修（±64，正负延迟均覆盖），
  然后统一裁剪/补零到**精确 720000 采样**。
- **响度与峰值（对原文的修正）**：
  1. 两路各自按 integrated loudness 独立增益 `g_i = TARGET_LUFS - L_i` —— 响度差恒为 0，不留响度线索；
  2. 再对这一对施加**同一个**线性衰减 `delta = min(0, TRUE_PEAK_DBTP - max(TP_i + g_i))`，
     使两路 True Peak 都 ≤ `-1 dBTP`。
  **不使用限幅器**：限幅是非线性处理，在 32–64 kbps 会引入与 MP3 编码无关的失真，等于给低码率题加额外线索。
  True Peak 用 4× 多相过采样测量。
- 自检（不满足即报错）：长度 = 720000、`|ΔLUFS| ≤ MAX_LUFS_DIFF_LU`、峰值 ≤ -1 dBTP + 0.05、对齐残差 ≤ 1 采样。
- 输出 `soundfile` IEEE float32 WAV；两路长度相同 → 字节数相同，避免体积泄漏。

### 题目生成

- 模式：`short` 或 `detailed`；真实题数 = `SHORT_TEST_TOTAL-1` / `DETAILED_TEST_TOTAL-2`，安慰剂 1 / 2 道。
- 码率按 `ladder[i % len(ladder)]` 循环取（升序 → 简单到难）：把题量常量调大即自动变成多次重复，
  无需改 API（这是弥补"逐档单题"局限的入口）。
- 从选中音频池轮流取（round-robin），每首第 k 次出场在 `seg_0/seg_1` 间交替；`(segment, bitrate)` 优先取未用过的组合。
- 安慰剂题追加在末尾；默认 A、B 是**同一片段的两份无损拷贝**（内容完全相同），否则"听不出"不是唯一正确答案。
- 每题随机 `lossless_is_a`；题目仅存内存。
- `/api/start` 一次性准备好全部题目的刺激并硬链接进 session 目录 —— 会话自包含，之后删库不影响进行中的测试。

## 7. API

```
GET    /api/library                       → { items, cache_bytes }
POST   /api/library/upload        form-data: files   → { results: [{filename,status,audio_id,reason}] }
DELETE /api/library/{audio_id}            → 200 { deleted, freed_bytes } / 404 / 409（被**尚未完成**的测试引用）
DELETE /api/cache                         → 清空片段与刺激缓存 { freed_bytes }

POST /api/start                   body: { audio_ids, mode } → { session_id, total, mode }
GET  /api/question?session_id=&index=       → 仅 { index, total, question_id, url_a, url_b }
POST /api/answer                  body: { session_id, question_id, choice, reaction_ms }
GET  /api/result?session_id=
POST /api/end                     body: { session_id }

GET /                             → static/index.html
GET /static/{file}
GET /api/audio/{session_id}/{token}         → FileResponse(WAV, 支持 Range)
```

`/api/question` 不返回码率、编码器、哪个是无损、是否安慰剂。

**不透明 token**：`/api/audio` 只接受 `secrets.token_hex(8)` 生成的随机 token，经内存 manifest 映射到
session 目录内的硬链接；磁盘文件名（`_ll` / `_mp3` / `seg0_96k`）绝不出现在 HTTP 层 —— 否则验收 #8 不成立，
码率还能从文件名直接推断。两路 `Content-Length` 恒等（同为 720000 采样 float32）。

其他约定：

- `/api/answer` 的 `choice ∈ {a, b, tie}`，`reaction_ms` 从**题目渲染时刻**起算；同题重复提交幂等覆盖。
  返回 `{ ok, index, next_index, done, correct_side }`：`correct_side ∈ {"a","b",null}`（`null` = 安慰剂题），
  **仅作答后**才返回，而且整条链路始终不返回码率（验收 #8 成立）。
- 删除库音频只在存在**尚未完成**（`len(answers) < len(questions)`）且引用它的会话时返回 409；
  答完全部题目或 `/api/end` 之后即可删除。
- 判定：真实题选中无损侧为正确，选 `tie` 记为"未听出"（不正确）；安慰剂题选 `tie` 才是正确。
- 结果给逐档统计 + `≤96 / 128–192 / ≥224 kbps` 分档聚合 + Wilson 95% 区间；逐档只有单题时前端标注"样本不足"。
- `/api/start` 会校验：`mode ∈ {short, detailed}`、`audio_ids` 去重后 1～3 个且都存在；参数不合法 400，音频不存在 404。

## 8. 前端

单页三视图：

**库管理**：列表、导入、删除、checkbox 选素材（1～3）、模式选择、开始测试。

**测试**：题号/总数、进度条、A/B 播放按钮（循环、5 ms 淡入淡出）、三选项（A 更好 / B 更好 / 听不出）。不显示文件名、码率、编码器、时长。

**作答后停留在本题**（不自动跳题）：高亮已选项、其余变暗，显示「已记录：X 更好」，并提供
「查看答案」（**可选**手动揭晓，默认不自动揭晓以免对后续题产生反馈偏置）与「下一题 →」（末题为「查看结果 →」）。
音频未载入完不允许作答，保证 `reaction_ms` 有效。
键盘：`1/2/3` 作答、`A/B/空格` 切换试听、回车或空格进入下一题。
离开测试页、点「再测一次」、或关闭标签页（`sendBeacon`）时都会调用 `/api/end`，不会留下"进行中"的引用。

**结果**：码率 × 正确率表、安慰剂“听不出”比例、文字结论、再测一次 / 返回库。

深色背景，大按钮，移动端可用，不做花哨动效。

## 9. 验收标准

1. `pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000` 直接跑。
2. Windows 浏览器访问 `http://localhost:8000` 走通导入、测试、结果。
3. FLAC / WAV / AIFF 可导入，导入后为 48 kHz FLAC。
4. MP3 / AAC / OGG / Opus 被拒绝，前端显示明确错误。
5. 重复导入同 hash 被识别，不重复入库。
6. 删除库音频时同时删除 `cache/<audio_id>/`。
7. 每题 A/B 响度差 ≤ 0.3 LU。
8. 前端请求看不到码率 / 编码器。
9. 结果页正确率、安慰剂统计正确。
10. 后端重启清空 `data/session/`，保留 `data/library/`。
11. `/api/end` 后 session 目录被删。
12. 改 `SEGMENT_ALGO_VERSION` 后 cache 失效重算。

自动化覆盖对照（`make test`，49 项）：#1 见 README 的安装/运行两行命令；#3/#4/#5 由 `test_api.py`
的导入与拒绝用例覆盖；#6/#10/#11 由清理用例覆盖（含硬链接 inode 断言，以及"答完/结束后才能删除"的回归用例）；
#7 的实测值为 `ΔLUFS = 0.0000`（远严于 0.3，`test_prepare_pair_loudness_and_peak` 断言 ≤ 0.05）；#8 由
`test_question_does_not_leak_bitrate_or_codec` 覆盖（键集合白名单 + 禁用子串 + 两路 `Content-Length` 相等）；
#9 由 `test_result_counts_correct_answers` / `test_result_flags_guessing` 用手算期望值覆盖；
#12 由 `test_segment_algo_version_invalidates_cache` / `test_stimuli_algo_version_moves_cache_dir` 覆盖。
**只有 #2（Windows 浏览器实操）与 #7 的主观听感需要人来点一遍。**

## 10. 非目标

- 不做 ABX。
- 不做自适应难度。
- 不做账号、登录、权限。
- 不做多用户并发、持久历史。
- 不做 AAC / Opus 对比，只做 MP3。
- 不做 192 kHz 差异测试。
- 不做动态片段长度。
- 不引入前端构建工具或框架。
- 不写数据库。

## 11. 实现顺序

1. `app.py` 骨架 + FastAPI + 静态文件。
2. `GET /api/library` + `index.json` 读写。
3. `POST /api/library/upload` + ffprobe + 转 FLAC + hash 去重。
4. `DELETE /api/library/{id}`。
5. 片段选择 + cache + 版本检查。
6. MP3 生成、解码、对齐、LUFS 归一化。
7. `/api/start` + 题目生成 + 内存 SESSIONS。
8. 前端库管理视图。
9. 前端测试视图 + Web Audio 播放。
10. `/api/answer` + `/api/result` + 结果视图。
11. `/api/end` + session 清理 + 启动清理。
12. 逐条跑验收标准。

## 12. 环境准备

```bash
# ffmpeg / ffprobe 用系统已有的 6.1.1，本项目不 apt install、不装任何全局包
make install     # = pip install -r requirements.txt，装进本项目专用 conda 环境 tin_ear
make run         # 前台启动（Ctrl+C 停止）；或 make run-bg 后台启动 + make stop 停止
make test        # 49 项自动化测试
```

- pip 缓存与临时文件都留在项目内（`.pip-cache/`、`data/tmp/`），不写系统 temp、`~/.cache`，
  也不动 `base` 或其他项目的 conda 环境。
- 依赖全是 wheel（`soundfile` 自带 libsndfile），不需要编译器或系统库。

测试注意：关 Windows 音效增强、Dolby、EQ、空间音频；系统音量固定；WSL 不出声，用 Windows 浏览器访问 localhost 播放。

## 13. 关键决策

- 无数据库：一次性玩，不需历史。
- 统一 48k FLAC：省空间、统一处理、无损参考；`index.json` 记源文件元信息以保留可追溯性。
- 固定 15 秒：盲测长度必须一致，否则统计被污染；48 kHz 下恰好 720000 采样 = 625 个 MP3 帧，
  与帧长精确对齐，裁剪后无残余。
- **配对独立响度增益 + 共同峰值衰减**（而不是"各自归一化 + 限幅压峰"）：响度差恒为 0，且全程零非线性，
  不给低码率题额外线索。
- 刺激持久缓存 + session 硬链接：重复测试不再重算（原设计每轮重算约 130 MB 且测完即丢）。
- 不透明 token：URL 与响应都不泄漏码率 / 编码器 / 是否安慰剂。
- 安慰剂题 = 同一片段的两份无损拷贝：专门检测"听到了不存在的差异"。
- 不用 librosa：依赖重，ffmpeg + numpy + scipy 足够。
- 单档 n=1 的统计局限：默认题量下逐档正确率只有 0 / 100%，因此结论以**分档** Wilson 区间为准；
  要真实统计功效就把 `SHORT_TEST_TOTAL` / `DETAILED_TEST_TOTAL` 调大（阶梯会循环重复）。

## 14. 测试

```bash
make test     # = TMPDIR=<项目>/data/tmp <tin_ear>/bin/python -m pytest -q
make run-bg   # 后台起服务（日志 data/server.log），make stop 停止
```

- `tests/conftest.py`：lavfi 合成音频（声部 + 粉噪 + 动态包络 + 瞬态，35 s / 65 s 两种）；
  `isolated_data` 夹具把 app 的全部数据目录重定向到本测试自己的临时目录；`fast_config` 把题量压到最小让 API 测试秒级完成。
- `tests/test_pipeline.py`：常量守护、对齐符号与真实 MP3 往返、等响度/峰值、**增益必须是纯线性**（防限幅器）、
  切片逐采样精确、片段选择确定性与不重叠、缓存版本失效、WAV 头规范（48 kHz / 2ch / 720000 / FLOAT）。
- `tests/test_api.py`：导入与去重、有损格式拒绝、删除与 409（含"答完全部题目后即可删除"的回归）、
  题目参数校验、**防泄漏**、`correct_side` 只在作答后返回且不含码率、安慰剂两路内容相同、作答幂等、
  结果统计与结论文案、启动清理、TTL 清理、路径穿越。
- 测试产物全部落在 `data/tmp/`（受 `TMPDIR` 约束），不写系统 temp 或用户目录。

## 15. 实测数据与实现记录（ffmpeg 6.1.1-3ubuntu5 / Python 3.12.15）

| 项 | 实测值 |
|---|---|
| 15 s @48 kHz | 720000 采样 = 恰好 625 个 MPEG-1 Layer III 帧 |
| MP3 编码到**文件**（`-write_xing 1` 生效） | 解码回来仍是 720000 采样，lag = 0 |
| MP3 编码到**管道**（无 LAME 头） | 解码得 721152 采样，lag = 1105（23.0 ms）；对齐器可兜住，结果仍 720000 / ΔLUFS = 0 |
| 每对刺激 | ΔLUFS = 0.0000 LU（32/64/128/320 kbps 均如此）；True Peak ≈ -10.3 dBTP；共同衰减 delta = 0 |
| 单个刺激文件 | 5760088 字节（5.76 MB + WAV 头），A/B 完全等长 |
| 缓存体积 | 每首最多 2 段 × (11 档 + 1 安慰剂) × 5.76 MB ≈ 130 MB（实测一次 detailed 后释放 130271006 字节） |
| `/api/start` 首次生成 10 对刺激 | 7.8 s；命中缓存后 < 0.2 s |
| 依赖 | 全部 wheel（soundfile 自带 libsndfile），无需编译器或系统库 |

实现期间修掉的三个真问题（都已加测试守护）：

1. **`-ss` 位置**：输出 seek（`-i` 之后）在 `start=40s` 会输出整段静音，必须用输入 seek。
2. **对齐精修符号**：`_estimate_lag` 精修阶段的平移方向与负延迟覆盖写反，需与 `_shifted` 保持一致
   （`aligned[k] = cand[k + lag]`）。
3. **响度公式**（对应 §6 修正）：若对两路施加"同一个增益"，两路响度差会保持原样而不是归零；
   正确做法是**各自**按 loudness 增益到目标，再施加**共同**的峰值余量衰减。
