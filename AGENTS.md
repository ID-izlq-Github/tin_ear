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
  data/
    library/              # 持久：48k FLAC 原音频
      index.json
      <audio_id>.flac
    cache/                # 持久：片段与刺激缓存
      <audio_id>/
        meta.json
        seg_0.flac
        seg_1.flac
        stimuli/
    session/              # 临时：当前测试刺激，测完即删
      <session_id>/
  requirements.txt
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

## 5. 存储规则

- `library/` 持久。启动不清理。
- `cache/` 持久。`meta.json` 中 `SEGMENT_ALGO_VERSION` 不匹配时重算片段。
- `session/` 临时。后端启动清空。`/api/start` 时清理超 `SESSION_TTL_SECONDS` 的旧目录。`/api/end` 删除当前 session 目录。
- `SESSIONS: dict[str, SessionState]` 保存活动会话。重启即丢。

## 6. 音频处理

### 导入到库

1. `ffprobe` 检查：必须无损编码（`pcm_*`、`flac`、`alac`、`wavpack`、`tta`），采样率 ≥ 44100 Hz，时长 ≥ 30 秒。
2. 转 48 kHz / 立体声 / FLAC：
   ```
   ffmpeg -y -i input -ar 48000 -ac 2 -c:a flac output.flac
   ```
3. SHA-256 去重。已存在则跳过。
4. 写 `data/library/<audio_id>.flac`，更新 `index.json`。

### 片段选择

1. 15 秒窗、5 秒步长滑动。
2. 每窗打分：`响度标准差 + 谱通量 + 瞬态密度 - 静音比例`（各分量 z-score 后相加）。
3. 取最高分窗口，起止写入 `cache/<audio_id>/meta.json`。
4. ffmpeg 截取，20 ms 淡入淡出，存 `seg_0.flac`。时长 ≥ 60 秒可再取 `seg_1.flac`。
5. 片段固定 15 秒，不动态调整。

### 生成刺激

- 无损版：片段 FLAC 解码为 48 kHz float32 WAV。
- 压缩版：
  ```
  ffmpeg -y -i seg.flac -c:a libmp3lame -b:a <bitrate>k temp.mp3
  ffmpeg -y -i temp.mp3 -ar 48000 -ac 2 -c:a pcm_f32le decoded.wav
  ```
- 用互相关对齐 MP3 解码版与 FLAC 版，裁到相同长度。
- 各自用 pyloudnorm 归一化到 `-23 LUFS`，True Peak ≤ `-1 dBTP`。
- FLAC 版与 MP3 版**独立归一化**，避免响度线索。
- 两份 WAV 写 `session/<session_id>/`。

### 题目生成

- 模式：`short` 或 `detailed`。
- 从选中音频池轮流取片段。
- 码率从低到高排序（简单 → 难）。
- 追加 1～2 道安慰剂题（A、B 均无损）。
- 每题随机 `lossless_is_a`。
- 题目仅存内存。

## 7. API

```
GET    /api/library
POST   /api/library/upload        form-data: files
DELETE /api/library/{audio_id}

POST /api/start                   body: { audio_ids, mode }
GET  /api/question?session_id=&index=
POST /api/answer                  body: { session_id, question_id, choice, reaction_ms }
GET  /api/result?session_id=
POST /api/end                     body: { session_id }

GET /                             → static/index.html
GET /static/{file}
GET /api/audio/{session_id}/{filename}
```

`/api/question` 不返回码率、编码器、哪个是无损。

## 8. 前端

单页三视图：

**库管理**：列表、导入、删除、checkbox 选素材（1～3）、模式选择、开始测试。

**测试**：题号/总数、进度条、A/B 播放按钮（循环、5 ms 淡入淡出）、三选项（A 更好 / B 更好 / 听不出）。不显示文件名、码率、编码器、时长。答后自动下一题。

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
sudo apt install ffmpeg
pip install fastapi uvicorn python-multipart numpy scipy pyloudnorm soundfile
```

测试注意：关 Windows 音效增强、Dolby、EQ、空间音频；系统音量固定；WSL 不出声，用 Windows 浏览器访问 localhost 播放。

## 13. 关键决策

- 无数据库：一次性玩，不需历史。
- 统一 48k FLAC：省空间、统一处理、无损参考。
- 保留原音频：库需要管理，删了无法重算。
- 固定 15 秒：盲测长度必须一致，否则统计被污染。
- 独立 LUFS 归一化：避免响度成为线索。
- 安慰剂题：检测瞎猜。
- 不用 librosa：依赖重，ffmpeg + numpy + scipy 足够。