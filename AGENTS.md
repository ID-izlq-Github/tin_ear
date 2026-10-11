# AGENTS.md — TinEar

> 文档分工：[README.md](README.md) 面向**使用者**（项目介绍、技术栈、安装、使用命令、排错），
> 本文件面向**维护者**（开发细节、设计取舍、测试、实测数据、项目现状与交接信息）。
> 许可证：MIT，见 [LICENSE](LICENSE)。

## 1. 项目

TinEar — 本地音频盲测工具。用户管理本地音频库，选 1～3 段开始测试，程序自动截取 15 秒片段，生成 MP3 压缩版，按码率阶梯出题，用户判断无损 FLAC 与不同码率 MP3 哪个更好。

一句话简介：A local blind test for MP3 bitrate perception.

定位：个人玩一玩，无多用户、无持久历史。

## 2. 技术栈

- Python 3.12（作者实测环境）+ FastAPI + Uvicorn；README 对使用者承诺 3.10+ —— 代码无 3.11+ 语法，
  `from __future__ import annotations` 兜住了 `X | None` 这类注解
- ffmpeg / ffprobe + numpy + scipy + pyloudnorm
- 原生 HTML/CSS/JS，无框架、无构建
- Web Audio API，AudioContext 固定 48000 Hz
- 无数据库。库信息用 JSON，会话状态用内存 dict
- WSL Ubuntu 跑后端，浏览器访问 `http://localhost:8000`（作者的验证环境，代码本身跨平台）
- 许可证：MIT

禁止：Node.js、JDK、Docker、Electron、SQLite、librosa、React、Vue、Vite、Webpack。

## 3. 目录结构

```
tinear/
  app.py                  # 后端全部逻辑，单文件（`python app.py` 就是最省事的启动入口）
  static/
    index.html
    app.js
    style.css
  tests/                  # pytest：conftest.py / test_pipeline.py / test_api.py
  data/                   # 全部 gitignore，运行时自建
    library/              # 持久：48k FLAC 原音频
      index.json
      <audio_id>.flac
    cache/                # 持久：片段与测试音频缓存
      <audio_id>/
        meta.json         # 候选窗口、分数、片段起止
        seg_0.flac
        seg_1.flac        # 仅当源时长 ≥ 60 秒
        stimuli/
          v2/             # v{STIMULI_ALGO_VERSION}，任何改变测试音频内容的常量都要 bump（当前 = 2）
            index.json    # 每个 (片段, 码率) 组合的 LUFS / True Peak / lag / 增益
            seg0_96k_ll.wav + seg0_96k_mp3.wav
            seg0_p0.wav   # 安慰剂（A/B 同一份文件）
    session/              # 临时：当前测试音频的硬链接，测完即删
      <session_id>/
    tmp/                  # 临时：上传落盘与 ffmpeg 中间文件，启动清空
  data/server.log         # make run-bg 的后台日志（make clean 会删）
  data/server.pid         # make run-bg / make stop 用的 pid 文件
  requirements.txt        # 运行期依赖（正常安装只需要这一份）
  requirements-dev.txt    # 测试依赖（pytest / httpx）
  requirements.lock.txt   # 作者机器实测通过的环境快照（pip list --format=freeze），仅供参考
  pytest.ini              # 测试产物保留策略（只留失败那一轮的临时目录）
  Makefile                # install / install-dev / run / run-bg / stop / test
                          # clean-cache / clean / size
  LICENSE                 # MIT
  README.md               # 面向使用者
  AGENTS.md               # 本文件，面向维护者
```

## 4. 配置常量（app.py 顶部）

```python
SAMPLE_RATE = 48000
CHANNELS = 2
SEGMENT_SECONDS = 15
TARGET_LUFS = -18.0        # 等响归一化目标（原 -23 = EBU R128 广播标准，日常试听偏轻）
TRUE_PEAK_DBTP = -1.0
TRUE_PEAK_SLACK_DB = 0.05    # 峰值上限的浮点容差（自检用）

SHORT_TEST_BITRATES = [48, 96, 160, 224, 320]
DETAILED_TEST_BITRATES = [32, 48, 64, 80, 96, 128, 160, 192, 256, 320]

SHORT_TEST_TOTAL = 6       # 5 真实 + 1 安慰剂（快速模式：每档 1 题）
DETAILED_TEST_TOTAL = 22   # 20 真实 + 2 安慰剂（详细模式：每档 2 题）

SEGMENT_ALGO_VERSION = 1
SESSION_TTL_SECONDS = 7200
```

### 4.1 实现新增常量

```python
SEGMENT_STEP_SECONDS = 5
SEGMENT_SAMPLES = 720000           # = 625 个 MPEG-1 Layer III 帧，与采样率精确对齐
SEGMENT_FADE_SECONDS = 0.02
STIMULI_ALGO_VERSION = 2           # 测试音频缓存版本：改 TARGET_LUFS / 峰值上限 / MP3 参数都要 bump
                                   # （v2：TARGET_LUFS -23 → -18）
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
STIMULI_CONCURRENCY_MAX = 8        # 单题生成实测 ~0.70 s，8 路之后基本饱和
STIMULI_CONCURRENCY = _resolve_concurrency()   # = min(可用核数, 上限)，TINEAR_CONCURRENCY 可覆盖
FFMPEG_CONCURRENCY = STIMULI_CONCURRENCY       # ffmpeg 子进程槽位随任务槽位走（原来是写死的 2，实际用不到）
MIN_SOURCE_SAMPLE_RATE, MIN_SOURCE_SECONDS = 44100, 30
SHUFFLE_ALL_QUESTIONS = True       # 真实题与安慰剂题统一打乱，位置不携带任何信息
PLACEBO_SAME_SEGMENT = True        # 默认：安慰剂 = 同一片段的两份无损拷贝
PREWARM_SHORT_COMPLEMENT = True    # 详细模式顺手生成快速模式独有的那 1～2 对（见 §6）
RESULT_BRACKETS = [(0, 96, "≤96 kbps"), (128, 192, "128–192 kbps"), (224, 10**9, "≥224 kbps")]
CHANCE_LEVEL = 1.0 / 3.0           # 三选一（A/B/听不出）的瞎猜基线，不是 1/2
ALPHA = 0.05                       # 单侧精确二项检验的显著性水平
POWER_TARGET, P1_FOR_POWER = 0.8, 0.8   # 算"每档需要多少题"用的功效与备择听出率
MIN_N_FOR_SIGNIFICANCE = 3         # 少于 3 题任何结果都无法判定（n=1 的最小 p 就是 1/3）
```

注意：`SHORT_TEST_BITRATES` 并非 `DETAILED_TEST_BITRATES` 的子集（**224 kbps 只在 short 档**），
两条阶梯并集共 11 档，这也是测试音频缓存的档位上限。

统计口径（实测 `POWER_REQUIRED_N = 7`）：单侧精确二项检验 `P(X ≥ k | p = 1/3)`，每档题数与临界答对数 ——
n=1 任何结果都不可能显著（最小 p = 1/3）、n=3 需 3/3（p=0.037）、n=4 需 4/4（p=0.012）、
n=5 需 4/5（p=0.045）、n=10 需 7/10（p=0.020）。详细模式 20 道真实题下分档 n = 10 / 6 / 4。

## 5. 存储规则

- `library/` 持久。启动不清理。
- `cache/` 持久。`meta.json` 中 `SEGMENT_ALGO_VERSION` 不匹配时重算片段；`stimuli/v{STIMULI_ALGO_VERSION}/`
  作为测试音频缓存，版本不匹配即整目录重算。
- 缓存键是 **`(audio_id, seg_index, bitrate)`**，没有任何淘汰/过期逻辑：生成过的组合永久保留，
  体积上限 = 两套阶梯并集 11 档 × 2 个片段 + 2 个安慰剂 = 46 个 WAV ≈ 267 MB/首（MiB 口径；只跑快速
  模式约 70 MB/首、详细模式实测 245.8 MB/首、加上它预热的快速补集约 257 MB/首）。**是否命中只取决于这一轮的题表**，
  而题表由「模式 + 素材集合」决定 —— `audio_ids` 在 `/api/start` 里按库内顺序排序，与前端勾选顺序无关，
  所以同素材重测必然 100% 复用（曾经因为直接用勾选顺序做 round-robin，换一下勾选先后就"部分重生成"）。
- `session/` 与 `tmp/` 临时。后端启动清空（`data/library/`、`data/cache/` 保留）。
  `/api/start` 时清理超 `SESSION_TTL_SECONDS` 的旧目录与无主目录。`/api/end` 删除当前 session 目录。
- `session/` 内是 `cache/` 测试音频的**硬链接**（跨文件系统时回退为复制），因此
  `/api/end` 只删链接，测试音频缓存仍在，重复测试几乎瞬时开始。`/api/start` 里 `os.link` 的
  目标 token 与进度 token 是两个独立变量（同名会互相覆盖，出过一个真 bug）。`/api/start` 的响应里报告本轮
  `stimuli: {generated, reused}`，前端把它显示出来，"到底复用没复用"不再靠猜。
- `tempfile.tempdir` 指向 `data/tmp/`（`_bootstrap` 里设置）：运行时草稿都在项目内，`make clean` 能收走。
- pip 下载缓存**不再重定向**到项目内：旧版本用 `PIP_CACHE_DIR` 顶到 `.pip-cache/`，已取消（见 §13、§16）。
- `SESSIONS: dict[str, SessionState]` 保存活动会话。重启即丢。

## 6. 音频处理

### 导入到库

1. `ffprobe` 检查：必须无损编码（`pcm_*`、`flac`、`alac`、`wavpack`、`tta`），采样率 ≥ 44100 Hz，时长 ≥ 30 秒。
2. 转 48 kHz / 立体声 / FLAC（**必须显式 `-map 0:a:0`**，见下面的封面流坑）：
   ```
   ffmpeg -y -i input -map 0:a:0 -map_metadata -1 -ar 48000 -ac 2 -c:a flac output.flac
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

> ⚠️ 实测坑（已修）：切片命令**必须带 `-map 0:a:0`**。带内嵌封面（`attached_pic` 视频流）的源在
> `start=0` 时会把封面帧一起写进 `seg_*.flac`（`start>0` 时封面帧落在窗口外，反而看不出来），
> 于是一路传到 MP3 编码步骤、muxer 想写 ID3v2 APIC 而我们的 `-idv2_version 0` 关掉了 ID3v2 →
> `rc=234 Attached pictures were requested, but the ID3v2 header is disabled`，`/api/start` 直接 500。
> 同一个 `-map 0:a:0` 也已加到导入转码与 MP3 编码两处，由
> `test_cut_segment_drops_attached_picture` / `test_import_strips_attached_picture` /
> `test_start_with_cover_art_source_builds_stimuli` 守护。

### 生成测试音频

- 缓存在 `cache/<audio_id>/stimuli/v{STIMULI_ALGO_VERSION}/`，键 = `seg<i>_<bitrate>k`（安慰剂为 `seg<i>_p<seg>`）；
  命中即复用，`session/<session_id>/` 只放硬链接，**不再每轮重算**。
- **并发生成**：`_prepare_session` 用 `asyncio.gather` + 任务信号量并发跑每道题，并发数 =
  `min(可用核数, STIMULI_CONCURRENCY_MAX=8)`（`_resolve_concurrency()`，`TINEAR_CONCURRENCY` 可覆盖；
  核数优先取 `os.sched_getaffinity(0)`，Windows 回退 `os.cpu_count()`）。`_FFMPEG_SEM` 与它同容量 ——
  原来的 `FFMPEG_CONCURRENCY = 2` 是写死的，而且因为逐题串行，实际永远只用得到 1。
- **详细模式顺手预热快速补集**（`PREWARM_SHORT_COMPLEMENT`）：`/api/start` 在 detailed 模式下会再跑一次
  `_build_questions("short", …)`，取它的请求集合与 detailed 的差集（通常只有 1 对，即 224 kbps 那一对）
  一起生成。这样"详细 → 快速"也是 0.01 s。用**请求差集**而不是"224 kbps × 所有片段"，阶梯以后改也自动对齐。
  这些额外条目只写缓存、不挂到题目上，计数单独报 `prewarmed`（否则用户会看到"22 题却新生成 23 段"）；
  进度条的总量把预热量算进去（4.4 s / 23 个）。反方向（快速 → 详细）缺 17 个组合，是阶梯覆盖问题，
  预热消不掉 —— 要消掉只能首次生成全集（§13 里权衡过，没做）。
- **并发下的两个共享资源**（都出过或差点出问题）：
  1. `stimuli/index.json` 是「读 → 改 → 写」，必须在锁里**重新读一次**再合并写，
     否则后写的会覆盖别人刚登记的条目（文件在、索引丢 → 下次白重算）；
  2. `_ensure_segments` 对同一个 audio_id 串行化，否则两个任务会同时判定"没缓存"、
     各自 `rmtree` + 重算，互相删掉对方产物。
  两把锁都通过 `_loop_scoped_lock(box, key)` 按**「事件循环 + key」**取：`asyncio.Lock` 一旦被争用
  就绑定当时的循环，而 pytest 每个用例都跑在新循环里，跨用例复用同一把模块级锁会抛
  `is bound to a different event loop`（由 `test_shared_locks_survive_multiple_event_loops` 守护）。
  任一道题失败时 `_prepare_session` 会 cancel 其余任务，`_run_process` 在被取消时 kill 掉 ffmpeg 子进程。
- 无损版：片段 FLAC 解码为 48 kHz float32 母版。
- 压缩版（CBR，且写文件以便 LAME/Xing gapless 头生效）：
  ```
  ffmpeg -y -i seg.flac -map 0:a:0 -map_metadata -1 -id3v2_version 0 -write_id3v1 0 -write_xing 1 \
         -c:a libmp3lame -b:a <bitrate>k temp.mp3
  ffmpeg -v error -i temp.mp3 -f f32le -ac 2 -ar 48000 -      # 直接管道进 numpy
  ```
- 对齐：FFT 互相关粗估（前 6 秒，`|lag| ≤ 8192`）+ 逐点平均绝对差精修（±64，正负延迟均覆盖），
  然后统一裁剪/补零到**精确 720000 采样**。
- **响度与峰值（对原文的修正）**：
  1. 两路各自按 integrated loudness 独立增益 `g_i = TARGET_LUFS - L_i` —— 响度差恒为 0，不留响度线索；
     `TARGET_LUFS = -18`（实测本机素材波峰因数 ≈ 12.7 dB → 真峰值 ≈ -5.3 dBTP，距 -1 dBTP 上限还有 4.3 dB；
     若调到 -14 以上，动态大的段落会开始触发峰值衰减，那时整题会比你设定的目标轻一点）。
  2. 再对这一对施加**同一个**线性衰减 `delta = min(0, TRUE_PEAK_DBTP - max(TP_i + g_i))`，
     使两路 True Peak 都 ≤ `-1 dBTP`。
  **不使用限幅器**：限幅是非线性处理，在 32–64 kbps 会引入与 MP3 编码无关的失真，等于给低码率题加额外线索。
  True Peak 用 4× 多相过采样测量。
- 自检（不满足即报错）：长度 = 720000、`|ΔLUFS| ≤ MAX_LUFS_DIFF_LU`、峰值 ≤ -1 dBTP + 0.05、对齐残差 ≤ 1 采样。
- 输出 `soundfile` IEEE float32 WAV；两路长度相同 → 字节数相同，避免体积泄漏。

### 题目生成

- 模式：`short` 或 `detailed`；真实题数 = `SHORT_TEST_TOTAL-1` / `DETAILED_TEST_TOTAL-2`，安慰剂 1 / 2 道。
- 码率按 `ladder[i % len(ladder)]` 循环取：把题量常量调大即自动变成多次重复（`detailed` 默认 20 道 → 每档 2 题），
  无需改 API。
- 出题顺序：先按上面的规则把真实题与安慰剂题**建好**（`(segment, bitrate)` 组合优先取未用过的，
  这样每轮用到的测试音频集合完全一致 → 缓存必然全部复用），然后**对整张题表 `rng.shuffle`**
  （`SHUFFLE_ALL_QUESTIONS = True`），最后统一重编号 `index` / `question_id`。
  真实题与安慰剂题一起打乱："第几题"既不能预判码率高低，也不能预判哪几题是无差异的，
  逼着人每题都真听（否则无脑按"听不出"会把真实题全部答错）。
- 从选中音频池轮流取（round-robin），每首第 k 次出场在 `seg_0/seg_1` 间交替。
- 安慰剂题的 A、B 默认是**同一片段的两份无损拷贝**（内容完全相同），否则"听不出"不是唯一正确答案。
- 每题随机 `lossless_is_a`；题目仅存内存。
- `/api/start` 一次性准备好全部题目的测试音频并硬链接进 session 目录 —— 会话自包含，之后删库不影响进行中的测试。

## 7. API

```
GET    /api/library                       → { items, cache_bytes, tmp_bytes }
POST   /api/library/upload        form-data: files   → { results: [{filename,status,audio_id,reason}] }
DELETE /api/library/{audio_id}            → 200 { deleted, freed_bytes } / 404 / 409（被**尚未完成**的测试引用）
DELETE /api/cache                         → 清空片段与测试音频缓存 data/cache { freed_bytes }

POST /api/start                   body: { audio_ids, mode, progress_token? }
                                  → { session_id, total, mode, target_lufs,
                                      stimuli: {generated, reused, prewarmed} }
GET  /api/progress?token=                   → { found, phase, done, total, finished, error }（只读内存）
GET  /api/question?session_id=&index=       → 仅 { index, total, question_id, url_a, url_b }
POST /api/answer                  body: { session_id, question_id, choice, reaction_ms }
                                  → { ok, index, next_index, done, accepted, reveal }
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

- `/api/answer` 的 `choice ∈ {a, b, tie}`，`reaction_ms` 从**题目渲染时刻**起算。**首答即最终**：
  同题重复提交时，选项相同 → 幂等回放同样的响应；选项不同 → `accepted: false`、`reveal: null`，不改分也不泄漏答案。
- 返回 `{ ok, index, next_index, done, accepted, reveal }`。`reveal` **仅在首答（或同选项重放）时**非 null：
  ```
  "reveal": {
    "placebo": false,                 # 安慰剂题：A/B 都是无损
    "correct_side": "a",              # {"a","b"}，安慰剂题为 null
    "same_segment": false,            # 安慰剂且 A/B 是同一份文件 → true
    "sides": { "a": {"kind": "lossless", "bitrate": null},
               "b": {"kind": "mp3",      "bitrate": 96} }
  }
  ```
  也就是说：**未作答的题目与音频接口始终零泄漏**（验收 #8），只有作答之后才揭晓本题两侧的具体身份。
- 删除库音频只在存在**尚未完成**（`len(answers) < len(questions)`）且引用它的会话时返回 409；
  答完全部题目或 `/api/end` 之后即可删除。
- 判定：真实题选中无损侧为正确，选 `tie` 记为"未听出"（不正确）；安慰剂题选 `tie` 才是正确。
- 结果统计：两层都用**单侧精确二项检验**（H0 = `CHANCE_LEVEL` = 1/3，`alpha` = 0.05），Wilson 区间降级为
  描述性；每行给 `p_value / significant / status / note`（`status ∈ {significant, not_significant, insufficient}`，
  是给前端做视觉分级的稳定枚举），顶层给 `overall`（全部真实题）、`by_bitrate`、`by_bracket`、
  `chance_level`、`alpha`、`power_required_n`（= 7）、`power: {required_n, max_per_bitrate, target_power, assumed_rate}`
  与 `placebo: {n, tie, tie_rate, warning}`。`note` 三态：`样本不足（n<3，无法判定）` / `显著高于瞎猜基线` /
  `与瞎猜基线无显著差异`。`n < MIN_N_FOR_SIGNIFICANCE` 时 `significant` 恒为 false
  （n=1 的最小 p 就是 1/3，结构上不可能显著）。`conclusion` 是同一套信息的纯文本版本（测试与复制用）。
- **静态资源强制重验证**：中间件给 `/` 与 `/static/*` 加 `Cache-Control: no-cache, must-revalidate`
  （命中 ETag 时是 304），并且 `index.html` 用 `/static/app.js?v=N` 引用资源。原因见 §15 的第 5 条：
  浏览器拿旧 app.js 调新接口会**静默失败**。改前端后记得 bump `?v=`。
- `/api/start` 会校验：`mode ∈ {short, detailed}`、`audio_ids` 去重后 1～3 个且都存在；参数不合法 400，音频不存在 404。
- `/api/start` 先把 `audio_ids` **按库内顺序排序**再出题：前端发来的是勾选顺序（`Set` 迭代序），
  排序保证「同一批素材」永远得到同一张题表 → 缓存 100% 复用。排序键是 `(库内下标, audio_id)`。
- **准备进度**：`progress_token` 由前端生成（`crypto.randomUUID`，有回退实现），随 `/api/start` 提交；
  后端把 `{phase, done, total, finished, error}` 写进内存 `PROGRESS`（`PROGRESS_TTL_SECONDS = 300`，
  在 `/api/start` 与 `/api/progress` 里顺手清过期项）。phase 依次是 `segments`（逐个素材分析+切片）
  与 `stimuli`（并发生成，每完成一题 +1）。找不到 token 返回 `{"found": false}` 而不是 404 ——
  前端因此不必处理异常分支。失败时 `error` 带上给用户看的文案。
  > 坑：`/api/start` 里 `os.link` 的目标文件名曾经也叫 `token`，与进度 token 撞名后进度会被覆盖，
  > 现在分别叫 `progress_token` 与 `audio_token`。

## 8. 前端

单页三视图：

**库管理**：列表、导入、删除、checkbox 选素材（1～3）、模式选择、开始测试。
页面里有两条提示：格式/时长要求；**选曲建议**（优先配器复杂、动态强烈的曲目，安静单调的段落即使在
32 kbps 下也可能听不出差别）。模式标签写明题量与每档题数（快速 6 题 · 每档 1 题 / 详细 22 题 · 每档 2 题）。
点「开始盲测」后，前端生成 `progress_token` 并轮询 `/api/progress`（350 ms 一次）：状态栏显示
「正在分析并切分片段（1/2）…」「正在生成测试音频（7/22）…」，按钮下方出现一条进度条。
**进度条延迟 300 ms 再显示** —— 命中缓存时整个准备只有几毫秒，不能闪一下。请求结束（成功或失败）
都会停掉轮询并收起进度条。
开始测试时的状态栏会报告本轮测试音频是"新生成 N 段 / 复用 M 段"。
「清理测试音频缓存（N MB）」按钮下面还有一行小字，把两个占用分开写清楚（避免让人以为按钮没生效）：
`测试音频缓存 X MB（上面那个按钮释放的就是它；下次测试重新生成）· 临时文件 Y KB（上传/编码草稿，后端每次启动自动清空，不需要手动管）`。

**测试**：题号/总数、进度条、A/B 播放按钮（循环、5 ms 淡入淡出）、三选项（A 更好 / B 更好 / 听不出）。不显示文件名、码率、编码器、时长。
A/B 下面有**试听音量滑条**（−40 dB…0 dB，到底 = 静音；`↑/↓` 键 ±2 dB）：它接在 5 ms 淡入淡出节点之后、
输出之前，**A/B 共用同一个 GainNode**，所以物理上不可能只把某一边调响，不影响判定。
滑条旁边那句提示由 `/api/start` 返回的 `target_lufs` 填出来（"测试音频已等响归一化到 −18 LUFS…"），
避免把数字写死在 HTML 里。

**作答后停留在本题**（不自动跳题）：高亮已选项、其余变暗，显示「已记录：X 更好」，并提供
「查看答案」（**可选**手动揭晓，默认不自动揭晓以免对后续题产生反馈偏置）与「下一题 →」（末题为「查看结果 →」）。
点「查看答案」会显示**两侧的具体身份**，例如 `A：无损 / B：MP3 96 kbps —— 无损在 B（正确答案：B 更好）。✘ 你选错了`；
安慰剂题显示 `A：无损 / B：无损（同一片段的两份拷贝）—— 安慰剂题，正确答案是「听不出」`。
这个函数**任何情况下都必须显示点东西**：拿不到 `reveal`（例如浏览器还在用旧脚本）时显示
「没能取到本题的答案信息……请按 Ctrl+Shift+R 强制刷新后重试」，绝不静默（见 §15 第 5 条）。
若出现同一题改选项的请求（正常 UI 不会发生），提示"本题已作答（首答即最终），本次修改已忽略"。
音频未载入完不允许作答，保证 `reaction_ms` 有效。
键盘：`1/2/3` 作答、`A/B/空格` 切换试听、回车或空格进入下一题。
离开测试页、点「再测一次」、或关闭标签页（`sendBeacon`）时都会调用 `/api/end`，不会留下"进行中"的引用。

**结果**：顶上一张**结论卡**（DOM 拼装，不用 innerHTML）—— 大字总体 `4/5 · 80%` + 状态徽章、
三个分档 chip（`≤96 kbps 2/2 · 样本不足`）、安慰剂题提示（`warning` 时变黄色）、样本量提示。
下面两张表：`码率区间 | 题数 | 答对 | 正确率 | 95% 区间 | 判定`（逐档表多一列"平均反应"），
「判定」列是彩色徽章（`显著高于 1/3` / `无显著差异` / `样本不足`），n≥3 时徽章下再附一行 `p = …`；
总体数字只出现在结论卡里，不再在表里重复一行。整段文字说明收进 `<details>`「统计口径与样本量说明」
（含纯文本 `conclusion` 便于复制），默认收起。表格外层 `.table-wrap` 可横向滚动，窄屏不挤成一团。

深色背景，大按钮，移动端可用，不做花哨动效。

## 9. 验收标准

1. `pip install -r requirements.txt && python app.py` 直接跑（更显式的写法是
   `uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1`；前提：系统已有 `ffmpeg` / `ffprobe`）。
2. 浏览器访问 `http://localhost:8000` 走通导入、测试、结果（作者在 Windows 浏览器上验证）。
3. FLAC / WAV / AIFF 可导入，导入后为 48 kHz FLAC。
4. MP3 / AAC / OGG / Opus 被拒绝，前端显示明确错误。
5. 重复导入同 hash 被识别，不重复入库。
6. 删除库音频时同时删除 `cache/<audio_id>/`。
7. 每题 A/B 响度差 ≤ 0.3 LU。
8. **作答之前**的任何前端请求都看不到码率 / 编码器 / 文件名 / 是否安慰剂；作答之后 `/api/answer`
   才揭晓**本题**两侧的具体身份（无损 / MP3 xx kbps）。
9. 结果页正确率、安慰剂统计、p 值与"样本不足"判定正确；每个统计行都带 `status` 枚举供前端上色。
10. 后端重启清空 `data/session/`，保留 `data/library/`。
11. `/api/end` 后 session 目录被删。
12. 改 `SEGMENT_ALGO_VERSION` 后 cache 失效重算。
13. 带内嵌封面的源可以正常导入并完成出题（不再 rc=234）。
14. 首答即最终：改了选项不改分、不揭晓答案；同选项重复提交幂等。
15. 出题顺序每轮随机（含安慰剂），而每轮用到的测试音频集合完全一致（缓存必然复用）。
16. `/` 与 `/static/*` 返回 `Cache-Control: no-cache, must-revalidate`，页面用 `?v=N` 引用 JS/CSS ——
    浏览器不会拿旧脚本去调新接口（见 §15 第 5 条）。
17. 试听音量滑条只改变监听增益（A/B 共用同一节点，不可能单边调整）；测试音频文件本身统一归一到 −18 LUFS，
    `/api/start` 把 `target_lufs` 报给前端。
18. **选曲顺序不影响题表**：同一批素材、不同勾选顺序，第二轮 `generated == 0`（否则缓存复用退化）。
19. **并发准备与串行结果一致**：并发数换成 1 / 4，缓存索引条目一个不少，解码后的样本逐样本一致。
20. **准备进度可查**：`/api/start` 带 `progress_token` 时 `/api/progress` 能拿到 `phase/done/total`，
    完成后 `finished == true`；未知 token 返回 `{"found": false}`。
21. **详细模式预热快速补集**：冷缓存跑详细 → 响应里 `prewarmed >= 1`；紧接着切快速 →
    `generated == 0`（预热前这一步要补 1 对、约 0.7 s）。`PREWARM_SHORT_COMPLEMENT = False` 可关闭。

自动化覆盖对照（`make test`，71 项）：#1 见 README「快速开始」（`python app.py`）；#3/#4/#5 由 `test_api.py`
的导入与拒绝用例覆盖；#6/#10/#11 由清理用例覆盖（含硬链接 inode 断言、"答完/结束后才能删除"的回归用例，
以及 `test_clear_cache_only_touches_stimuli_cache`：**「清理测试音频缓存」只清 `data/cache`**，
不碰 `data/tmp` / `data/session` / `data/library`）；
#7 的实测值为 `ΔLUFS = 0.0000`（远严于 0.3，`test_prepare_pair_loudness_and_peak` 断言 ≤ 0.05）；#8 由
`test_question_does_not_leak_bitrate_or_codec` 覆盖（键集合白名单 + 禁用子串 + 两路 `Content-Length` 相等），
"作答后才揭晓"由 `test_answer_reveals_both_sides_only_after_answering` 覆盖；#9 由
`test_result_counts_correct_answers` / `test_result_flags_guessing` / `test_result_statistics_payload` /
`test_binom_p_value_and_required_n` / `test_row_stats_marks_small_samples_as_undecidable` 覆盖；
#12 由 `test_segment_algo_version_invalidates_cache` / `test_stimuli_algo_version_moves_cache_dir` 覆盖；
#13 由 `test_cut_segment_drops_attached_picture` / `test_import_strips_attached_picture` /
`test_start_with_cover_art_source_builds_stimuli` 覆盖；#14 由 `test_answer_is_final_and_replay_is_idempotent`
覆盖；#15 由 `test_questions_shuffle_but_keep_identical_stimulus_set` / `test_stimuli_cached_and_reusable` 覆盖；
#16 由 `test_static_assets_are_revalidated` 覆盖（含"前端不再引用旧 `correct_side` 字段"的静态断言）；
#17 由 `test_frontend_volume_slider_is_wired` 覆盖（滑条存在、接线成 `gain → volume → destination`、
    `/api/start` 返回 `target_lufs == -18.0`）；
#18 由 `test_selection_order_does_not_change_stimulus_set` 覆盖；#19 由
`test_concurrent_stimuli_generation_is_identical_to_serial` / `test_concurrency_autodetect` /
`test_shared_locks_survive_multiple_event_loops` 覆盖；
#20 由 `test_progress_endpoint_reports_generation` / `test_progress_tracking_lifecycle` /
`test_frontend_prepare_progress_is_wired` 覆盖；#21 由 `test_detailed_round_prewarms_short_complement` /
`test_prewarm_is_detailed_only_and_switchable` 覆盖。
**只有 #2（浏览器实操）与 #7 的主观听感需要人来点一遍。**

## 10. 非目标（维护者红线）

README 的「它不做什么」是同一份清单的**用户口径**（只留用户能感知的 + 一句为什么）；
这里保留**规范口径**：判断"这个需求该不该做"时以本节为准，理由与交换条件在 §13。

**产品范围**

- 不做 ABX：当前是 A/B/听不出 三选一。真要 ABX 化，先看 §16「若日后要接着做」里为什么把它排在了后面。
- 不做自适应难度：难度就是码率阶梯，题量由 `SHORT_TEST_TOTAL` / `DETAILED_TEST_TOTAL` 决定。
- 不做账号、登录、权限，也不做多用户并发与持久历史：会话在内存、库信息一个 JSON，重启即丢（§5）。
- 不做 AAC / Opus 对比、不做 192 kHz 差异测试：只测 MP3；导入还要求源采样率 ≥ 44.1 kHz、时长 ≥ 30 秒。
- 不做动态片段长度：15 秒是所有统计与缓存算法的前提，改 `SEGMENT_SECONDS` 要连带重验对齐/缓存/统计。

**实现约束**

- 不引入前端构建工具或框架、不写数据库，以及 §2 那份技术禁用清单（Node / Docker / SQLite / librosa / …）。
- 这些不是"做不到"，而是"个人玩一玩"的定位下它们都会显著抬高维护成本；想加之前先确认 §13 的取舍仍然成立。

## 11. 实现顺序（历史记录）

项目已完整，这里只留一句脉络：**后端管线**（库读写 → 导入/去重 → 片段选择 → MP3 + 对齐 + 响度 → 出题）
→ **前端三视图** → **作答/结果/统计** → **会话清理与启动清理** → 逐条跑验收标准（§9）。
要复刻按这个顺序即可；后续新增功能不必受它约束。

## 12. 环境准备

```bash
# 前置：系统已有 ffmpeg / ffprobe（实测 6.1.1）；本项目不 apt install、不装任何全局包
make install     # = pip install -r requirements.txt（解释器自动探测，可用 PY= 覆盖）
make run         # 前台启动（Ctrl+C 停止）；或 make run-bg 后台启动 + make stop 停止
make test        # 71 项自动化测试（需要 make install-dev）
```

- **解释器**：`Makefile` 的 `PY` 会按「项目内 `.venv` → 作者的 conda 环境 `tin_ear` → `python3` → `python`」
  自动挑选，也可以用 `make run PY=/path/to/python` 覆盖。`HOST`（默认 `0.0.0.0`）与 `PORT`（默认 `8000`）同样可覆盖。
- **最省事的启动入口**：`python app.py` —— 就是 §16 里的 `__main__` 块，读 `HOST` / `PORT` 环境变量
  （默认 `0.0.0.0:8000`）、打印访问地址与"0.0.0.0 无认证"的提示，然后 `uvicorn.run(app, …)`。
  直接传 app 对象而不是 `"app:app"` 字符串：省掉一次模块导入，也避免两个 `FastAPI` 实例。
  **固定单进程**（会话在内存里）。由 `test_python_app_py_entry_point_serves_the_api` 守护 —— 它会真的
  起子进程、轮询 `/api/library`，所以 README 首推的这条命令不会悄悄坏掉。
- **公开仓库的可移植性**：README 给出通用 `venv` 路线；`make` 只是作者自用的便捷层，
  任何机器上 `python -m uvicorn app:app` 都能跑起来（只要装了 `requirements.txt` 与系统 ffmpeg）。
- **安全**：这是本地单用户工具，接口无认证。默认绑 `0.0.0.0` 是为了虚拟机 + 宿主机浏览器访问；
  共享/不可信网络请用 `HOST=127.0.0.1` 或自行加反代与鉴权。
- 临时文件留在项目内（`TMPDIR=data/tmp`，运行时另由 `_bootstrap` 的 `tempfile.tempdir` 兜底）：
  测试合成音频与 ffmpeg 中间文件动辄几百 MB，放项目内才好清理。**pip 下载缓存不再重定向**，
  用 pip 自己的默认位置；`make install/run/test` 都会打印选中的解释器，避免"静默用了别的环境"。
- **清理**（`data/library/` 里的音频永不删除，由 `make clean` 的固定目标列表保证）：
  `make size` 看占用；`make clean-cache` 清 `data/cache/`（快速模式跑满约 70 MB/首、重生成约 1 s；
  详细模式 245.8 MB/首、重生成约 4 s；含预热的快速补集则约 257 MB/首）；
  `make clean` 清所有可重建项（含 `data/session`、`data/tmp`、`__pycache__`、`.pytest_cache`、`data/server.log`，
  以及旧版本遗留的 `.pip-cache/`）。
  不要用 `git clean -xfd`：`data/` 在 `.gitignore` 里，会把用户导入的音频一起删掉。
- **非 Makefile 安装（README「最省事」/ 方式一：venv / Windows）**：清理就是直接删同名目录（`_bootstrap` 下次启动会重建），
  跑测试用 `TMPDIR="$PWD/data/tmp" python -m pytest -q`（PowerShell 用 `$env:TMPDIR`）——
  README「想清理磁盘占用怎么办」与「测试」两节给了 macOS/Linux 与 PowerShell 的可复制命令。
- 依赖全是 wheel（`soundfile` 自带 libsndfile），不需要编译器或系统库。

测试注意：关 Windows 音效增强、Dolby、EQ、空间音频；系统音量固定；WSL 不出声，用宿主机浏览器访问 localhost 播放。

## 13. 关键决策

**架构与音质**

- 无数据库：一次性玩，不需历史。
- 统一 48k FLAC：省空间、统一处理、无损参考；`index.json` 记源文件元信息以保留可追溯性。
- 固定 15 秒：盲测长度必须一致，否则统计被污染；48 kHz 下恰好 720000 采样 = 625 个 MP3 帧，
  与帧长精确对齐，裁剪后无残余。
- **配对独立响度增益 + 共同峰值衰减**（而不是"各自归一化 + 限幅压峰"）：响度差恒为 0，且全程零非线性，
  不给低码率题额外线索。
- **等响归一化目标取 -18 LUFS（而不是 -23）**：-23 是 EBU R128 广播标准，日常试听要一直把系统音量拧很大；
  -18 实测只把真峰值推到 -5.3 dBTP（离 -1 dBTP 上限还有 4.3 dB），几乎不会触发共同峰值衰减。
  若继续提到 -14 以上，波峰因数大的段落会被峰值上限压回、导致各题响度不一致（A/B 内部仍然等响，题仍然有效）。
  **注意：调这个常量不会改变题目难度** —— 两路是同一组增益同时放大，ΔLUFS 恒为 0、MP3 的伪影/信号比在编码时就定死了。
  改它必须 bump `STIMULI_ALGO_VERSION`，否则会复用旧响度的缓存。
- 不用 librosa：依赖重，ffmpeg + numpy + scipy 足够。
- ffmpeg 的所有转码/切片/编码命令都显式 `-map 0:a:0`：内嵌封面是独立的 `attached_pic` 视频流，
  一旦混进片段就会让 MP3 编码以 rc=234 失败（详见 §6）。

**缓存与性能**

- 测试音频持久缓存 + session 硬链接：重复测试不再重算（原设计每轮重算约 130 MB 且测完即丢）。
- **缓存"够用就好"：按需生成 + 详细模式顺手补补集**。题表按「模式 + 素材集合」确定、`audio_ids` 排序后
  round-robin，所以同一模式重测 100% 复用；切模式只补缺的那几档。**消掉"详细 → 快速"的 0.7 s** 用的是
  极便宜的补集预热（`PREWARM_SHORT_COMPLEMENT`，+1～2 对 ≈ 11 MB/首）；**消掉"快速 → 详细"的 17 个缺口**
  则只能首次生成全集（11 档 × 片段，快速模式首轮从 6 个变 22 个、磁盘 +约 190 MB/首）——
  对"只用快速模式的人"代价太大，所以不做。
- **生成按 CPU 核数并发**（`min(核数, 8)`，实测 22 题 15.3 s → 3.9 s）：单题 0.70 s 里 0.46 s 是
  我们自己的响度+对齐计算、0.25 s 是 ffmpeg 编解码，两者都能并行；8 路之后收益迅速变小（12 路 3.5 s），
  所以封顶 8。并发带来的共享资源（`index.json` 读改写、同一素材的片段生成）各自加了锁。

**盲测有效性与统计**

- 不透明 token：URL 与响应都不泄漏码率 / 编码器 / 是否安慰剂；**作答之后**才在 `/api/answer` 里揭晓本题两侧身份
  （为了"看清楚刚才听的是什么"），揭晓只针对已作答的本题，不会泄漏后面的题目。
- 安慰剂题 = 同一片段的两份无损拷贝：专门检测"听到了不存在的差异"。
- 出题**全表乱序**（含安慰剂）：位置不携带信息，逼着每题都真听；同时先建题再打乱，
  保证每轮 (片段, 码率) 集合不变 → 缓存 100% 复用。
- **统计口径用 1/3 而不是 1/2**：三选一（A 更好 / B 更好 / 听不出）的瞎猜基线是 1/3；
  显著性判定用**单侧精确二项检验**（小样本下 Wilson 区间只能当描述性参考），
  并明确"每档 < 3 题无法判定"（n=1 的最小 p 就是 1/3，结构上不可能显著）。
  要让结论变硬就把 `SHORT_TEST_TOTAL` / `DETAILED_TEST_TOTAL` 调大（阶梯会循环重复）；
  达到 80% 功效约需每档 7 题（由 `_required_n()` 算出并显示在结果页）。
- 作答**首答即最终**：否则"先点听不出 → 看揭晓码率 → 改成正解"就能刷分。
- **前端再加一层试听音量**（-40…0 dB 滑条 + `↑/↓`）：它只改监听增益（A/B 共用同一个 GainNode），
  不写回文件、不可能单边调整，因此既方便又不会破坏盲测；界面把 `target_lufs` 显示出来，
  让用户知道"文件本身已经够响，滑条只是舒适度"。

**环境与文档**

- **保留 `python app.py` 作为首推启动方式**：README 的"最省事"一节用它，显式 uvicorn 命令与 `make run`
  退居"参数看得见/作者自用"。理由：新用户记不住 `--workers 1` 与 `-m uvicorn app:app` 这一串，
  而入口本来就在 `__main__` 里；顺带把地址与安全提示打印出来（WSL 场景最需要）。
- **用户可见的名词统一叫「测试音频」**：`stimulus/刺激` 是心理声学的正式术语，但在中文 UI 里第一联想
  是电刺激，读起来怪。只改**面向用户**的文案与错误信息；内部标识（`stimuli/` 目录、
  `STIMULI_ALGO_VERSION`、`_ensure_stimuli`、`STIMULUS_SUBTYPE`）保留 —— `STIMULUS_SUBTYPE` 更是
  soundfile 的 WAV subtype 术语，改名反而混淆。
- **pip 缓存不写进项目**：原来用 `PIP_CACHE_DIR` 把 pip 缓存顶到 `.pip-cache/`，收益只是"不写 `~/.cache`"，
  代价是项目目录常驻 72 MB、与 pip 生态惯例相悖、还得单独写一段文档解释。已取消。

## 14. 测试

```bash
make test     # = TMPDIR=<项目>/data/tmp $(PY) -m pytest -q（PY 见 §12，可覆盖）
make run-bg   # 后台起服务（日志 data/server.log），make stop 停止
```

- `tests/conftest.py`：lavfi 合成音频（声部 + 粉噪 + 动态包络 + 瞬态，35 s / 65 s 两种）；
  `attach_cover_flac()` 造带 `attached_pic` 封面的 FLAC；`stream_types()` 用 ffprobe 看文件里有哪些流；
  `isolated_data` 夹具把 app 的全部数据目录重定向到本测试自己的临时目录；`fast_config` 把题量压到最小让 API 测试秒级完成。
- `tests/test_pipeline.py`：常量守护、对齐符号与真实 MP3 往返、等响度/峰值、**增益必须是纯线性**（防限幅器）、
  切片逐采样精确与**切片必须丢掉封面流**、片段选择确定性与不重叠、缓存版本失效、
  出题乱序（集合恒定 / 顺序变化 / 安慰剂位置打散 / 每档题数）、二项检验与功效阈值、
  缓存复用计数、WAV 头规范（48 kHz / 2ch / 720000 / FLOAT）、
  **选曲顺序不影响题表**（`test_selection_order_does_not_change_stimulus_set`）、
  **并发与串行产物一致 + 并发数自适应**（`test_concurrent_stimuli_generation_is_identical_to_serial`
  会比较缓存索引条目数与**解码后的样本**；注意不能比文件字节 —— libsndfile 的 `PEAK` chunk 带时间戳）、
  **进度状态机与过期清理**（`test_progress_tracking_lifecycle` / `test_concurrency_autodetect`）、
  **共享锁的循环作用域**（`test_shared_locks_survive_multiple_event_loops`：两个 `asyncio.run` 循环里
  各争用一次同一把锁 —— 这正是 `_loop_scoped_lock` 要防的坑）、
  **详细模式预热快速补集**（`test_detailed_round_prewarms_short_complement` /
  `test_prewarm_is_detailed_only_and_switchable`：冷缓存 detailed 报告 `prewarmed >= 1`，
  接着 short 必须 `generated == 0`；关掉开关后 short 又需要补 1 个）。
- `tests/test_api.py`：导入与去重（含带封面源的导入）、有损格式拒绝、删除与 409（含"答完全部题目后即可删除"的回归）、
  题目参数校验、**防泄漏**、`reveal` 只在作答后返回且只含本题、首答即最终与幂等重放、安慰剂两路内容相同、
  带封面源的端到端出题回归、结果统计（p 值/显著/样本不足/`status` 枚举）与结论文案、
  **静态资源强制重验证**（`test_static_assets_are_revalidated`）、
  **试听音量滑条接线**（`test_frontend_volume_slider_is_wired`：`gain → volume → destination`、
  `/api/start` 报 `target_lufs`）、
  **「清理测试音频缓存」的作用域**（`test_clear_cache_only_touches_stimuli_cache`：只清 `data/cache`）、
  **准备进度端点**（`test_progress_endpoint_reports_generation`：未知 token 返回 `found: false`，
  完成后 `finished/done/total` 正确）、**前端进度条接线**（`test_frontend_prepare_progress_is_wired`）、
  启动清理、TTL 清理、路径穿越；以及一条特殊的端到端用例
  **`test_python_app_py_entry_point_serves_the_api`**（守护 README 首推的 `python app.py`）：
  把 `app.py` 与 `static/` 复制到临时目录再起真子进程（`HOST=127.0.0.1 PORT=<空闲端口>`），
  轮询 `/api/library` 直到 200 再把进程收掉。**不能**直接在当前工作区起子进程 ——
  `_bootstrap()` 会清空 `data/tmp`，而那正是 pytest 临时目录所在的地方。
- `pytest.ini`：`tmp_path_retention_policy = failed` + `tmp_path_retention_count = 1` —— 测试产物落在
  项目内 `data/tmp/`，合成音频一次就有几百 MB，默认策略会留下最近几轮（实测堆到 907 MB）。
  注意 `count = 1` 的含义是"会话级 `tmp_path_factory` 目录保留最近 1 轮"，所以**失败的那一轮**会留下
  `synth0/` 之类的目录（几十到上百 MB）供排查；全绿时只剩一个空的 `pytest-0/` 骨架。
- 测试产物全部落在 `data/tmp/`（受 `TMPDIR` 约束），不写系统 temp 或用户目录。

## 15. 实测数据与实现记录（ffmpeg 6.1.1-3ubuntu5 / Python 3.12.15）

| 项 | 实测值 |
|---|---|
| 15 s @48 kHz | 720000 采样 = 恰好 625 个 MPEG-1 Layer III 帧 |
| MP3 编码到**文件**（`-write_xing 1` 生效） | 解码回来仍是 720000 采样，lag = 0 |
| MP3 编码到**管道**（无 LAME 头） | 解码得 721152 采样，lag = 1105（23.0 ms）；对齐器可兜住，结果仍 720000 / ΔLUFS = 0 |
| 每个 (片段, 码率) 组合 | ΔLUFS = 0.0000 LU（32/64/128/320 kbps 均如此）；TARGET_LUFS = -18 → True Peak ≈ -5.3 dBTP；共同衰减 delta = 0（距 -1 dBTP 上限还有 4.3 dB） |
| 响度目标对比 | -23 LUFS（EBU R128）时真峰值 ≈ -10.3 dBTP，余量 9.3 dB —— 白白浪费动态余量、日常试听偏轻；-18 后余量 4.3 dB |
| 单个测试音频文件 | 5760088 字节（5.76 MB + WAV 头），A/B 完全等长 |
| 缓存体积 | detailed（20 真实 + 2 安慰剂、2 个片段）实测 **245.8 MB**/首（**预热功能之前**的数字）；加上预热的快速补集约 **257 MB**/首；两套阶梯并集 11 档 × 2 片段 + 2 安慰剂 ≈ **267 MB**/首（46 个 WAV + 片段）。旧版单片段设计下 detailed 只释放 130271006 字节，已不适用 |
| `/api/start` 首次生成 10 对测试音频（**串行时代**的旧数据，并发行见下） | 7.8 s；命中缓存后 < 0.2 s |
| `/api/start` 首次 detailed（22 题、20 对真实测试音频 + 2 安慰剂） | **15.8 s**（`generated: 22, reused: 0`）；再开一轮 `generated: 0, reused: 22`，**0.01 s** |
| 单题生成耗时拆解（24 核机器，65 s 合成源） | **746 ms/题**：`_prepare_pair`（响度+对齐，numpy）462 ms、ffmpeg 编码 163 ms、解码 83 ms、写 2 个 WAV 24 ms、读片段 FLAC 14 ms —— **瓶颈是我们自己的 numpy，不是 ffmpeg** |
| 并发生成加速比（同 22 题冷启动，`_prepare_session` 并发） | 1 路 15.32 s / 2 路 8.32 s（1.84×）/ 4 路 5.01 s（3.05×）/ 6 路 4.12 s（3.72×）/ **8 路 3.87 s（3.95×）** / 12 路 3.52 s（4.35×）—— 8 之后基本饱和，所以 `STIMULI_CONCURRENCY_MAX = 8` |
| 缓存命中（22 题全部复用） | 0.004 s（只读 index.json + 硬链接） |
| 带内嵌封面的源（复现 rc=234） | 未加 `-map 0:a:0` 时 `ffmpeg` 返回 rc=234 + `Attached pictures were requested, but the ID3v2 header is disabled`；加上后三步全部 rc=0 |
| 统计口径实测 | `POWER_REQUIRED_N = 7`；20/20 的精确 p = 2.9e-10（`p<0.001`）；22 题一轮的安慰剂位置实测出现过 [0, 17]（即被正确打散） |
| 随机 token 的"包含码率数字"概率 | 16 位 hex、15 个两位窗口 → 含某个指定的两位数概率 **5.74%**（实测 20000 次采样）；这就是那条防泄漏断言原来的假阳性率，已在测试里固定随机源修掉 |
| 并发冷启动端到端（真 uvicorn + HTTP，1 首 65 s 素材、detailed；**预热功能之前**的实测） | `generated: 22, reused: 0` 用时 **4.0 s**（进度 0→22 正常上报）；同素材第二轮 `generated: 0, reused: 22`；详细→快速补生成 **1** 个（224 kbps 只属快速阶梯）、快速→详细 `0 / 22`；缓存 245.5 MB/首 |
| 详细模式预热快速补集（同上素材，冷缓存） | detailed：`generated: 22, reused: 0, prewarmed: 1`，用时 **4.4 s**（进度总量 23）；紧接着 short：`generated: 0, reused: 6`，用时 **0.009 s**（预热前这一步是 0.7 s） |
| 依赖 | 全部 wheel（soundfile 自带 libsndfile），无需编译器或系统库 |

> 另有一个**不是 bug 的坑**：比较测试音频的**文件字节**永远不可能相等 —— libsndfile 写 WAV 时会加一个
> `PEAK` chunk，里面带 `timeStamp`（`time_t`）。实测同一段数组连写两次，`data` chunk 完全相同、
> 解码后逐样本相同，只有那 4 个时间戳字节不同。所以"产物一致性"要比较**解码后的样本**，
> 不要比文件 sha256（`test_concurrent_stimuli_generation_is_identical_to_serial` 就是这么写的）。

实现期间修掉的五个真问题（都已加测试守护）：

1. **`-ss` 位置**：输出 seek（`-i` 之后）在 `start=40s` 会输出整段静音，必须用输入 seek。
2. **对齐精修符号**：`_estimate_lag` 精修阶段的平移方向与负延迟覆盖写反，需与 `_shifted` 保持一致
   （`aligned[k] = cand[k + lag]`）。
3. **响度公式**（对应 §6 修正）：若对两路施加"同一个增益"，两路响度差会保持原样而不是归零；
   正确做法是**各自**按 loudness 增益到目标，再施加**共同**的峰值余量衰减。
4. **内嵌封面流**（对应 §6 修正）：`-map_metadata -1` 清不掉 `attached_pic` 视频流，切片时必须 `-map 0:a:0`，
   否则封面帧进入片段、MP3 muxer 试图写 ID3v2 APIC 而被 `-id3v2_version 0` 拒绝 → rc=234。
   注意该 bug 只在"最佳片段起点 = 0"时触发（起点 > 0 时封面帧落在窗口外）。
5. **浏览器缓存旧 `app.js` 导致的静默失败**：揭晓改成 `reveal` 之后，旧脚本仍去读 `data.correct_side`，
   拿到 `undefined` 后在 `undefined.toUpperCase()` 上抛异常；而它抛异常前已经先把「查看答案」按钮
   `hidden` 掉了，于是用户只看到"按钮消失、页面什么都不显示"。修法三条同时上：
   `/` 与 `/static/*` 强制重验证（§7）、`index.html` 用 `?v=N` 引用资源、揭晓函数改成**任何分支都显示东西**
   （拿不到 `reveal` 就提示强制刷新）。由 `test_static_assets_are_revalidated` 守护。
   > 教训：凡是"点了按钮但什么都没发生"的交互，函数里都不能有会抛异常的隐式假设。

## 16. 项目现状（交接摘要）

- **状态**：功能完整可用。`make test` **71 项**全绿（约 1 分钟）。
- **改动历史**：看 `git log --oneline`。最近一轮（缓存顺序无关 + 并发生成 + 进度条 + 术语改「测试音频」+
  环境配置收敛 + 详细模式预热快速补集 + `python app.py` 一键启动）之后，README/AGENTS 都已同步；
  历史上修过的真问题都在 §15，不必从提交信息里考古。
- **关键实测**：每对 ΔLUFS = 0.0000 LU、真峰值 ≈ −5.3 dBTP（−18 LUFS 下）、两路各 5760088 字节（无体积泄漏）、
  22 题详细模式**串行 15.8 s / 8 路并发 3.9 s**、缓存复用 0.01 s、测试音频缓存 245.8 MB/首（含预热 257 MB）、
  详细模式顺手预热后**两个方向切换都是 0.01 s**、`POWER_REQUIRED_N = 7`。完整表见 §15。
- **待人工验证**：验收 #2（浏览器实操，含新的进度条）与 #7 的主观听感；其余 20 条已自动化（对照表见 §9）。
  验收 #1（安装即跑）现在由 `python app.py` 这条命令覆盖，并有测试真的把子进程起起来探活。
- **已定位并修掉的测试假阳性**：`test_question_does_not_leak_bitrate_or_codec` 曾约每十轮挂一次，
  现象是 `assert '96' not in '47398a8e90e79657'`。根因**不在产品代码**：不透明 token 是随机 16 位 hex，
  15 个两位窗口里出现 `96` 的概率实测 **5.74%**，而那条断言把"token 恰好含码率数字"误判成泄漏
  （HEAD 基线同样复现，见 §15 的 A/B 记录）。修法是把测试里的 `secrets.token_hex` 换成**只含字母、
  互不相同**的确定序列，断言因此变确定；token 仍然与码率无关。
  > 教训：拿"包含某子串"当安全断言，遇到随机串必然偶发假阳性；要么固定随机源，要么改判结构性质。
- **可一行调整的行为**（都在 `app.py` 顶部）：`TARGET_LUFS`（改完要 bump `STIMULI_ALGO_VERSION`）、
  `SHORT_TEST_TOTAL` / `DETAILED_TEST_TOTAL` 与两条码率阶梯、`SHUFFLE_ALL_QUESTIONS`、
  `PLACEBO_SAME_SEGMENT`、`PREWARM_SHORT_COMPLEMENT`、`SEG_WEIGHTS`、
  `CHANCE_LEVEL` / `ALPHA` / `MIN_N_FOR_SIGNIFICANCE`、
  `STIMULI_CONCURRENCY_MAX`（并发上限；运行时可用 `TINEAR_CONCURRENCY=4` 覆盖）；
  前端是否自动揭晓答案在 `static/app.js` 的 `renderConfirmation()`（默认只显示"已记录"，答案需手动点）；
  试听音量范围/步长在 `static/index.html` 的 `#volume-input` 与 `app.js` 的 `VOLUME_*` 常量。
  **改完 `static/` 记得把 `index.html` 里的 `?v=N` 加一。**
- **数据与清理**：`data/library/` 是用户的音频（持久，永不自动删除）；`data/cache/` 可随时重建；
  `data/session/`、`data/tmp/` 每次启动自动清空。清理命令见 §12（`make size` / `clean-cache` / `clean`），
  面向用户的说明在 README「想清理磁盘占用怎么办」；pip 缓存由 pip 自己管（`python -m pip cache purge`）。
- **若日后要接着做**：优先考虑「提高检验功效」（`DETAILED_TEST_TOTAL` 调大或新增重复档位模式）、
  ABX 化（当前非目标）、以及给多点位/多版本对比留接口；改动测试音频内容一律 bump `STIMULI_ALGO_VERSION`。
