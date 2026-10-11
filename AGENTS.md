# AGENTS.md — TinEar

> 文档分工：[README.md](README.md) 面向**使用者**（项目介绍、技术栈、安装、使用命令、排错），
> 本文件面向**维护者**（开发细节、设计取舍、测试、实测数据、项目现状与交接信息）。
> 许可证：MIT，见 [LICENSE](LICENSE)。

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
- WSL Ubuntu 跑后端，浏览器访问 `http://localhost:8000`（作者的验证环境，代码本身跨平台）
- 许可证：MIT

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
          v2/             # v{STIMULI_ALGO_VERSION}，任何改变刺激内容的常量都要 bump（当前 = 2）
            index.json    # 每对刺激的 LUFS / True Peak / lag / 增益
            seg0_96k_ll.wav + seg0_96k_mp3.wav
            seg0_p0.wav   # 安慰剂（A/B 同一份文件）
    session/              # 临时：当前测试刺激的硬链接，测完即删
      <session_id>/
    tmp/                  # 临时：上传落盘与 ffmpeg 中间文件，启动清空
  .pip-cache/             # pip 下载缓存（项目内，gitignore）
  requirements.txt        # 运行期依赖（正常安装只需要这一份）
  requirements-dev.txt    # 测试依赖（pytest / httpx）
  requirements.lock.txt   # 作者机器实测通过的环境快照（pip list --format=freeze），仅供参考
  pytest.ini              # 测试产物保留策略（只留失败那一轮的临时目录）
  Makefile                # install / install-dev / run / run-bg / stop / test
                          # clean-cache / clean-pip-cache / clean / size
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
STIMULI_ALGO_VERSION = 2           # 刺激缓存版本：改 TARGET_LUFS / 峰值上限 / MP3 参数都要 bump
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
FFMPEG_CONCURRENCY = 2
MIN_SOURCE_SAMPLE_RATE, MIN_SOURCE_SECONDS = 44100, 30
SHUFFLE_ALL_QUESTIONS = True       # 真实题与安慰剂题统一打乱，位置不携带任何信息
PLACEBO_SAME_SEGMENT = True        # 默认：安慰剂 = 同一片段的两份无损拷贝
RESULT_BRACKETS = [("≤96 kbps"), ("128–192 kbps"), ("≥224 kbps")]
CHANCE_LEVEL = 1.0 / 3.0           # 三选一（A/B/听不出）的瞎猜基线，不是 1/2
ALPHA = 0.05                       # 单侧精确二项检验的显著性水平
POWER_TARGET, P1_FOR_POWER = 0.8, 0.8   # 算"每档需要多少题"用的功效与备择听出率
MIN_N_FOR_SIGNIFICANCE = 3         # 少于 3 题任何结果都无法判定（n=1 的最小 p 就是 1/3）
```

注意：`SHORT_TEST_BITRATES` 并非 `DETAILED_TEST_BITRATES` 的子集（**224 kbps 只在 short 档**），
两条阶梯并集共 11 档，这也是刺激缓存的档位上限。

统计口径（实测 `POWER_REQUIRED_N = 7`）：单侧精确二项检验 `P(X ≥ k | p = 1/3)`，每档题数与临界答对数 ——
n=1 任何结果都不可能显著（最小 p = 1/3）、n=3 需 3/3（p=0.037）、n=4 需 4/4（p=0.012）、
n=5 需 4/5（p=0.045）、n=10 需 7/10（p=0.020）。详细模式 20 道真实题下分档 n = 10 / 6 / 4。

## 5. 存储规则

- `library/` 持久。启动不清理。
- `cache/` 持久。`meta.json` 中 `SEGMENT_ALGO_VERSION` 不匹配时重算片段；`stimuli/v{STIMULI_ALGO_VERSION}/`
  作为刺激缓存，版本不匹配即整目录重算。
- `session/` 与 `tmp/` 临时。后端启动清空（`data/library/`、`data/cache/` 保留）。
  `/api/start` 时清理超 `SESSION_TTL_SECONDS` 的旧目录与无主目录。`/api/end` 删除当前 session 目录。
- `session/` 内是 `cache/` 刺激文件的**硬链接**（跨文件系统时回退为复制），因此
  `/api/end` 只删链接，刺激缓存仍在，重复测试几乎瞬时开始。`/api/start` 的响应里报告本轮
  `stimuli: {generated, reused}`，前端把它显示出来，"到底复用没复用"不再靠猜。
- `tempfile.tempdir` 指向 `data/tmp/`：不写系统 temp 或用户目录。
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

### 生成刺激

- 缓存在 `cache/<audio_id>/stimuli/v{STIMULI_ALGO_VERSION}/`，键 = `seg<i>_<bitrate>k`（安慰剂为 `seg<i>_p<seg>`）；
  命中即复用，`session/<session_id>/` 只放硬链接。**不再每轮重算**（原 §6 写 session 目录与 §3 的 `stimuli/` 自相矛盾）。
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
  这样每轮用到的刺激集合完全一致 → 缓存必然全部复用），然后**对整张题表 `rng.shuffle`**
  （`SHUFFLE_ALL_QUESTIONS = True`），最后统一重编号 `index` / `question_id`。
  真实题与安慰剂题一起打乱："第几题"既不能预判码率高低，也不能预判哪几题是无差异的，
  逼着人每题都真听（否则无脑按"听不出"会把真实题全部答错）。
- 从选中音频池轮流取（round-robin），每首第 k 次出场在 `seg_0/seg_1` 间交替。
- 安慰剂题的 A、B 默认是**同一片段的两份无损拷贝**（内容完全相同），否则"听不出"不是唯一正确答案。
- 每题随机 `lossless_is_a`；题目仅存内存。
- `/api/start` 一次性准备好全部题目的刺激并硬链接进 session 目录 —— 会话自包含，之后删库不影响进行中的测试。

## 7. API

```
GET    /api/library                       → { items, cache_bytes, tmp_bytes }
POST   /api/library/upload        form-data: files   → { results: [{filename,status,audio_id,reason}] }
DELETE /api/library/{audio_id}            → 200 { deleted, freed_bytes } / 404 / 409（被**尚未完成**的测试引用）
DELETE /api/cache                         → 清空片段与刺激缓存 data/cache { freed_bytes }

POST /api/start                   body: { audio_ids, mode }
                                  → { session_id, total, mode, stimuli: {generated, reused}, target_lufs }
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

## 8. 前端

单页三视图：

**库管理**：列表、导入、删除、checkbox 选素材（1～3）、模式选择、开始测试。
页面里有两条提示：格式/时长要求；**选曲建议**（优先配器复杂、动态强烈的曲目，安静单调的段落即使在
32 kbps 下也可能听不出差别）。模式标签写明题量与每档题数（快速 6 题 · 每档 1 题 / 详细 22 题 · 每档 2 题）。
开始测试时的状态栏会报告本轮刺激是"新生成 N 个 / 复用 M 个"。
「清理刺激缓存（N MB）」按钮下面还有一行小字，把两个占用分开写清楚（避免让人以为按钮没生效）：
`刺激缓存 X MB（上面那个按钮释放的就是它；下次测试重新生成）· 临时文件 Y KB（上传/编码草稿，后端每次启动自动清空，不需要手动管）`。

**测试**：题号/总数、进度条、A/B 播放按钮（循环、5 ms 淡入淡出）、三选项（A 更好 / B 更好 / 听不出）。不显示文件名、码率、编码器、时长。
A/B 下面有**试听音量滑条**（−40 dB…0 dB，到底 = 静音；`↑/↓` 键 ±2 dB）：它接在 5 ms 淡入淡出节点之后、
输出之前，**A/B 共用同一个 GainNode**，所以物理上不可能只把某一边调响，不影响判定。
滑条旁边那句提示由 `/api/start` 返回的 `target_lufs` 填出来（"刺激已等响归一化到 −18 LUFS…"），
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

1. `pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000` 直接跑
   （前提：系统已有 `ffmpeg` / `ffprobe`）。
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
15. 出题顺序每轮随机（含安慰剂），而每轮用到的刺激集合完全一致（缓存必然复用）。
16. `/` 与 `/static/*` 返回 `Cache-Control: no-cache, must-revalidate`，页面用 `?v=N` 引用 JS/CSS ——
    浏览器不会拿旧脚本去调新接口（见 §15 第 5 条）。
17. 试听音量滑条只改变监听增益（A/B 共用同一节点，不可能单边调整）；刺激文件本身统一归一到 −18 LUFS，
    `/api/start` 把 `target_lufs` 报给前端。

自动化覆盖对照（`make test`，61 项）：#1 见 README 的安装/运行两行命令；#3/#4/#5 由 `test_api.py`
的导入与拒绝用例覆盖；#6/#10/#11 由清理用例覆盖（含硬链接 inode 断言、"答完/结束后才能删除"的回归用例，
以及 `test_clear_cache_only_touches_stimuli_cache`：**「清理刺激缓存」只清 `data/cache`**，
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
    `/api/start` 返回 `target_lufs == -18.0`）。
**只有 #2（浏览器实操）与 #7 的主观听感需要人来点一遍。**

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
# 前置：系统已有 ffmpeg / ffprobe（实测 6.1.1）；本项目不 apt install、不装任何全局包
make install     # = pip install -r requirements.txt（解释器自动探测，可用 PY= 覆盖）
make run         # 前台启动（Ctrl+C 停止）；或 make run-bg 后台启动 + make stop 停止
make test        # 61 项自动化测试（需要 make install-dev）
```

- **解释器**：`Makefile` 的 `PY` 会按「项目内 `.venv` → 作者的 conda 环境 `tin_ear` → `python3` → `python`」
  自动挑选，也可以用 `make run PY=/path/to/python` 覆盖。`HOST`（默认 `0.0.0.0`）与 `PORT`（默认 `8000`）同样可覆盖。
- **公开仓库的可移植性**：README 给出通用 `venv` 路线；`make` 只是作者自用的便捷层，
  任何机器上 `python -m uvicorn app:app` 都能跑起来（只要装了 `requirements.txt` 与系统 ffmpeg）。
- **安全**：这是本地单用户工具，接口无认证。默认绑 `0.0.0.0` 是为了虚拟机 + 宿主机浏览器访问；
  共享/不可信网络请用 `HOST=127.0.0.1` 或自行加反代与鉴权。
- pip 缓存与临时文件都留在项目内（`.pip-cache/`、`data/tmp/`），不写系统 temp、`~/.cache`，
  也不动 `base` 或其他项目的 conda 环境。
- **清理**（`data/library/` 里的音频永不删除，由 `make clean` 的固定目标列表保证）：
  `make size` 看占用；`make clean-pip-cache` 清 `.pip-cache/`（实测 72 MB）；
  `make clean-cache` 清 `data/cache/`（实测 67 MB/首，短模式重生成约 8 s）；
  `make clean` 清所有可重建项（含 `data/session`、`data/tmp`、`__pycache__`、`.pytest_cache`、`data/server.log`）。
  不要用 `git clean -xfd`：`data/` 在 `.gitignore` 里，会把用户导入的音频一起删掉。
- 依赖全是 wheel（`soundfile` 自带 libsndfile），不需要编译器或系统库。

测试注意：关 Windows 音效增强、Dolby、EQ、空间音频；系统音量固定；WSL 不出声，用宿主机浏览器访问 localhost 播放。

## 13. 关键决策

- 无数据库：一次性玩，不需历史。
- 统一 48k FLAC：省空间、统一处理、无损参考；`index.json` 记源文件元信息以保留可追溯性。
- 固定 15 秒：盲测长度必须一致，否则统计被污染；48 kHz 下恰好 720000 采样 = 625 个 MP3 帧，
  与帧长精确对齐，裁剪后无残余。
- **配对独立响度增益 + 共同峰值衰减**（而不是"各自归一化 + 限幅压峰"）：响度差恒为 0，且全程零非线性，
  不给低码率题额外线索。
- 刺激持久缓存 + session 硬链接：重复测试不再重算（原设计每轮重算约 130 MB 且测完即丢）。
- 不透明 token：URL 与响应都不泄漏码率 / 编码器 / 是否安慰剂；**作答之后**才在 `/api/answer` 里揭晓本题两侧身份
  （为了"看清楚刚才听的是什么"），揭晓只针对已作答的本题，不会泄漏后面的题目。
- 安慰剂题 = 同一片段的两份无损拷贝：专门检测"听到了不存在的差异"。
- 出题**全表乱序**（含安慰剂）：位置不携带信息，逼着每题都真听；同时先建题再打乱，
  保证每轮 (片段, 码率) 集合不变 → 缓存 100% 复用。
- 不用 librosa：依赖重，ffmpeg + numpy + scipy 足够。
- **统计口径用 1/3 而不是 1/2**：三选一（A 更好 / B 更好 / 听不出）的瞎猜基线是 1/3；
  显著性判定用**单侧精确二项检验**（小样本下 Wilson 区间只能当描述性参考），
  并明确"每档 < 3 题无法判定"（n=1 的最小 p 就是 1/3，结构上不可能显著）。
  要让结论变硬就把 `SHORT_TEST_TOTAL` / `DETAILED_TEST_TOTAL` 调大（阶梯会循环重复）；
  达到 80% 功效约需每档 7 题（由 `_required_n()` 算出并显示在结果页）。
- 作答**首答即最终**：否则"先点听不出 → 看揭晓码率 → 改成正解"就能刷分。
- **等响归一化目标取 -18 LUFS（而不是 -23）**：-23 是 EBU R128 广播标准，日常试听要一直把系统音量拧很大；
  -18 实测只把真峰值推到 -5.3 dBTP（离 -1 dBTP 上限还有 4.3 dB），几乎不会触发共同峰值衰减。
  若继续提到 -14 以上，波峰因数大的段落会被峰值上限压回、导致各题响度不一致（A/B 内部仍然等响，题仍然有效）。
  **注意：调这个常量不会改变题目难度** —— 两路是同一组增益同时放大，ΔLUFS 恒为 0、MP3 的伪影/信号比在编码时就定死了。
  改它必须 bump `STIMULI_ALGO_VERSION`，否则会复用旧响度的缓存。
- **前端再加一层试听音量**（-40…0 dB 滑条 + `↑/↓`）：它只改监听增益（A/B 共用同一个 GainNode），
  不写回文件、不可能单边调整，因此既方便又不会破坏盲测；界面把 `target_lufs` 显示出来，
  让用户知道"文件本身已经够响，滑条只是舒适度"。
- ffmpeg 的所有转码/切片/编码命令都显式 `-map 0:a:0`：内嵌封面是独立的 `attached_pic` 视频流，
  一旦混进片段就会让 MP3 编码以 rc=234 失败（详见 §6）。

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
  缓存复用计数、WAV 头规范（48 kHz / 2ch / 720000 / FLOAT）。
- `tests/test_api.py`：导入与去重（含带封面源的导入）、有损格式拒绝、删除与 409（含"答完全部题目后即可删除"的回归）、
  题目参数校验、**防泄漏**、`reveal` 只在作答后返回且只含本题、首答即最终与幂等重放、安慰剂两路内容相同、
  带封面源的端到端出题回归、结果统计（p 值/显著/样本不足/`status` 枚举）与结论文案、
  **静态资源强制重验证**（`test_static_assets_are_revalidated`）、
  **试听音量滑条接线**（`test_frontend_volume_slider_is_wired`：`gain → volume → destination`、
  `/api/start` 报 `target_lufs`）、
  **「清理刺激缓存」的作用域**（`test_clear_cache_only_touches_stimuli_cache`：只清 `data/cache`）、
  启动清理、TTL 清理、路径穿越。
- `pytest.ini`：`tmp_path_retention_policy = failed` —— 测试产物落在项目内 `data/tmp/`，
  合成音频一次就有几百 MB，默认策略会留下最近几轮（实测堆到 907 MB），改成"只留失败那一轮"。
- 测试产物全部落在 `data/tmp/`（受 `TMPDIR` 约束），不写系统 temp 或用户目录。

## 15. 实测数据与实现记录（ffmpeg 6.1.1-3ubuntu5 / Python 3.12.15）

| 项 | 实测值 |
|---|---|
| 15 s @48 kHz | 720000 采样 = 恰好 625 个 MPEG-1 Layer III 帧 |
| MP3 编码到**文件**（`-write_xing 1` 生效） | 解码回来仍是 720000 采样，lag = 0 |
| MP3 编码到**管道**（无 LAME 头） | 解码得 721152 采样，lag = 1105（23.0 ms）；对齐器可兜住，结果仍 720000 / ΔLUFS = 0 |
| 每对刺激 | ΔLUFS = 0.0000 LU（32/64/128/320 kbps 均如此）；TARGET_LUFS = -18 → True Peak ≈ -5.3 dBTP；共同衰减 delta = 0（距 -1 dBTP 上限还有 4.3 dB） |
| 响度目标对比 | -23 LUFS（EBU R128）时真峰值 ≈ -10.3 dBTP，余量 9.3 dB —— 白白浪费动态余量、日常试听偏轻；-18 后余量 4.3 dB |
| 单个刺激文件 | 5760088 字节（5.76 MB + WAV 头），A/B 完全等长 |
| 缓存体积 | 单段跑满 11 档 ≈ 130 MB（实测一次旧 detailed 后释放 130271006 字节）；新 detailed（20 真实 + 2 安慰剂、2 个片段全用上）实测 **245.8 MB**/首 |
| `/api/start` 首次生成 10 对刺激 | 7.8 s；命中缓存后 < 0.2 s |
| `/api/start` 首次 detailed（22 题、20 对真实刺激 + 2 安慰剂） | **15.8 s**（`generated: 22, reused: 0`）；再开一轮 `generated: 0, reused: 22`，**0.01 s** |
| 带内嵌封面的源（复现 rc=234） | 未加 `-map 0:a:0` 时 `ffmpeg` 返回 rc=234 + `Attached pictures were requested, but the ID3v2 header is disabled`；加上后三步全部 rc=0 |
| 统计口径实测 | `POWER_REQUIRED_N = 7`；20/20 的精确 p = 2.9e-10（`p<0.001`）；22 题一轮的安慰剂位置实测出现过 [0, 17]（即被正确打散） |
| 依赖 | 全部 wheel（soundfile 自带 libsndfile），无需编译器或系统库 |

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

- **状态**：功能完整可用。`make test` **61 项**全绿（约 1 分钟）。
- **提交历史**：`git log --oneline` —— 脚手架 → 后端 + 管线测试 → 前端 + API 测试 + 文档 →
  会话/作答两个修复 → 封面流修复、乱序与揭晓、统计口径与 22 题 → 上传前的许可证与可移植性整理 →
  揭晓静默失败修复 + 结果页改版 → 清理目标与临时目录治理 → 响度提到 −18 LUFS + 试听音量滑条。
- **关键实测**：每对 ΔLUFS = 0.0000 LU、真峰值 ≈ −5.3 dBTP（−18 LUFS 下）、两路各 5760088 字节（无体积泄漏）、
  22 题详细模式首轮 15.8 s / 缓存复用 0.01 s、刺激缓存 245.8 MB/首、`POWER_REQUIRED_N = 7`。
  完整表见 §15。
- **待人工验证**：验收 #2（浏览器实操）与 #7 的主观听感；其余 15 条已自动化（对照表见 §9）。
- **可一行调整的行为**（都在 `app.py` 顶部）：`TARGET_LUFS`（改完要 bump `STIMULI_ALGO_VERSION`）、
  `SHORT_TEST_TOTAL` / `DETAILED_TEST_TOTAL` 与两条码率阶梯、`SHUFFLE_ALL_QUESTIONS`、
  `PLACEBO_SAME_SEGMENT`、`SEG_WEIGHTS`、`CHANCE_LEVEL` / `ALPHA` / `MIN_N_FOR_SIGNIFICANCE`；
  前端是否自动揭晓答案在 `static/app.js` 的 `renderConfirmation()`（默认只显示"已记录"，答案需手动点）；
  试听音量范围/步长在 `static/index.html` 的 `#volume-input` 与 `app.js` 的 `VOLUME_*` 常量。
  **改完 `static/` 记得把 `index.html` 里的 `?v=N` 加一。**
- **数据与清理**：`data/library/` 是用户的音频（持久，永不自动删除）；`data/cache/` 可随时重建；
  `data/session/`、`data/tmp/` 每次启动自动清空。清理命令见 §12（`make size` / `clean-pip-cache` /
  `clean-cache` / `clean`），面向用户的说明在 README「想清理磁盘占用怎么办」。
- **若日后要接着做**：优先考虑「提高检验功效」（`DETAILED_TEST_TOTAL` 调大或新增重复档位模式）、
  ABX 化（当前非目标）、以及给多点位/多版本对比留接口；改动刺激内容一律 bump `STIMULI_ALGO_VERSION`。
