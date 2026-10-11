# TinEar 实测数据与实现记录

> 本文件是 [AGENTS.md](AGENTS.md) §15 的主体，面向**维护者**。表里每个数字都是本机实测值；
> **改音频管线、更新体积口径、或考古某个数字时看它**。之所以拆出来：这一节是查阅型资料，

> 而 AGENTS.md 每次会话都要整份载入，拆开后它从 62.1 KB 降到 57.2 KB（本轮再补 Windows `os.replace` 的记录后为 59.6 KB，原因详见 AGENTS.md §15）。
> 文中出现的 `§N`、`#N` 均指 [AGENTS.md](AGENTS.md) 的章节与验收条目编号。

## 15.1 实测数据

环境：ffmpeg 6.1.1-3ubuntu5 / Python 3.12.15。

| 项 | 实测值 |
|---|---|
| 15 s @48 kHz | 720000 采样 = 恰好 625 个 MPEG-1 Layer III 帧 |
| MP3 编码到**文件**（`-write_xing 1` 生效） | 解码回来仍是 720000 采样，lag = 0 |
| MP3 编码到**管道**（无 LAME 头） | 解码得 721152 采样，lag = 1105（23.0 ms）；对齐器可兜住，结果仍 720000 / ΔLUFS = 0 |
| 每个 (片段, 码率) 组合 | ΔLUFS = 0.0000 LU（32/64/128/320 kbps 均如此）；TARGET_LUFS = -18 → True Peak ≈ -5.3 dBTP；共同衰减 delta = 0（距 -1 dBTP 上限还有 4.3 dB） |
| 响度目标对比 | -23 LUFS（EBU R128）时真峰值 ≈ -10.3 dBTP，余量 9.3 dB —— 白白浪费动态余量、日常试听偏轻；-18 后余量 4.3 dB |
| 单个测试音频文件 | **PCM_24：4320044 字节**（4.32 MB；44 字节头 + 采样数据，整数位深无 `PEAK` chunk）；同内容 float32 为 5760088、PCM_16 为 2880044。两路总是完全等长 |
| 交付位深的量化误差 | PCM_24 实测「落盘样本 − 内存数组」的最大逐样本差 = **1.19e-7 = 恰好 1 LSB**（libsndfile 是截断而非四舍五入，容差因此取 1.5 LSB + 1e-6 兜底浮点）。≈ -138 dBFS，距 0.05 dB 的自检容差三个数量级；PCM_16 为 3.05e-5（≈ -101 dBFS） |
| 缓存体积（1 首 65 s 素材、detailed 含预热 1 对；含 `seg_*.flac` 与 meta.json 3.9 MB） | 三次实测的递进：**253.4 MB**（float32，20 对各自一份无损侧）→ **136.4 MB**（无损侧按内容共享，float32，−47.7%）→ **103.3 MB**（再改 PCM_24 交付，**合计 −59.2%**；23 个 WAV = 21 个压缩侧 + 2 份共享无损侧 + 0，安慰剂复用共享文件）。只跑快速模式 44.2 → **34.1 MB**；档位上限（11 档 × 2 片段 + 2 份共享无损侧 = 24 个 WAV）≈ **108 MB/首**。三次的冷启动/命中都在 4.3～4.8 s / 0.004～0.011 s，落盘自检只多读 2 个文件（+9 ms/题） |
| `/api/start` 首次生成 10 对测试音频（**串行时代**的旧数据，并发行见下） | 7.8 s；命中缓存后 < 0.2 s |
| `/api/start` 首次 detailed（22 题、20 对真实测试音频 + 2 安慰剂） | **15.8 s**（`generated: 22, reused: 0`）；再开一轮 `generated: 0, reused: 22`，**0.01 s** |
| 单题生成耗时拆解（24 核机器，65 s 合成源） | **746 ms/题**：`_prepare_pair`（响度+对齐，numpy）462 ms、ffmpeg 编码 163 ms、解码 83 ms、写 2 个 WAV 24 ms、读片段 FLAC 14 ms —— **瓶颈是我们自己的 numpy，不是 ffmpeg** |
| 并发生成加速比（同 22 题冷启动，`_prepare_session` 并发） | 1 路 15.32 s / 2 路 8.32 s（1.84×）/ 4 路 5.01 s（3.05×）/ 6 路 4.12 s（3.72×）/ **8 路 3.87 s（3.95×）** / 12 路 3.52 s（4.35×）—— 8 之后基本饱和，所以 `STIMULI_CONCURRENCY_MAX = 8` |
| 缓存命中（22 题全部复用，改造前） | 0.004 s（只读 index.json + 硬链接） |
| 带内嵌封面的源（复现 rc=234） | 未加 `-map 0:a:0` 时 `ffmpeg` 返回 rc=234 + `Attached pictures were requested, but the ID3v2 header is disabled`；加上后三步全部 rc=0 |
| 统计口径实测 | `POWER_REQUIRED_N = 7`；20/20 的精确 p = 2.9e-10（`p<0.001`）；22 题一轮的安慰剂位置实测出现过 [0, 17]（即被正确打散） |
| 随机 token 的"包含码率数字"概率 | 16 位 hex、15 个两位窗口 → 含某个指定的两位数概率 **5.74%**（实测 20000 次采样）；这就是那条防泄漏断言原来的假阳性率，已在测试里固定随机源修掉 |
| 并发冷启动端到端（真 uvicorn + HTTP，1 首 65 s 素材、detailed；**预热功能之前**的实测） | `generated: 22, reused: 0` 用时 **4.0 s**（进度 0→22 正常上报）；同素材第二轮 `generated: 0, reused: 22`；详细→快速补生成 **1** 个（224 kbps 只属快速阶梯）、快速→详细 `0 / 22`；缓存 245.5 MB/首 |
| 详细模式预热快速补集（同上素材，冷缓存） | detailed：`generated: 22, reused: 0, prewarmed: 1`，用时 **4.4 s**（进度总量 23）；紧接着 short：`generated: 0, reused: 6`，用时 **0.009 s**（预热前这一步是 0.7 s） |
| `python app.py` 的启动提示 | 必须走 **stderr + `flush=True`**：stdout 被重定向时是块缓冲，实测提示会一直压到进程退出才出现，被 `timeout` / Ctrl+C 打断时**根本看不到**。uvicorn 那行 `Uvicorn running on http://0.0.0.0:…` 是绑定地址，已用日志过滤器拦掉（过滤器装在 `Config()` 之后，免得被它的 `dictConfig` 覆盖） |
| 依赖 | 全部 wheel（soundfile 自带 libsndfile），无需编译器或系统库 |

> 一个**与交付位深有关的坑**：libsndfile 写 **float32** WAV 时会加一个 `PEAK` chunk，里面带 `timeStamp`
> （`time_t`，1 秒粒度）。实测同一段数组写两次：同秒内字节完全相同，**隔 1.2 s 写就只差那 4 个字节**
> （`data` chunk 与解码样本都相同）。所以历史结论是"不要比文件 sha256"。
> **交付改成 PCM_24 之后这条不再成立** —— 整数位深不写 `PEAK` chunk，文件恰好 = 44 字节头 + 采样数据，
> 隔 1.2 s 两次写入字节完全相同（实测）。测试仍然比**解码后的样本**：那是语义上该比的东西，
> 而且换成 FLOAT 交付时 `PEAK` 差异又会回来（`test_concurrent_stimuli_generation_is_identical_to_serial`）。

## 15.2 实现期间修掉的五个真问题

机制见 AGENTS.md §6 的两条 ⚠️ 与 §7 的「静态资源强制重验证」；每条都已加测试守护。

1. **`-ss` 位置**：输出 seek（`-i` 之后）会输出整段静音，必须用输入 seek。
2. **对齐精修符号**：精修阶段的平移方向写反，须与 `_shifted` 一致（`aligned[k] = cand[k + lag]`）。
3. **响度公式**：对两路施加同一个增益不会让响度差归零，必须各自归一后再施加共同的峰值余量衰减。
4. **内嵌封面流**：三条 ffmpeg 命令都要带 `-map 0:a:0`，否则 rc=234（只在最佳片段起点 = 0 时触发）。
5. **浏览器缓存旧 `app.js`**：接口改名后旧脚本抛异常，而它抛异常前先把「查看答案」按钮 `hidden` 了，
   于是用户只看到"点了没反应"。修法是强制重验证 + `?v=N` + 揭晓函数任何分支都显示东西。
   > 教训：凡是"点了按钮但什么都没发生"的交互，函数里都不能有会抛异常的隐式假设。

## 15.3 Windows 真机上的 `os.replace` 拒绝访问（WinError 5）

**现象**（作者在原生 Windows 上跑 `python app.py` 时实测到的，WSL 下从未出现）：

```
POST /api/start HTTP/1.1" 500 Internal Server Error
  File "...\app.py", line 960, in _ensure_stimuli
    await asyncio.to_thread(_write_wav_atomic, path_ll, out_ll)
  File "...\app.py", line 345, in _write_wav_atomic
    os.replace(tmp, path)
PermissionError: [WinError 5] 拒绝访问。:
  '...\data\cache\60c978a6f2c100fe\stimuli\v3\.seg1_ll.wav.2cd4c848fd69.tmp'
  -> '...\data\cache\60c978a6f2c100fe\stimuli\v3\seg1_ll.wav'
```

**根因**：`os.replace` 在 Windows 上走 `MoveFileEx(MOVEFILE_REPLACE_EXISTING)`，只要目标文件还有**别的句柄
开着**（哪怕只是只读）就抛 `WinError 5` / `WinError 32`；POSIX 的 `rename` 完全允许替换一个正被打开的文件，
所以这条路径在 WSL 下永远测不出来。触发它的是**并发生成**：`STIMULI_CONCURRENCY` 默认最多 8 路，而
`seg{S}_ll.wav` 是同一片段**所有码率共用**的一份（§6 的按内容共享），于是同一片段的好几个码率任务同时
写同一个路径 —— 一个任务刚写完就开始 `_verify_written_pair`（`sf.info` + `sf.read` **读**这个文件做落盘
自检），另一个任务正好在这时 `os.replace` 它 → 拒绝访问。文件名是 `seg1_ll.wav`（共享无损侧）而不是
`seg1_96k_mp3.wav`，正是这条共享路径的直接证据。

**修法（两层）**：

1. **根治**：`_STIMULUS_WRITE_LOCKS` + `_stimulus_write_lock(path)`，把「写文件 + 落盘自检」按**路径**串行；
   取锁顺序按路径字符串排序（两个路径一次按住，反向加锁会死锁）。这样"读自检"和"替换"根本不会重叠。
2. **兜底**：`_replace_with_retry` 对"被占用"做有限重试（`_REPLACE_RETRY_DELAYS` 合计约 1.6 s），
   覆盖本进程控制不了的外部句柄（浏览器的 `FileResponse`、杀毒/索引器扫描、资源管理器预览）；
   只对 `PermissionError` / `winerror ∈ {5, 32, 33}` 重试，真正的参数/权限错误一次就抛。
   顺带把 `_write_json`、导入落库两处 `os.replace` 也换成它 —— 同一类竞态（`_ensure_stimuli` 在锁外
   `_read_json(index.json)` 与别人替换它）在 Windows 上同样会 500。

**顺带修掉的小毛病**：`_write_wav_atomic` / `_write_json` 替换最终失败时会把 `<name>.<token>.tmp`
留在缓存目录里（旧代码没有 `finally` 清理）；现在失败即删。

**守护测试**（`tests/test_pipeline.py`，Linux 上也能跑）：
`test_write_wav_atomic_retries_windows_sharing_violation`（前两次 `PermissionError`、第三次成功）、
`test_write_wav_atomic_gives_up_and_cleans_tmp`、`test_replace_with_retry_does_not_swallow_real_errors`、
`test_shared_stimulus_write_and_verify_are_serialized` —— 最后一条是本根因的结构化守护：它把
`_verify_written_pair` 的读窗口人为拉长（`sleep(0.05)`），若"写"与"读自检"发生重叠就记一笔。
**实测把锁去掉后这条用例必然失败**（抓到重叠），加锁后恒为空 —— 所以它不是"在 Linux 上永远通过"的摆设。

**旧版本残留**：跑过旧代码的机器，`data/cache/<audio_id>/stimuli/v3/` 里可能留着
`.seg1_ll.wav.<token>.tmp`（每个 ≈ 4.3 MB，等于一条交付 WAV），直接删掉即可，
不影响缓存（缺的条目会自动重算）。

> 教训：`os.replace` / `os.rename` 不是跨平台等价的 —— 涉及"并发写同一路径 + 有人读它"的代码，
> 在 WSL/macOS 上永远绿，到 Windows 才炸。要么让读写不重叠，要么对"被占用"重试，最好两者都做。

> 复现/规避（给还不想升级代码的人）：`TINEAR_CONCURRENCY=1`（详见 §6 的 `_resolve_concurrency`）
> 会把并发生成压成串行，进程内不再出现"一边读自检一边替换"，这条报错随之消失（代价是冷启动回到 ~15 s）。

