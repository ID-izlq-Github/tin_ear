'use strict';

/* TinEar 前端：库管理 / 盲测 / 结果三视图 + Web Audio A/B 播放。
   注意：测试视图里绝不显示文件名、码率、编码器、时长与体积。 */

const $ = (id) => document.getElementById(id);

const state = {
  items: [],
  selected: new Set(),
  sessionId: null,
  mode: 'short',
  total: 0,
  index: 0,
  question: null,
  questionShownAt: 0,
  answered: false,
  lastAnswer: null,
  bufferCache: new Map(),
  busy: false,
  sampleRateWarned: false,
  cacheBytes: 0,
  tmpBytes: 0,
};

const audio = {
  ctx: null,
  gain: null,
  volume: null,
  source: null,
  buffers: { a: null, b: null },
  side: null,
  playing: false,
  startedAt: 0,
  offset: 0,
  duration: 0,
};

const FADE = 0.005; // 5 ms 淡入淡出
const VOLUME_FLOOR_DB = -40; // 滑条到底 = 静音
const VOLUME_STEP_DB = 2; // 键盘 ±2 dB
const PROGRESS_POLL_MS = 350; // 准备进度轮询间隔
const PROGRESS_SHOW_DELAY_MS = 300; // 命中缓存时几毫秒就结束了，进度条不要闪一下

/* 试听音量：监听级增益节点，A/B 共用同一个节点，所以不可能只把一边调响。
   测试音频本身已归一到后端常量 TARGET_LUFS（见 /api/start 的 target_lufs），
   所以默认 0 dB 就是"文件原样播放"。 */
function volumeDb() {
  const slider = $('volume-input');
  return slider ? Number(slider.value) : 0;
}

function applyVolume() {
  const db = volumeDb();
  const label = db <= VOLUME_FLOOR_DB ? '静音' : `${db} dB`;
  if ($('volume-label')) $('volume-label').textContent = label;
  if (audio.volume) {
    audio.volume.gain.value = db <= VOLUME_FLOOR_DB ? 0 : Math.pow(10, db / 20);
  }
}

function nudgeVolume(deltaDb) {
  const slider = $('volume-input');
  if (!slider) return;
  const next = Math.min(0, Math.max(VOLUME_FLOOR_DB, Number(slider.value) + deltaDb));
  slider.value = String(next);
  applyVolume();
}

/* ------------------------------------------------------------------ 工具 */

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  if (text) {
    try { data = JSON.parse(text); } catch (error) { data = null; }
  }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : `${response.status} ${response.statusText}`;
    throw new Error(detail);
  }
  return data;
}

function setStatus(id, message, kind = '') {
  const element = $(id);
  element.textContent = message || '';
  element.className = 'status' + (kind ? ' ' + kind : '');
}

function showBanner(message) {
  const banner = $('banner');
  banner.textContent = message;
  banner.classList.remove('hidden');
}

function showView(name) {
  for (const view of ['library', 'test', 'result']) {
    $('view-' + view).classList.toggle('hidden', view !== name);
  }
}

function formatDuration(seconds) {
  const total = Math.round(seconds || 0);
  const minutes = Math.floor(total / 60);
  return `${minutes}:${String(total % 60).padStart(2, '0')}`;
}

/* 准备进度：token 由前端生成，随 /api/start 一起提交；生成期间轮询 /api/progress。
   命中缓存时几毫秒就结束 —— 所以进度条延迟 PROGRESS_SHOW_DELAY_MS 再显示，避免闪一下。 */
function makeProgressToken() {
  if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
  return `p${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function progressText(entry) {
  if (!entry || !entry.found) return '正在准备测试音频…';
  const done = entry.done || 0;
  const total = entry.total || 0;
  if (entry.phase === 'segments') {
    return total > 1 ? `正在分析并切分片段（${done}/${total} 段素材）…` : '正在分析并切分片段…';
  }
  return total ? `正在生成测试音频（${done}/${total}）…` : '正在准备测试音频…';
}

function showPrepareProgress(done, total) {
  const wrap = $('prepare-progress');
  const bar = $('prepare-progress-bar');
  if (!wrap || !bar) return;
  wrap.classList.remove('hidden');
  bar.style.width = `${total > 0 ? Math.round((done / total) * 100) : 0}%`;
}

function hidePrepareProgress() {
  const wrap = $('prepare-progress');
  const bar = $('prepare-progress-bar');
  if (wrap) wrap.classList.add('hidden');
  if (bar) bar.style.width = '0%';
}

function formatSize(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function pct(value) {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`;
}

function ciText(row) {
  if (!row || !row.n || row.n < 3) return '—';
  return `${Math.round(row.ci_low * 100)}%–${Math.round(row.ci_high * 100)}%`;
}

function pText(row) {
  if (!row || row.p_value === null || row.p_value === undefined) return '—';
  return row.p_value < 0.001 ? '<0.001' : row.p_value.toFixed(3);
}

function buildTable(table, headers, rows) {
  table.textContent = '';
  const thead = document.createElement('thead');
  const headRow = document.createElement('tr');
  headers.forEach((label) => {
    const th = document.createElement('th');
    th.textContent = label;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  const tbody = document.createElement('tbody');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    row.forEach((value, index) => {
      const td = document.createElement('td');
      if (index > 0) td.className = 'num';
      if (value !== null && typeof value === 'object') {
        // {badge: {kind, label}, text, sub} —— 判定列用徽章 + 一行小字
        if (value.badge) {
          const badge = document.createElement('span');
          badge.className = 'badge ' + value.badge.kind;
          badge.textContent = value.badge.label;
          td.appendChild(badge);
        } else if (value.text !== undefined) {
          td.textContent = String(value.text);
        }
        if (value.sub) {
          const sub = document.createElement('div');
          sub.className = 'sub';
          sub.textContent = value.sub;
          td.appendChild(sub);
        }
      } else {
        td.textContent = String(value === null || value === undefined ? '—' : value);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
}

/* 统计判定 → 视觉分级（与后端 _row_stats 的 status 枚举一一对应） */
const STATUS_INFO = {
  significant: { kind: 'ok', label: '显著高于 1/3' },
  not_significant: { kind: 'muted', label: '无显著差异' },
  insufficient: { kind: 'warn', label: '样本不足' },
};

function statusOf(row) {
  return STATUS_INFO[(row && row.status) || 'insufficient'] || STATUS_INFO.insufficient;
}

function verdictCell(row) {
  const cell = { badge: statusOf(row) };
  if (row.n >= 3 && row.p_value !== null && row.p_value !== undefined) cell.sub = `p = ${pText(row)}`;
  return cell;
}

/* 结论卡：总体数字 + 分档徽章 + 安慰剂提示 + 样本量提示 */
function renderSummary(result) {
  const card = $('summary-card');
  const overall = result.overall || { n: 0, correct: 0, status: 'insufficient' };
  card.textContent = '';

  const head = document.createElement('div');
  head.className = 'summary-head';
  const big = document.createElement('span');
  big.className = 'summary-num';
  big.textContent = overall.n ? `${overall.correct}/${overall.n}` : '—';
  const rate = document.createElement('span');
  rate.className = 'summary-rate';
  rate.textContent = overall.accuracy === null || overall.accuracy === undefined
    ? ''
    : pct(overall.accuracy);
  const badge = document.createElement('span');
  const info = statusOf(overall);
  badge.className = 'badge ' + info.kind;
  badge.textContent = info.label;
  head.append(big, rate, badge);
  card.appendChild(head);

  const sub = document.createElement('div');
  sub.className = 'summary-sub';
  const bits = ['真实题总体'];
  if (overall.n && overall.n < 3) bits.push(`只有 ${overall.n} 题，任何结果都无法判定`);
  else if (overall.p_value !== null && overall.p_value !== undefined) bits.push(`p = ${pText(overall)}`);
  sub.textContent = bits.join(' · ');
  card.appendChild(sub);

  if (result.by_bracket && result.by_bracket.length) {
    const chips = document.createElement('div');
    chips.className = 'chips';
    for (const row of result.by_bracket) {
      const chip = document.createElement('div');
      chip.className = 'chip';
      const name = document.createElement('b');
      name.textContent = row.label;
      const count = document.createElement('span');
      count.className = 'chip-count';
      count.textContent = `${row.correct}/${row.n}`;
      const chipBadge = document.createElement('span');
      const chipInfo = statusOf(row);
      chipBadge.className = 'badge ' + chipInfo.kind;
      chipBadge.textContent = chipInfo.label;
      chip.append(name, count, chipBadge);
      chips.appendChild(chip);
    }
    card.appendChild(chips);
  }

  const placebo = result.placebo || { n: 0 };
  const placeboLine = document.createElement('div');
  placeboLine.className = 'summary-line' + (placebo.warning ? ' warn' : '');
  placeboLine.textContent = placebo.n
    ? `安慰剂题：${placebo.n} 题中 ${placebo.tie} 题选了「听不出」（${pct(placebo.tie_rate)}）。`
      + (placebo.warning
        ? '你在两路完全相同的情况下仍然"听出"了更好的一边 —— 前面的正确率要打折看。'
        : '这是好事：说明你基本不会凭空听出差异。')
    : '本次没有作答安慰剂题。';
  card.appendChild(placeboLine);

  const power = result.power || {};
  if (power.required_n) {
    const powerLine = document.createElement('div');
    powerLine.className = 'summary-line muted';
    powerLine.textContent = `样本量：每个码率最多 ${power.max_per_bitrate || 0} 题；`
      + `要让每档都达到 ${Math.round((power.target_power || 0.8) * 100)}% 检验功效`
      + `（假定真听出率 ${power.assumed_rate || 0.8}），约需每档 ${power.required_n} 题。`;
    card.appendChild(powerLine);
  }
}

/* -------------------------------------------------------- Web Audio 播放 */

function ensureAudioContext() {
  if (!audio.ctx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    audio.ctx = new Ctor({ sampleRate: 48000, latencyHint: 'playback' });
    // 链路：source → gain（5 ms 淡入淡出，0..1）→ volume（用户滑条，A/B 共用）→ 输出
    audio.gain = audio.ctx.createGain();
    audio.gain.gain.value = 0;
    audio.volume = audio.ctx.createGain();
    audio.gain.connect(audio.volume);
    audio.volume.connect(audio.ctx.destination);
    applyVolume();
  }
  if (Math.abs(audio.ctx.sampleRate - 48000) > 1 && !state.sampleRateWarned) {
    state.sampleRateWarned = true;
    showBanner(`注意：设备实际采样率是 ${audio.ctx.sampleRate} Hz，浏览器正在重采样，可能削弱差异的可听性。`);
  }
  if (audio.ctx.state === 'suspended') audio.ctx.resume().catch(() => {});
  return audio.ctx;
}

function currentPosition() {
  if (!audio.playing || !audio.ctx || !audio.duration) return audio.offset;
  const elapsed = Math.max(0, audio.ctx.currentTime - audio.startedAt);
  return (audio.offset + elapsed) % audio.duration;
}

function releaseSource(source) {
  if (!source) return;
  source.onended = null;
  window.setTimeout(() => { try { source.stop(); } catch (error) {} source.disconnect(); }, 80);
}

async function playSide(side) {
  const buffer = audio.buffers[side];
  if (!buffer) return;
  const ctx = ensureAudioContext();
  if (ctx.state === 'suspended') { try { await ctx.resume(); } catch (error) {} }

  const position = currentPosition();
  const now = ctx.currentTime;
  const gainParam = audio.gain.gain;
  const previous = audio.source;
  const switching = Boolean(previous);

  gainParam.cancelScheduledValues(now);
  gainParam.setValueAtTime(gainParam.value, now);
  const startAt = switching ? now + FADE : now;
  if (switching) {
    gainParam.linearRampToValueAtTime(0, now + FADE);
    gainParam.linearRampToValueAtTime(1, now + 2 * FADE);
    try { previous.stop(now + FADE + 0.001); } catch (error) {}
    releaseSource(previous);
  } else {
    gainParam.linearRampToValueAtTime(1, now + FADE);
  }

  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.loop = true;
  source.connect(audio.gain);
  source.start(startAt, position % buffer.duration);

  audio.source = source;
  audio.side = side;
  audio.playing = true;
  audio.startedAt = startAt;
  audio.offset = position % buffer.duration;
  audio.duration = buffer.duration;
  updatePlayUI();
}

function stopPlayback(resetPosition = true) {
  if (audio.ctx) {
    const now = audio.ctx.currentTime;
    const gainParam = audio.gain.gain;
    gainParam.cancelScheduledValues(now);
    gainParam.setValueAtTime(gainParam.value, now);
    gainParam.linearRampToValueAtTime(0, now + FADE);
  }
  const position = audio.playing ? currentPosition() : audio.offset;
  if (audio.source) {
    const source = audio.source;
    if (audio.ctx) { try { source.stop(audio.ctx.currentTime + FADE + 0.001); } catch (error) {} }
    releaseSource(source);
  }
  audio.source = null;
  audio.playing = false;
  audio.side = null;
  audio.offset = resetPosition ? 0 : position;
  updatePlayUI();
}

function toggleSide(side) {
  if (audio.playing && audio.side === side) stopPlayback(false);
  else playSide(side);
}

function updatePlayUI() {
  for (const side of ['a', 'b']) {
    const button = $('btn-play-' + side);
    const active = audio.playing && audio.side === side;
    button.classList.toggle('playing', active);
    button.querySelector('.state').textContent = active ? '播放中（点击暂停）' : '播放';
  }
}

/* ------------------------------------------------------------- 音频缓冲 */

async function getBuffer(url) {
  if (state.bufferCache.has(url)) return state.bufferCache.get(url);
  const response = await fetch(url);
  if (!response.ok) throw new Error('音频加载失败');
  const data = await response.arrayBuffer();
  ensureAudioContext();
  const buffer = await audio.ctx.decodeAudioData(data);
  state.bufferCache.set(url, buffer);
  return buffer;
}

async function prefetch(index) {
  if (!state.sessionId || index >= state.total) return;
  try {
    const question = await api(
      `/api/question?session_id=${encodeURIComponent(state.sessionId)}&index=${index}`
    );
    await Promise.all([getBuffer(question.url_a), getBuffer(question.url_b)]);
  } catch (error) {
    /* 预取失败不影响当前题目 */
  }
}

/* --------------------------------------------------------------- 音频库 */

async function loadLibrary() {
  const data = await api('/api/library');
  state.items = data.items || [];
  state.cacheBytes = data.cache_bytes || 0;
  state.tmpBytes = data.tmp_bytes || 0;
  for (const audioId of [...state.selected]) {
    if (!state.items.some((item) => item.audio_id === audioId)) state.selected.delete(audioId);
  }
  renderLibrary();
}

function renderLibrary() {
  const cacheBytes = state.cacheBytes || 0;
  const list = $('library-list');
  list.textContent = '';
  if (!state.items.length) {
    const empty = document.createElement('li');
    empty.className = 'empty';
    empty.textContent = '库还是空的，先导入一些无损音频吧。';
    list.appendChild(empty);
  }
  for (const item of state.items) {
    const row = document.createElement('li');
    row.classList.toggle('selected', state.selected.has(item.audio_id));

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = state.selected.has(item.audio_id);
    checkbox.addEventListener('change', () => toggleSelect(item.audio_id, checkbox.checked));

    const meta = document.createElement('div');
    meta.className = 'meta';
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = item.display_name;
    const detail = document.createElement('div');
    detail.className = 'detail';
    detail.textContent = `${formatDuration(item.duration)} · 源 ${item.source_sample_rate} Hz/${item.source_channels}ch · ${formatSize(item.size_bytes)}`;
    meta.append(name, detail);

    const remove = document.createElement('button');
    remove.className = 'btn ghost';
    remove.textContent = '删除';
    remove.addEventListener('click', () => deleteItem(item.audio_id, item.display_name));

    row.append(checkbox, meta, remove);
    list.appendChild(row);
  }
  $('selected-count').textContent = String(state.selected.size);
  $('btn-start').disabled = state.selected.size < 1;
  $('btn-clear-cache').textContent = `清理测试音频缓存（${formatSize(cacheBytes)}）`;
  $('btn-clear-cache').disabled = cacheBytes === 0;
  $('cache-info').textContent =
    `测试音频缓存 ${formatSize(cacheBytes)}（上面那个按钮释放的就是它；下次测试重新生成）`
    + ` · 临时文件 ${formatSize(state.tmpBytes || 0)}（上传/编码草稿，后端每次启动自动清空，不需要手动管）`;
}

function toggleSelect(audioId, checked) {
  if (checked && state.selected.size >= 3 && !state.selected.has(audioId)) {
    setStatus('upload-status', '最多只能选择 3 段素材。', 'error');
    renderLibrary();
    return;
  }
  if (checked) state.selected.add(audioId);
  else state.selected.delete(audioId);
  renderLibrary();
}

async function deleteItem(audioId, displayName) {
  if (!window.confirm(`删除「${displayName}」？对应的片段与测试音频缓存也会一起删除。`)) return;
  try {
    await api(`/api/library/${encodeURIComponent(audioId)}`, { method: 'DELETE' });
    state.selected.delete(audioId);
    setStatus('upload-status', `已删除「${displayName}」。`, 'ok');
    await loadLibrary();
  } catch (error) {
    setStatus('upload-status', `删除失败：${error.message}`, 'error');
  }
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  const form = new FormData();
  files.forEach((file) => form.append('files', file, file.name));
  setStatus('upload-status', `正在导入 ${files.length} 个文件…`);
  try {
    const data = await api('/api/library/upload', { method: 'POST', body: form });
    const lines = data.results.map((result) => {
      if (result.status === 'imported') return `${result.filename}：已导入`;
      if (result.status === 'duplicate') return `${result.filename}：重复，已跳过`;
      return `${result.filename}：被拒绝 — ${result.reason}`;
    });
    const hasError = data.results.some((result) => result.status === 'rejected');
    setStatus('upload-status', lines.join('\n'), hasError ? 'error' : 'ok');
    await loadLibrary();
  } catch (error) {
    setStatus('upload-status', `导入失败：${error.message}`, 'error');
  }
}

/* ---------------------------------------------------------------- 测试 */

/* 结束会话：释放 session 目录与后端引用，之后才允许删除库里的音频。
   beacon=true 用于页面关闭（sendBeacon 能在卸载过程中发出请求）。 */
async function endSession({ beacon = false } = {}) {
  const sessionId = state.sessionId;
  if (!sessionId) return;
  state.sessionId = null;
  const payload = JSON.stringify({ session_id: sessionId });
  if (beacon && navigator.sendBeacon) {
    navigator.sendBeacon('/api/end', new Blob([payload], { type: 'application/json' }));
    return;
  }
  try {
    await api('/api/end', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
    });
  } catch (error) {
    /* 会话可能已被 TTL 清理，忽略 */
  }
}

async function startTest() {
  if (state.busy || !state.selected.size) return;
  state.busy = true;
  ensureAudioContext();
  await endSession(); // 先收掉上一轮，避免留下"进行中"的引用
  const mode = document.querySelector('input[name="mode"]:checked').value;
  $('btn-start').disabled = true;
  hidePrepareProgress();
  setStatus('prepare-status', '正在准备测试音频：已有的片段直接复用，缺的才生成并做响度/对齐校准…');

  // 进度：POST 还在飞的时候轮询 /api/progress（冷启动要十几秒，暖缓存几毫秒）
  const progressToken = makeProgressToken();
  const startedAt = Date.now();
  let polling = true; // 结束后可能还有一次在飞的查询，不能让它把"就绪"提示覆盖回去
  const poller = window.setInterval(async () => {
    try {
      const entry = await api(`/api/progress?token=${encodeURIComponent(progressToken)}`);
      if (!polling) return;
      if (!entry || !entry.found) return;
      if (Date.now() - startedAt < PROGRESS_SHOW_DELAY_MS) return; // 太快就别显示进度条
      setStatus('prepare-status', progressText(entry));
      showPrepareProgress(entry.done || 0, entry.total || 0);
    } catch (error) {
      /* 进度查询失败不影响主流程，静默忽略 */
    }
  }, PROGRESS_POLL_MS);

  try {
    const data = await api('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        audio_ids: Array.from(state.selected),
        mode,
        progress_token: progressToken,
      }),
    });
    state.sessionId = data.session_id;
    state.total = data.total;
    state.mode = data.mode;
    state.bufferCache.clear();
    const stimuli = data.stimuli;
    // 详细模式会顺手预热"快速模式才用到的那几对"，如实报出来（不是本轮题目，但占了一点时间）
    const prewarmed = stimuli && stimuli.prewarmed
      ? `（顺手预热 ${stimuli.prewarmed} 段，下次切快速模式同样秒开）`
      : '';
    setStatus(
      'prepare-status',
      stimuli ? `测试音频就绪：新生成 ${stimuli.generated} 段，复用 ${stimuli.reused} 段。${prewarmed}` : '',
      'ok'
    );
    // 让用户知道文件本身已经归一到多少 LUFS（这样"要不要调滑条"心里有数）
    if ($('volume-hint') && data.target_lufs !== undefined) {
      $('volume-hint').textContent =
        `测试音频已等响归一化到 ${data.target_lufs} LUFS（A/B 同步）；滑条只影响试听响度，不影响判定`;
    }
    applyVolume();
    await loadQuestion(0);
  } catch (error) {
    setStatus('prepare-status', `准备失败：${error.message}`, 'error');
  } finally {
    polling = false;
    window.clearInterval(poller);
    hidePrepareProgress();
    state.busy = false;
    $('btn-start').disabled = state.selected.size < 1;
  }
}

function resetAnswerArea() {
  state.answered = false;
  state.lastAnswer = null;
  document.querySelectorAll('.answer').forEach((button) => {
    button.disabled = true; // 音频没载入完不让作答，避免污染 reaction_ms
    button.classList.remove('selected', 'dim');
  });
  $('confirm-area').classList.add('hidden');
  $('reveal-line').classList.add('hidden');
  $('reveal-line').textContent = '';
  $('recorded-line').textContent = '';
  $('btn-reveal').classList.remove('hidden');
  $('btn-next').textContent = '下一题 →';
}

async function loadQuestion(index) {
  const question = await api(
    `/api/question?session_id=${encodeURIComponent(state.sessionId)}&index=${index}`
  );
  stopPlayback(true);
  audio.buffers = { a: null, b: null };
  state.index = index;
  state.question = question;
  showView('test');
  resetAnswerArea();
  $('question-progress').textContent = `第 ${index + 1} / ${question.total} 题`;
  $('progress-bar').style.width = `${Math.round((index / question.total) * 100)}%`;
  setStatus('answer-status', '正在载入音频…');
  try {
    const [bufferA, bufferB] = await Promise.all([getBuffer(question.url_a), getBuffer(question.url_b)]);
    audio.buffers = { a: bufferA, b: bufferB };
    audio.offset = 0;
    audio.duration = bufferA.duration;
    setStatus('answer-status', '');
  } catch (error) {
    setStatus('answer-status', `音频载入失败：${error.message}`, 'error');
    return;
  }
  document.querySelectorAll('.answer').forEach((button) => { button.disabled = false; });
  state.questionShownAt = performance.now();
  updatePlayUI();
  prefetch(index + 1);
}

const CHOICE_LABELS = { a: 'A 更好', b: 'B 更好', tie: '听不出' };

async function submitAnswer(choice) {
  if (state.busy || state.answered || !state.question || !state.sessionId) return;
  const reaction = Math.max(0, Math.round(performance.now() - state.questionShownAt));
  setStatus('answer-status', '正在记录…');
  try {
    const data = await api('/api/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: state.sessionId,
        question_id: state.question.question_id,
        choice,
        reaction_ms: reaction,
      }),
    });
    stopPlayback(true);
    state.answered = true;
    state.lastAnswer = {
      choice,
      accepted: data.accepted !== false,
      reveal: data.reveal || null,
      done: data.done,
    };
    renderConfirmation();
  } catch (error) {
    setStatus('answer-status', `提交失败：${error.message}`, 'error');
  }
}

function renderConfirmation() {
  const answer = state.lastAnswer;
  document.querySelectorAll('.answer').forEach((button) => {
    const chosen = button.dataset.choice === answer.choice;
    button.classList.toggle('selected', chosen);
    button.classList.toggle('dim', !chosen);
  });
  if (answer.accepted) {
    $('recorded-line').textContent = `已记录：${CHOICE_LABELS[answer.choice] || answer.choice}`;
    $('btn-reveal').classList.remove('hidden');
  } else {
    $('recorded-line').textContent = '本题已作答（首答即最终），本次修改已忽略。';
    $('btn-reveal').classList.add('hidden');
  }
  $('btn-next').textContent = answer.done ? '查看结果 →' : '下一题 →';
  $('reveal-line').classList.add('hidden');
  $('reveal-line').textContent = '';
  $('confirm-area').classList.remove('hidden');
  setStatus('answer-status', '');
}

/* 可选：查看答案。默认不自动揭晓，避免对后续题产生反馈偏置。
   注意：这里任何情况下都要显示点东西 —— 之前旧脚本读不到新接口字段时抛异常，
   结果只把按钮藏起来了，页面上什么都没出现（很难排查）。 */
function revealAnswer() {
  const answer = state.lastAnswer;
  if (!answer) return;
  $('btn-reveal').classList.add('hidden');
  const line = $('reveal-line');
  const show = (text, kind = '') => {
    line.textContent = text;
    line.className = 'reveal' + (kind ? ' ' + kind : '');
    line.classList.remove('hidden');
  };

  const reveal = answer.reveal;
  if (!reveal || !reveal.sides || !reveal.sides.a || !reveal.sides.b) {
    show('没能取到本题的答案信息（通常是浏览器还缓存着旧版脚本）：请按 Ctrl+Shift+R 强制刷新后重试。', 'warn');
    return;
  }

  const sideLabel = (side) => (reveal.sides[side].kind === 'lossless'
    ? '无损'
    : `MP3 ${reveal.sides[side].bitrate} kbps`);
  if (reveal.placebo) {
    const same = reveal.same_segment ? '同一片段的两份拷贝' : '两个片段，都是无损';
    const verdict = answer.choice === 'tie' ? '✔ 你选了「听不出」' : '✘ 你在没有差异的两路之间"听出了"更好的一边';
    show(`A：无损 / B：无损（${same}）—— 这是安慰剂题，正确答案是「听不出」。${verdict}`);
    return;
  }
  const correct = answer.choice === reveal.correct_side;
  const verdict = correct
    ? '✔ 你选对了'
    : (answer.choice === 'tie' ? '✘ 这一题其实能听出来' : '✘ 你选错了');
  show(`A：${sideLabel('a')} / B：${sideLabel('b')} —— 无损在 ${reveal.correct_side.toUpperCase()}`
    + `（正确答案：${CHOICE_LABELS[reveal.correct_side]}）。${verdict}`);
}

async function advance() {
  if (!state.answered || state.busy || !state.lastAnswer) return;
  state.busy = true;
  const answer = state.lastAnswer;
  state.answered = false;
  state.lastAnswer = null;
  $('confirm-area').classList.add('hidden');
  try {
    if (answer.done) await showResult();
    else await loadQuestion(state.index + 1);
  } catch (error) {
    setStatus('answer-status', `无法进入下一题：${error.message}`, 'error');
  } finally {
    state.busy = false;
  }
}

async function showResult() {
  stopPlayback(true);
  const result = await api(`/api/result?session_id=${encodeURIComponent(state.sessionId)}`);

  renderSummary(result);

  buildTable(
    $('bracket-table'),
    ['码率区间', '题数', '答对', '正确率', '95% 区间', '判定'],
    (result.by_bracket || []).map((row) => [
      row.label, row.n, row.correct, pct(row.accuracy), ciText(row), verdictCell(row),
    ])
  );
  buildTable(
    $('bitrate-table'),
    ['码率 (kbps)', '题数', '答对', '正确率', '95% 区间', '判定', '平均反应'],
    (result.by_bitrate || []).map((row) => [
      row.bitrate,
      row.n,
      row.correct,
      pct(row.accuracy),
      ciText(row),
      verdictCell(row),
      row.mean_reaction_ms ? `${(row.mean_reaction_ms / 1000).toFixed(1)} s` : '—',
    ])
  );

  const chance = result.chance_level !== undefined
    ? `${Math.round(result.chance_level * 100)}%`
    : '33%';
  const needed = (result.power && result.power.required_n) || result.power_required_n || 7;
  $('stats-note').textContent = [
    `瞎猜基线是 ${chance}，不是 50%：三个选项里只有一个是"无损那侧"，闭着眼睛乱选也有约 ${chance} 的正确率。`,
    `判定用单侧精确二项检验（H0：正确率 = 瞎猜基线），p < ${result.alpha || 0.05} 才算显著高于瞎猜。`,
    '不足 3 题的行一律标「样本不足」—— n=1 时最小的 p 就是 1/3，结构上不可能显著。',
    '「95% 区间」是 Wilson 区间，只作描述性参考，样本小时会非常宽。',
    `要让每档都达到 80% 检验功效（假定真听出率 0.8），约需每档 ${needed} 题。`,
    '想要更硬的结论，就把 app.py 顶部的 DETAILED_TEST_TOTAL / SHORT_TEST_TOTAL 调大（码率阶梯会自动循环）。',
  ].join('\n');
  $('conclusion-text').textContent = result.conclusion || '';

  showView('result');
}

/* ------------------------------------------------------------- 事件绑定 */

function wireEvents() {
  $('file-input').addEventListener('change', async (event) => {
    await uploadFiles(event.target.files);
    event.target.value = '';
  });
  $('btn-refresh').addEventListener('click', () => loadLibrary().catch((error) => {
    setStatus('upload-status', `刷新失败：${error.message}`, 'error');
  }));
  $('btn-clear-cache').addEventListener('click', async () => {
    if (!window.confirm('清空所有片段与测试音频缓存？库里的音频不受影响，下次测试会重新生成。')) return;
    try {
      const data = await api('/api/cache', { method: 'DELETE' });
      setStatus('upload-status', `已清理测试音频缓存，释放 ${formatSize(data.freed_bytes)}。`, 'ok');
      await loadLibrary();
    } catch (error) {
      setStatus('upload-status', `清理失败：${error.message}`, 'error');
    }
  });
  $('btn-start').addEventListener('click', startTest);
  $('btn-retest').addEventListener('click', startTest);
  $('btn-back').addEventListener('click', async () => {
    stopPlayback(true);
    await endSession();
    showView('library');
    loadLibrary().catch(() => {});
  });

  $('btn-play-a').addEventListener('click', () => toggleSide('a'));
  $('btn-play-b').addEventListener('click', () => toggleSide('b'));
  $('volume-input').addEventListener('input', applyVolume);
  $('btn-reveal').addEventListener('click', revealAnswer);
  $('btn-next').addEventListener('click', advance);
  document.querySelectorAll('.answer').forEach((button) => {
    button.addEventListener('click', () => submitAnswer(button.dataset.choice));
  });

  window.addEventListener('pagehide', () => { endSession({ beacon: true }); });

  document.addEventListener('keydown', (event) => {
    if ($('view-test').classList.contains('hidden')) return;
    if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;
    const key = event.key.toLowerCase();
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') { // 试听音量 ±2 dB
      event.preventDefault();
      nudgeVolume(event.key === 'ArrowUp' ? VOLUME_STEP_DB : -VOLUME_STEP_DB);
      return;
    }
    if (state.answered) {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); advance(); }
      return;
    }
    if (key === '1') submitAnswer('a');
    else if (key === '2') submitAnswer('b');
    else if (key === '3') submitAnswer('tie');
    else if (event.key === ' ') { event.preventDefault(); toggleSide(audio.side === 'a' ? 'b' : 'a'); }
    else if (key === 'a') toggleSide('a');
    else if (key === 'b') toggleSide('b');
  });
}

(async function init() {
  wireEvents();
  try {
    await loadLibrary();
  } catch (error) {
    setStatus('upload-status', `无法连接后端：${error.message}`, 'error');
  }
})();
