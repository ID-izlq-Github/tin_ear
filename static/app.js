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
};

const audio = {
  ctx: null,
  gain: null,
  source: null,
  buffers: { a: null, b: null },
  side: null,
  playing: false,
  startedAt: 0,
  offset: 0,
  duration: 0,
};

const FADE = 0.005; // 5 ms 淡入淡出

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
  if (!row || !row.n) return '—';
  return `${Math.round(row.ci_low * 100)}%–${Math.round(row.ci_high * 100)}%`;
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
      td.textContent = String(value);
      if (index > 0) td.className = 'num';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.append(thead, tbody);
}

/* -------------------------------------------------------- Web Audio 播放 */

function ensureAudioContext() {
  if (!audio.ctx) {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    audio.ctx = new Ctor({ sampleRate: 48000, latencyHint: 'playback' });
    audio.gain = audio.ctx.createGain();
    audio.gain.gain.value = 0;
    audio.gain.connect(audio.ctx.destination);
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
  $('btn-clear-cache').textContent = `清理刺激缓存（${formatSize(cacheBytes)}）`;
  $('btn-clear-cache').disabled = cacheBytes === 0;
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
  if (!window.confirm(`删除「${displayName}」？对应的片段与刺激缓存也会一起删除。`)) return;
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
  setStatus('prepare-status', '正在准备测试：截取片段、生成并校准压缩版（首次可能要十几秒）…');
  try {
    const data = await api('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_ids: Array.from(state.selected), mode }),
    });
    state.sessionId = data.session_id;
    state.total = data.total;
    state.mode = data.mode;
    state.bufferCache.clear();
    setStatus('prepare-status', '');
    await loadQuestion(0);
  } catch (error) {
    setStatus('prepare-status', `准备失败：${error.message}`, 'error');
  } finally {
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
    state.lastAnswer = { choice, correctSide: data.correct_side, done: data.done };
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
  $('recorded-line').textContent = `已记录：${CHOICE_LABELS[answer.choice] || answer.choice}`;
  $('btn-next').textContent = answer.done ? '查看结果 →' : '下一题 →';
  $('btn-reveal').classList.remove('hidden');
  $('reveal-line').classList.add('hidden');
  $('reveal-line').textContent = '';
  $('confirm-area').classList.remove('hidden');
  setStatus('answer-status', '');
}

/* 可选：查看答案。默认不自动揭晓，避免对后续题产生反馈偏置。 */
function revealAnswer() {
  const answer = state.lastAnswer;
  if (!answer) return;
  $('btn-reveal').classList.add('hidden');
  const text = answer.correctSide === null
    ? '本题 A 与 B 都是无损（安慰剂题），正确答案是「听不出」。'
    : `本题无损的是 ${answer.correctSide.toUpperCase()}，正确答案是「${CHOICE_LABELS[answer.correctSide]}」。`;
  $('reveal-line').textContent = text;
  $('reveal-line').classList.remove('hidden');
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
  $('conclusion').textContent = result.conclusion;

  buildTable(
    $('bracket-table'),
    ['码率区间', '题数', '答对', '正确率', '95% 置信区间'],
    result.by_bracket.map((row) => [row.label, row.n, row.correct, pct(row.accuracy), ciText(row)])
  );
  buildTable(
    $('bitrate-table'),
    ['码率 (kbps)', '题数', '答对', '正确率', '95% 置信区间', '平均反应', '备注'],
    result.by_bitrate.map((row) => [
      row.bitrate,
      row.n,
      row.correct,
      pct(row.accuracy),
      ciText(row),
      row.mean_reaction_ms ? `${(row.mean_reaction_ms / 1000).toFixed(1)} s` : '—',
      row.n <= 1 ? '单题，样本不足' : '',
    ])
  );

  const placebo = result.placebo;
  $('placebo-line').textContent = placebo.n
    ? `安慰剂题（A、B 均为无损）：共 ${placebo.n} 题，其中 ${placebo.tie} 题选了「听不出」（${pct(placebo.tie_rate)}）。若这一比例偏低，说明倾向于听到并不存在的差异。`
    : '本次没有安慰剂题。';

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
    if (!window.confirm('清空所有片段与刺激缓存？库里的音频不受影响，下次测试会重新生成。')) return;
    try {
      const data = await api('/api/cache', { method: 'DELETE' });
      setStatus('upload-status', `已清理刺激缓存，释放 ${formatSize(data.freed_bytes)}。`, 'ok');
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
