// แก้ชื่อโปรเจกต์และคลาสตรงนี้ให้ตรงกับโมเดลจริง
const CONFIG = {
  title: 'Smart Helmet Detection',
  subtitle: 'ระบบตรวจจับการสวมหมวกกันน็อคจากรูปภาพและวิดีโอ',
  classes: [
    { name: 'helmet', color: '#16a34a' },
    { name: 'no_helmet', color: '#dc2626' },
  ],
};

const MODES = {
  image: { accept: 'image/*', hint: 'JPG, PNG, WEBP' },
  video: { accept: 'video/*', hint: 'MP4, WEBM, MOV' },
};

const $ = (s) => document.querySelector(s);
const el = {
  file: $('#file'), drop: $('#drop'), dropHint: $('#drop-hint'),
  fileInfo: $('#file-info'), fileName: $('#file-name'), fileSize: $('#file-size'),
  error: $('#error'), conf: $('#conf'), confOut: $('#conf-out'),
  showBoxes: $('#show-boxes'), run: $('#run'), empty: $('#empty'), stage: $('#stage'),
  img: $('#img'), vid: $('#vid'), overlay: $('#overlay'), loading: $('#loading'),
  progress: $('#progress'), viewerBar: $('#viewer-bar'), viewerMeta: $('#viewer-meta'),
  download: $('#download'), results: $('#results'), cards: $('#cards'), bars: $('#bars'),
  timelineWrap: $('#timeline-wrap'), timeline: $('#timeline'), thead: $('#thead'), tbody: $('#tbody'),
};

const state = { mode: 'image', file: null, url: null, result: null, busy: false };

const classColor = Object.fromEntries(CONFIG.classes.map((c) => [c.name, c.color]));
const media = () => (state.mode === 'image' ? el.img : el.vid);
const threshold = () => Number(el.conf.value);
const rand = (a, b) => a + Math.random() * (b - a);
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.ceil(b / 1e3)} KB`);

/* ---------- mock model ---------- */

// shortcut: ผลลัพธ์จำลองทั้งหมด — ตอนเชื่อมโมเดลจริงให้แทนฟังก์ชันนี้ด้วย fetch ไปยัง API (เช่น FastAPI + YOLO11) แล้วคืนค่ารูปแบบเดียวกัน
async function detect(mode, mediaEl) {
  await fakeProgress(mode === 'image' ? 900 : 1800);
  const randomBox = () => {
    const w = rand(0.07, 0.28);
    const h = clamp(w * rand(0.6, 1.5), 0.06, 0.5);
    return { w, h, x: rand(0, 1 - w), y: rand(0, 1 - h) };
  };
  const cls = () => pick(CONFIG.classes).name;

  if (mode === 'image') {
    const dets = Array.from({ length: Math.round(rand(4, 11)) }, () => ({
      cls: cls(), conf: rand(0.25, 0.98), ...randomBox(),
    }));
    return { kind: 'image', dets, ms: Math.round(rand(35, 120)) };
  }

  const duration = Number.isFinite(mediaEl.duration) && mediaEl.duration > 0 ? mediaEl.duration : 10;
  const tracks = Array.from({ length: Math.round(rand(6, 16)) }, (_, i) => {
    const t0 = rand(0, duration * 0.85);
    return {
      id: i + 1, cls: cls(), conf: rand(0.25, 0.97), ...randomBox(),
      t0, t1: Math.min(duration, t0 + rand(1.5, Math.max(2, duration * 0.5))),
      vx: rand(-0.12, 0.12), vy: rand(-0.04, 0.04),
    };
  });
  return { kind: 'video', tracks, duration, fps: rand(18, 32) };
}

function fakeProgress(ms) {
  return new Promise((resolve) => {
    const start = Date.now();
    const id = setInterval(() => {
      const p = Math.min(1, (Date.now() - start) / ms);
      el.progress.style.width = `${p * 100}%`;
      if (p === 1) { clearInterval(id); resolve(); }
    }, 30);
  });
}

// ตำแหน่งกล่องของ track ณ เวลา t (null ถ้าไม่อยู่ในเฟรม)
function trackAt(tr, t) {
  if (t < tr.t0 || t > tr.t1) return null;
  const dt = t - tr.t0;
  return { ...tr, x: clamp(tr.x + tr.vx * dt, 0, 1 - tr.w), y: clamp(tr.y + tr.vy * dt, 0, 1 - tr.h) };
}

function visible() {
  const r = state.result;
  if (!r) return [];
  const th = threshold();
  if (r.kind === 'image') return r.dets.filter((d) => d.conf >= th);
  return r.tracks.filter((tr) => tr.conf >= th);
}

/* ---------- drawing ---------- */

// พื้นที่ที่ภาพแสดงจริงภายใน element (กันกรณี letterbox ของวิดีโอ)
function contentRect(m) {
  const iw = m.naturalWidth || m.videoWidth, ih = m.naturalHeight || m.videoHeight;
  const cw = m.clientWidth, ch = m.clientHeight;
  if (!iw || !ih) return { x: 0, y: 0, w: cw, h: ch };
  const s = Math.min(cw / iw, ch / ih);
  return { x: (cw - iw * s) / 2, y: (ch - ih * s) / 2, w: iw * s, h: ih * s };
}

function drawBoxes(ctx, rect, boxes, scale = 1) {
  ctx.lineWidth = 2 * scale;
  ctx.font = `500 ${12 * scale}px "IBM Plex Mono", monospace`;
  ctx.textBaseline = 'top';
  for (const b of boxes) {
    const color = classColor[b.cls] || '#fff';
    const x = rect.x + b.x * rect.w, y = rect.y + b.y * rect.h;
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, b.w * rect.w, b.h * rect.h);
    const text = `${b.id ? `#${b.id} ` : ''}${b.cls} ${b.conf.toFixed(2)}`;
    const pad = 4 * scale, th = 16 * scale;
    const tw = ctx.measureText(text).width + pad * 2;
    const ly = y - th < rect.y ? y : y - th;
    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw, th);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x + pad, ly + 2 * scale);
  }
}

function drawOverlay() {
  const m = media(), c = el.overlay;
  const dpr = window.devicePixelRatio || 1;
  const w = m.clientWidth, h = m.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
    c.style.width = `${w}px`;
    c.style.height = `${h}px`;
  }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!state.result || !el.showBoxes.checked) return;
  let boxes = visible();
  if (state.result.kind === 'video') boxes = boxes.map((tr) => trackAt(tr, el.vid.currentTime)).filter(Boolean);
  drawBoxes(ctx, contentRect(m), boxes);
}

/* ---------- results panel ---------- */

function card(label, value, unit = '') {
  return `<div class="card"><div class="label">${label}</div><div class="value">${value}<span class="unit">${unit}</span></div></div>`;
}

function renderResults() {
  const r = state.result;
  el.results.hidden = !r;
  el.download.disabled = !r;
  if (!r) return;
  const items = visible();

  const counts = Object.fromEntries(CONFIG.classes.map((c) => [c.name, 0]));
  items.forEach((d) => counts[d.cls]++);
  const rate = items.length ? Math.round((counts.helmet / items.length) * 100) : 0;
  el.cards.innerHTML = card(r.kind === 'image' ? 'ผู้ขับขี่ที่ตรวจพบ' : 'ผู้ขับขี่ที่นับได้', items.length)
    + card('ไม่สวมหมวก', counts.no_helmet) + card('อัตราการสวมหมวก', rate, '%')
    + (r.kind === 'image' ? card('เวลาประมวลผล', r.ms, 'ms') : card('ความเร็วประมวลผล', r.fps.toFixed(1), 'FPS'));

  const max = Math.max(1, ...Object.values(counts));
  el.bars.innerHTML = CONFIG.classes.map((c) => `
    <div class="bar-row">
      <span class="name"><span class="dot" style="background:${c.color}"></span>${c.name}</span>
      <span class="track"><span class="fill" style="width:${(counts[c.name] / max) * 100}%;background:${c.color}"></span></span>
      <span class="count">${counts[c.name]}</span>
    </div>`).join('');

  el.timelineWrap.hidden = r.kind !== 'video';
  if (r.kind === 'video') renderTimeline();

  if (r.kind === 'image') {
    const m = el.img, W = m.naturalWidth, H = m.naturalHeight;
    el.thead.innerHTML = '<tr><th>#</th><th>คลาส</th><th>Confidence</th><th>x, y (px)</th><th>ขนาด (px)</th></tr>';
    el.tbody.innerHTML = [...items].sort((a, b) => b.conf - a.conf).map((d, i) => `
      <tr><td class="mono">${i + 1}</td><td>${dotName(d.cls)}</td><td class="mono">${d.conf.toFixed(3)}</td>
      <td class="mono">${Math.round(d.x * W)}, ${Math.round(d.y * H)}</td>
      <td class="mono">${Math.round(d.w * W)} × ${Math.round(d.h * H)}</td></tr>`).join('');
  } else {
    el.thead.innerHTML = '<tr><th>Track ID</th><th>คลาส</th><th>Confidence</th><th>พบครั้งแรก</th><th>พบครั้งสุดท้าย</th></tr>';
    el.tbody.innerHTML = [...items].sort((a, b) => a.t0 - b.t0).map((t) => `
      <tr><td class="mono">#${t.id}</td><td>${dotName(t.cls)}</td><td class="mono">${t.conf.toFixed(3)}</td>
      <td class="mono">${fmtTime(t.t0)}</td><td class="mono">${fmtTime(t.t1)}</td></tr>`).join('');
  }
  if (!items.length) el.tbody.innerHTML = '<tr><td colspan="5" class="muted">ไม่พบวัตถุที่ความมั่นใจสูงกว่า threshold — ลองลดค่า Confidence</td></tr>';
}

const dotName = (cls) => `<span class="name" style="display:inline-flex;align-items:center;gap:6px"><span class="dot" style="background:${classColor[cls]}"></span>${cls}</span>`;

function buckets(r, n) {
  const size = r.duration / n, tracks = visible();
  return Array.from({ length: n }, (_, i) => {
    const t = (i + 0.5) * size;
    return { t, n: tracks.filter((tr) => t >= tr.t0 && t <= tr.t1).length };
  });
}

function renderTimeline() {
  const r = state.result, W = 600, H = 140, pad = 8;
  const bs = buckets(r, Math.min(60, Math.max(10, Math.ceil(r.duration))));
  const max = Math.max(1, ...bs.map((b) => b.n));
  const bw = W / bs.length;
  el.timeline.innerHTML = bs.map((b, i) => {
    const h = (b.n / max) * (H - pad * 2);
    return `<rect x="${i * bw + 1}" y="${H - pad - h}" width="${Math.max(1, bw - 2)}" height="${h}" rx="1.5" fill="#1f5eff" opacity=".75"><title>${fmtTime(b.t)} · ${b.n} วัตถุ</title></rect>`;
  }).join('') + `<line id="playhead" x1="0" x2="0" y1="0" y2="${H}" stroke="#15181d" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
}

function updatePlayhead() {
  const ph = el.timeline.querySelector('#playhead');
  if (!ph) return;
  const x = (el.vid.currentTime / state.result.duration) * 600;
  ph.setAttribute('x1', x);
  ph.setAttribute('x2', x);
}

/* ---------- input handling ---------- */

function showError(msg) {
  el.error.textContent = msg;
  el.error.hidden = !msg;
}

function clearFile() {
  if (state.url) URL.revokeObjectURL(state.url);
  Object.assign(state, { file: null, url: null, result: null });
  el.img.removeAttribute('src');
  el.vid.removeAttribute('src');
  el.vid.load();
  el.img.hidden = el.vid.hidden = el.stage.hidden = true;
  el.empty.hidden = false;
  el.fileInfo.hidden = el.viewerBar.hidden = true;
  el.file.value = '';
  el.run.disabled = true;
  el.run.textContent = 'เริ่มประมวลผล';
  renderResults();
  drawOverlay();
}

function setMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  document.querySelectorAll('.seg button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.mode === mode));
  el.file.accept = MODES[mode].accept;
  el.dropHint.textContent = MODES[mode].hint;
  showError('');
  clearFile();
}

function loadFile(file) {
  if (!file) return;
  const kind = file.type.split('/')[0];
  if (kind !== 'image' && kind !== 'video') return showError('รองรับเฉพาะไฟล์รูปภาพหรือวิดีโอเท่านั้น');
  setMode(kind);
  clearFile();
  showError('');
  state.file = file;
  state.url = URL.createObjectURL(file);
  const m = media();
  m.src = state.url;
  m.hidden = el.stage.hidden = false;
  el.empty.hidden = true;
  el.fileName.textContent = file.name;
  el.fileSize.textContent = fmtSize(file.size);
  el.fileInfo.hidden = false;
}

function onMediaReady() {
  if (!state.file) return;
  const m = media();
  const w = m.naturalWidth || m.videoWidth, h = m.naturalHeight || m.videoHeight;
  el.viewerMeta.textContent = `${w} × ${h}${state.mode === 'video' ? ` · ${fmtTime(m.duration || 0)}` : ''}`;
  el.viewerBar.hidden = false;
  el.download.disabled = true;
  el.run.disabled = false;
  drawOverlay();
}

function onMediaError() {
  if (!state.file) return;
  const name = state.file.name;
  clearFile();
  showError(`เปิดไฟล์ "${name}" ไม่ได้ — เบราว์เซอร์อาจไม่รองรับฟอร์แมตนี้${state.mode === 'video' ? ' (แนะนำ MP4/H.264)' : ''}`);
}

async function run() {
  if (!state.file || state.busy) return;
  state.busy = true;
  el.run.disabled = true;
  el.loading.hidden = false;
  if (state.mode === 'video') el.vid.pause();
  try {
    state.result = await detect(state.mode, media());
  } finally {
    state.busy = false;
    el.loading.hidden = true;
    el.run.disabled = !state.file;
    el.run.textContent = 'ประมวลผลใหม่';
  }
  renderResults();
  drawOverlay();
  if (state.mode === 'video') { el.vid.currentTime = 0; el.vid.play().catch(() => {}); }
}

function download() {
  const r = state.result;
  if (!r) return;
  const save = (blob, name) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  const base = state.file.name.replace(/\.[^.]+$/, '');
  if (r.kind === 'image') {
    const m = el.img, W = m.naturalWidth, H = m.naturalHeight;
    const c = Object.assign(document.createElement('canvas'), { width: W, height: H });
    const ctx = c.getContext('2d');
    ctx.drawImage(m, 0, 0);
    drawBoxes(ctx, { x: 0, y: 0, w: W, h: H }, visible(), Math.max(1, W / 900));
    c.toBlob((b) => save(b, `${base}_result.png`), 'image/png');
  } else {
    const rows = visible().map((t) => [t.id, t.cls, t.conf.toFixed(3), t.t0.toFixed(2), t.t1.toFixed(2)].join(','));
    save(new Blob([['track_id,class,confidence,first_seen_s,last_seen_s', ...rows].join('\n')], { type: 'text/csv' }), `${base}_tracks.csv`);
  }
}

/* ---------- wire up ---------- */

$('#title').textContent = CONFIG.title;
$('#subtitle').textContent = CONFIG.subtitle;
document.title = CONFIG.title;

document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
el.drop.addEventListener('click', () => el.file.click());
el.file.addEventListener('change', () => loadFile(el.file.files[0]));
el.drop.addEventListener('dragover', (e) => { e.preventDefault(); el.drop.classList.add('over'); });
el.drop.addEventListener('dragleave', () => el.drop.classList.remove('over'));
el.drop.addEventListener('drop', (e) => {
  e.preventDefault();
  el.drop.classList.remove('over');
  loadFile(e.dataTransfer.files[0]);
});
// วางไฟล์นอกกรอบ: กันเบราว์เซอร์เปิดไฟล์ทับหน้าเว็บ
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

$('#remove').addEventListener('click', () => { clearFile(); showError(''); });
el.img.addEventListener('load', onMediaReady);
el.vid.addEventListener('loadedmetadata', onMediaReady);
el.img.addEventListener('error', onMediaError);
el.vid.addEventListener('error', onMediaError);

el.conf.addEventListener('input', () => {
  el.confOut.textContent = threshold().toFixed(2);
  renderResults();
  drawOverlay();
});
el.showBoxes.addEventListener('change', drawOverlay);
el.run.addEventListener('click', run);
el.download.addEventListener('click', download);

el.timeline.addEventListener('click', (e) => {
  if (state.result?.kind !== 'video') return;
  const box = el.timeline.getBoundingClientRect();
  el.vid.currentTime = clamp((e.clientX - box.left) / box.width, 0, 1) * state.result.duration;
});

new ResizeObserver(drawOverlay).observe(el.stage);

(function tick() {
  if (state.result?.kind === 'video') { drawOverlay(); updatePlayhead(); }
  requestAnimationFrame(tick);
})();
