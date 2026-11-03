"""Smart Helmet Detection — Streamlit app (YOLO11n, best.pt)

รันในเครื่อง:  streamlit run streamlit_app/app.py
"""
import os
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import altair as alt
import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

MODEL_PATH = Path(__file__).with_name('best.pt')
CLASSES = ['With Helmet', 'Without Helmet']
COLORS = {'With Helmet': (46, 108, 20), 'Without Helmet': (30, 38, 179)}  # BGR (#146c2e, #b3261e)
MIN_CONF = 0.1        # ตรวจรูปที่ค่าต่ำแล้วค่อยกรองด้วย slider จะได้ไม่ต้องรันโมเดลใหม่
MAX_IMAGE_MB = 20
MAX_VIDEO_MB = 100
MAX_VIDEO_S = 60
SAMPLE_FPS = 5        # บน CPU ตรวจ ~5 เฟรม/วินาที แล้ววาดกรอบล่าสุดค้างไว้ในเฟรมที่ข้าม

st.set_page_config(page_title='Smart Helmet Detection', page_icon=':material/sports_motorsports:', layout='wide')

# Material Design 3 (Google) — โทนสี/ฟอนต์/ไอคอน Material Symbols
# shortcut: CSS อิง data-testid ภายในของ Streamlit ถ้าอัปเดต Streamlit แล้วหน้าตาเพี้ยน ให้ตรวจ selector ตรงนี้
CSS = """
<style>
/* ไอคอน .ms ใช้ฟอนต์ Material Symbols Rounded ที่ Streamlit โหลดมาให้อยู่แล้ว ไม่ต้องดึงจาก Google Fonts ซ้ำ */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500&family=Roboto+Mono:wght@500&family=Roboto:wght@400;500&display=swap');
:root {
  --md-primary: #0b57d0; --md-on-primary: #fff; --md-primary-container: #d3e3fd; --md-on-primary-container: #041e49;
  --md-surface: #f8fafd; --md-surface-c: #ffffff; --md-surface-c-high: #f0f4f9;
  --md-on-surface: #1f1f1f; --md-on-surface-var: #444746; --md-outline: #c4c7c5; --md-outline-var: #e1e3e1;
  --md-error: #b3261e; --md-error-container: #f9dedc; --md-on-error-container: #410e0b;
  --md-success: #146c2e; --md-success-container: #c4eed0; --md-on-success-container: #072711;
  --md-warn-container: #ffefc9; --md-on-warn-container: #3d2e00;
}
.stApp :is(p, h1, h2, h3, h4, label, button, input, li, td, th, div[data-testid="stMarkdownContainer"]) {
  font-family: 'Roboto', 'Noto Sans Thai', system-ui, sans-serif;
}
[data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1120px; padding-top: 4.5rem; padding-bottom: 4rem; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stImage"] img { max-height: 70vh; object-fit: contain; border-radius: 16px; }
[data-testid="stVideo"] video { border-radius: 16px; }
/* กล่องอัปโหลด: ยังไม่มีไฟล์ = กล่องใหญ่ (ลากมาวาง/คลิกที่ไหนก็ได้), มีไฟล์แล้ว = แถบเล็กแสดงไฟล์ + ปุ่มลบ/เปลี่ยนของ Streamlit
   shortcut: อิง DOM ของ Streamlit 1.66 (instructions จะมีเฉพาะตอนยังไม่มีไฟล์) ถ้าอัปเดต Streamlit แล้วเพี้ยน ให้ตรวจ selector ตรงนี้ */
[data-testid="stFileUploaderDropzone"] { border-radius: 16px; border: 2px dashed #a8c7fa; background: var(--md-surface-c);
  cursor: pointer; transition: background .15s, border-color .15s; }
[data-testid="stFileUploaderDropzone"]:hover { background: #eef3fd; border-color: var(--md-primary); }
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileUploaderDropzoneInstructions"]) {
  flex-direction: column; justify-content: center; align-items: center; gap: 14px;
  min-height: 230px; padding: 30px 20px; text-align: center; border-radius: 24px; }
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileUploaderDropzoneInstructions"])::before {
  content: 'cloud_upload'; font-family: 'Material Symbols Rounded'; font-size: 56px; line-height: 1;
  color: var(--md-primary); width: 88px; height: 88px; border-radius: 50%; display: grid; place-items: center;
  background: var(--md-primary-container); font-feature-settings: 'liga'; -webkit-font-feature-settings: 'liga'; order: -2; }
[data-testid="stFileUploaderDropzoneInstructions"] { margin: 0; display: flex; flex-direction: column; align-items: center; order: -1; }
[data-testid="stFileUploaderDropzoneInstructions"] > * { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { display: block; font-size: 1.1rem; font-weight: 500;
  color: var(--md-on-surface); content: 'ลากไฟล์มาวางที่นี่ หรือคลิกเพื่อเลือกไฟล์'; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { display: block; margin-top: 4px; font-size: .88rem; color: var(--md-on-surface-var); }
.st-key-upl_image [data-testid="stFileUploaderDropzoneInstructions"]::after { content: 'รองรับ JPG, PNG, WEBP · ไม่เกิน 20 MB'; }
.st-key-upl_video [data-testid="stFileUploaderDropzoneInstructions"]::after { content: 'รองรับ MP4, MOV, WEBM, AVI, MKV · ไม่เกิน 100 MB และ 60 วินาที'; }
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileUploaderDropzoneInstructions"]) button {
  min-height: 44px; padding: 0 26px; border-radius: 999px; border: none; background: var(--md-primary); color: var(--md-on-primary); }
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileUploaderDropzoneInstructions"]) button:hover {
  background: #0842a0; color: var(--md-on-primary); }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] { border-radius: 999px; padding: 0 22px; min-height: 42px; font-weight: 500; }
[data-testid="stExpander"] details { border-radius: 16px; border-color: var(--md-outline-var); background: var(--md-surface-c); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }

.ms { font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr; display: inline-block;
  -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; -webkit-font-smoothing: antialiased;
  font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; font-size: 22px; vertical-align: middle; }
.ms.fill { font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24; }

.hero { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; padding: 22px 26px; margin-bottom: 8px;
  border-radius: 28px; background: var(--md-primary-container); color: var(--md-on-primary-container); }
.hero-icon { width: 60px; height: 60px; border-radius: 18px; background: var(--md-primary); color: var(--md-on-primary);
  display: grid; place-items: center; flex: none; }
.hero-icon .ms { font-size: 36px; }
.hero h1 { color: inherit; font-size: 1.6rem; font-weight: 500; margin: 0; padding: 0; letter-spacing: 0; }
.hero p { color: inherit; opacity: .8; margin: 2px 0 0; font-size: .95rem; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-left: auto; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 8px; font-size: .8rem;
  font-weight: 500; background: rgb(255 255 255 / .6); border: 1px solid rgb(4 30 73 / .12); white-space: nowrap; }
.chip .ms { font-size: 18px; }

.steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 8px 0 4px; }
.step { display: flex; gap: 14px; align-items: flex-start; padding: 16px 18px; border-radius: 16px;
  background: var(--md-surface-c); border: 1px solid var(--md-outline-var); }
.step .num { width: 40px; height: 40px; border-radius: 50%; flex: none; display: grid; place-items: center;
  background: var(--md-surface-c-high); color: var(--md-primary); }
.step b { display: block; font-weight: 500; color: var(--md-on-surface); }
.step small { color: var(--md-on-surface-var); }

.banner { display: flex; align-items: center; gap: 12px; padding: 14px 18px; border-radius: 16px; margin: 6px 0 10px; font-weight: 500; }
.banner .ms { font-size: 26px; }
.banner small { display: block; font-weight: 400; opacity: .85; }
.banner.error { background: var(--md-error-container); color: var(--md-on-error-container); }
.banner.success { background: var(--md-success-container); color: var(--md-on-success-container); }
.banner.info { background: var(--md-warn-container); color: var(--md-on-warn-container); }

.legend { display: flex; gap: 8px; flex-wrap: wrap; margin: 2px 0 8px; }
.legend .chip { background: var(--md-surface-c); border-color: var(--md-outline-var); color: var(--md-on-surface-var); }
.dot { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

.section { display: flex; align-items: center; gap: 8px; font-size: 1.15rem; font-weight: 500; color: var(--md-on-surface); margin: 18px 0 8px; }
.section .ms { color: var(--md-primary); }

.cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 4px 0 10px; }
.card { background: var(--md-surface-c); border: 1px solid var(--md-outline-var); border-radius: 16px; padding: 14px 16px; }
.card .head { display: flex; align-items: center; justify-content: space-between; color: var(--md-on-surface-var); font-size: .82rem; }
.card .head .ms { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-size: 20px;
  background: var(--md-surface-c-high); color: var(--md-primary); }
.card .value { font-family: 'Roboto Mono', monospace; font-size: 1.75rem; font-weight: 500; line-height: 1.25; margin-top: 6px; color: var(--md-on-surface); }
.card .unit { font-size: .85rem; color: var(--md-on-surface-var); margin-left: 4px; font-family: 'Roboto', 'Noto Sans Thai', sans-serif; }
.card.danger { background: var(--md-error-container); border-color: transparent; }
.card.danger .value, .card.danger .head { color: var(--md-on-error-container); }
.card.danger .head .ms { background: rgb(255 255 255 / .55); color: var(--md-error); }
.card.success { background: var(--md-success-container); border-color: transparent; }
.card.success .value, .card.success .head { color: var(--md-on-success-container); }
.card.success .head .ms { background: rgb(255 255 255 / .55); color: var(--md-success); }

.empty { text-align: center; color: var(--md-on-surface-var); padding: 26px 16px 22px; border-radius: 16px;
  background: var(--md-surface-c); border: 1px solid var(--md-outline-var); }
.empty .ms { font-size: 44px; color: var(--md-primary); display: block; margin: 0 auto 6px; }
.footer { text-align: center; color: var(--md-on-surface-var); font-size: .8rem; margin-top: 28px; }
@media (max-width: 760px) {
  .cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .steps { grid-template-columns: 1fr; }
  .chips { margin-left: 0; }
  .hero { padding: 18px; border-radius: 24px; }
}
</style>
"""


def icon(name, fill=False):
    return f'<span class="ms{" fill" if fill else ""}">{name}</span>'


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def draw(frame, dets):
    t = max(2, round(max(frame.shape[:2]) / 400))
    scale = t * 0.3
    for d in dets:
        x1, y1, x2, y2 = d['xyxy']
        color = COLORS[d['cls']]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, t)
        label = f"{'#%d ' % d['id'] if 'id' in d else ''}{d['cls']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t // 2))
        ly = y1 - th - 2 * t if y1 - th - 2 * t > 0 else y1
        lx = max(0, min(x1, frame.shape[1] - tw - 2 * t))  # ไม่ให้ป้ายล้นขอบขวา
        cv2.rectangle(frame, (lx, ly), (lx + tw + 2 * t, ly + th + 2 * t), color, -1)
        cv2.putText(frame, label, (lx + t, ly + th + t), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return frame


def to_dets(r, names):
    ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
    out = []
    for tid, b, c, k in zip(ids, r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        d = {'cls': names[int(k)], 'conf': float(c), 'xyxy': tuple(int(v) for v in b)}
        if tid is not None:
            d['id'] = tid
        out.append(d)
    return out


def summary(items, count_label, speed_label, speed_value, speed_unit, speed_icon):
    counts = Counter(d['cls'] for d in items)
    n = len(items)
    rate = round(counts['With Helmet'] / n * 100) if n else 0
    cards = [
        ('', 'groups', count_label, n, 'คน'),
        ('danger' if counts['Without Helmet'] else '', 'gpp_bad', 'ไม่สวมหมวก', counts['Without Helmet'], 'คน'),
        ('success' if n else '', 'verified_user', 'อัตราการสวมหมวก', rate, '%'),
        ('', speed_icon, speed_label, speed_value, speed_unit),
    ]
    st.html('<div class="cards">' + ''.join(
        f'<div class="card {tone}"><div class="head"><span>{label}</span>{icon(ic)}</div>'
        f'<div class="value">{value}<span class="unit">{unit}</span></div></div>'
        for tone, ic, label, value, unit in cards) + '</div>')
    return counts


def banner(counts, n):
    """สรุปผลแบบอ่านปุ๊บเข้าใจ: มีคนไม่สวมหมวก / ทุกคนสวม / ไม่พบใครเลย"""
    bad = counts['Without Helmet']
    if n == 0:
        tone, ic, title, sub = 'info', 'search_off', 'ไม่พบผู้ขับขี่ในไฟล์นี้', \
            'ลองใช้รูปที่เห็นศีรษะชัดขึ้น หรือลด Confidence threshold ใน "ตั้งค่าขั้นสูง"'
    elif bad:
        tone, ic, title, sub = 'error', 'warning', f'พบผู้ไม่สวมหมวกกันน็อค {bad} คน', \
            f'จากผู้ขับขี่ทั้งหมด {n} คน · กรอบสีแดงในภาพคือคนที่ไม่สวมหมวก'
    else:
        tone, ic, title, sub = 'success', 'check_circle', 'ทุกคนสวมหมวกกันน็อค', f'ตรวจพบผู้ขับขี่ {n} คน'
    st.html(f'<div class="banner {tone}">{icon(ic, fill=True)}<div>{title}<small>{sub}</small></div></div>')


def section(ic, title):
    st.html(f'<div class="section">{icon(ic)}{title}</div>')


LEGEND = ('<div class="legend">'
          '<span class="chip"><span class="dot" style="background:#146c2e"></span>With Helmet — สวมหมวก</span>'
          '<span class="chip"><span class="dot" style="background:#b3261e"></span>Without Helmet — ไม่สวมหมวก</span>'
          '</div>')


def class_chart(counts):
    df = pd.DataFrame({'คลาส': CLASSES, 'จำนวน': [counts[c] for c in CLASSES]})
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X('คลาส:N', title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('จำนวน:Q', title=None, axis=alt.Axis(tickMinStep=1)),
        color=alt.Color('คลาส:N', scale=alt.Scale(domain=CLASSES, range=['#146c2e', '#b3261e']), legend=None),
    ).properties(height=220)


@st.cache_data(show_spinner='กำลังตรวจจับ…', max_entries=20)
def detect_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, [], 0
    model = load_model()
    start = time.perf_counter()
    r = model.predict(img, conf=MIN_CONF, imgsz=640, verbose=False)[0]
    return img, to_dets(r, model.names), round((time.perf_counter() - start) * 1000)


def process_video(path, conf, progress):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if not 0 < fps <= 120:
        # shortcut: บางไฟล์ (เช่น WebM จากเบราว์เซอร์) รายงาน fps ผิด ใช้ 30 แทน ผลการนับถูก แต่วิดีโอผลลัพธ์อาจเล่นเร็ว/ช้ากว่าจริง
        fps = 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # บางไฟล์ไม่บอกจำนวนเฟรม (0 หรือติดลบ)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        raise ValueError('อ่านไฟล์วิดีโอไม่ได้ (แนะนำ MP4/H.264)')
    if total > 0 and total / fps > MAX_VIDEO_S:
        cap.release()
        raise ValueError(f'วิดีโอยาว {total / fps:.0f} วินาที เกินที่รองรับ ({MAX_VIDEO_S} วินาที)')

    # โมเดลใหม่ต่อวิดีโอ: tracker เก็บ state ไว้ในโมเดล ถ้าใช้ร่วมกันผู้ใช้หลายคนจะปนกัน
    model = YOLO(MODEL_PATH)
    raw = path + '.raw.mp4'
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    tracks = defaultdict(lambda: {'confs': [], 'cls': Counter(), 't0': None, 't1': None})
    per_second = Counter()
    current, i, n, t, next_t, start = [], 0, 0, 0.0, 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 or i / fps
            if t > MAX_VIDEO_S:
                raise ValueError(f'วิดีโอยาวเกินที่รองรับ ({MAX_VIDEO_S} วินาที)')
            if t >= next_t:  # ตรวจ ~SAMPLE_FPS เฟรม/วินาที
                next_t = t + 1 / SAMPLE_FPS
                r = model.track(frame, conf=conf, imgsz=640, persist=True, tracker='bytetrack.yaml', verbose=False)[0]
                n += 1
                current = [d for d in to_dets(r, model.names) if 'id' in d]
                for d in current:
                    tr = tracks[d['id']]
                    tr['confs'].append(d['conf'])
                    tr['cls'][d['cls']] += 1
                    tr['t0'] = t if tr['t0'] is None else tr['t0']
                    tr['t1'] = t
                per_second[int(t)] = max(per_second[int(t)], len(current))
            writer.write(draw(frame, current))
            i += 1
            if i % 15 == 0:
                if total > 0:
                    progress.progress(min(i / total, 1.0), text=f'กำลังประมวลผล… {i}/{total} เฟรม')
                else:
                    progress.progress(0.5, text=f'กำลังประมวลผล… {i} เฟรม')
    finally:
        cap.release()
        writer.release()
    elapsed = time.perf_counter() - start
    if i == 0:
        os.remove(raw)
        raise ValueError('อ่านไฟล์วิดีโอไม่ได้ (แนะนำ MP4/H.264)')

    # mp4v เล่นในเบราว์เซอร์ไม่ได้ แปลงเป็น H.264
    out = path + '.out.mp4'
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-i', raw,
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], check=True)
        video = Path(out).read_bytes()
    finally:
        for f in (raw, out):
            if os.path.exists(f):
                os.remove(f)

    rows = [{'id': tid, 'cls': tr['cls'].most_common(1)[0][0], 'conf': sum(tr['confs']) / len(tr['confs']),
             't0': tr['t0'], 't1': tr['t1']} for tid, tr in tracks.items()]
    timeline = pd.DataFrame({'จำนวนคน': [per_second[s] for s in range(int(t) + 1)]})
    return {'video': video, 'tracks': rows, 'timeline': timeline, 'fps': n / elapsed if elapsed else 0,
            'duration': t}


def fmt_time(s):
    return f'{int(s // 60)}:{int(s % 60):02d}'


# ---------------- UI ----------------

st.html(CSS)
st.html(f'<div class="hero"><div class="hero-icon">{icon("sports_motorsports", fill=True)}</div>'
        '<div><h1>Smart Helmet Detection</h1><p>ตรวจจับการสวมหมวกกันน็อคของผู้ขับขี่จากรูปภาพและวิดีโอ</p></div>'
        f'<div class="chips"><span class="chip">{icon("neurology")}YOLO11n</span>'
        f'<span class="chip">{icon("target")}mAP50 0.912</span></div></div>')

MODES = {'image': ':material/image: รูปภาพ', 'video': ':material/movie: วิดีโอ'}
c_mode, c_set = st.columns([1.3, 1], vertical_alignment='center')
mode = c_mode.segmented_control('ประเภทข้อมูล', list(MODES), default='image', format_func=MODES.get,
                                label_visibility='collapsed') or 'image'
with c_set.expander('ตั้งค่าขั้นสูง', icon=':material/tune:'):
    conf = st.slider('Confidence threshold', 0.05, 0.95, 0.3, 0.05,
                     help='0.3 = ค่าที่ F1 สูงสุดบน valid set · ค่าสูง: กรอบน้อยแต่แม่น · ค่าต่ำ: เจอครบแต่อาจผิดมากขึ้น')
CONF_COL = {'Confidence': st.column_config.ProgressColumn('Confidence', min_value=0.0, max_value=1.0, format='%.2f')}


def how_to(first_icon, first_title, first_sub):
    st.html('<div class="steps">' + ''.join(
        f'<div class="step"><div class="num">{icon(ic)}</div><div><b>{t}</b><small>{s}</small></div></div>'
        for ic, t, s in [(first_icon, first_title, first_sub),
                         ('image_search', '2. ระบบตรวจจับอัตโนมัติ', 'ตีกรอบเขียว = สวมหมวก · แดง = ไม่สวม'),
                         ('download', '3. ดูสรุปและดาวน์โหลด', 'จำนวนคน อัตราการสวมหมวก และไฟล์ผลลัพธ์')]) + '</div>')


if mode == 'image':
    with st.container(key='upl_image'):
        file = st.file_uploader('อัปโหลดรูป', type=['jpg', 'jpeg', 'png', 'webp'], label_visibility='collapsed')
    if not file:
        how_to('upload', '1. อัปโหลดรูป', f'JPG, PNG, WEBP · ไม่เกิน {MAX_IMAGE_MB} MB')
        st.stop()
    if file.size > MAX_IMAGE_MB * 1024 * 1024:
        st.error(f'ไฟล์ใหญ่เกิน {MAX_IMAGE_MB} MB', icon=':material/error:')
        st.stop()
    img, dets, ms = detect_image(file.getvalue())
    if img is None:
        st.error('อ่านไฟล์รูปไม่ได้ ลองใช้ไฟล์ JPG หรือ PNG', icon=':material/broken_image:')
        st.stop()
    items = [d for d in dets if d['conf'] >= conf]
    result = draw(img.copy(), items)
    counts = Counter(d['cls'] for d in items)

    banner(counts, len(items))
    st.html(LEGEND)
    tab_result, tab_orig = st.tabs([':material/image_search: ผลลัพธ์', ':material/image: ภาพต้นฉบับ'])
    tab_result.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), width='stretch')
    tab_orig.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width='stretch')
    ok, png = cv2.imencode('.png', result)
    st.download_button('ดาวน์โหลดรูปผลลัพธ์ (PNG)', png.tobytes(), f'{Path(file.name).stem}_result.png', 'image/png',
                       icon=':material/download:', type='primary')

    section('analytics', 'ผลการตรวจจับ')
    summary(items, 'ผู้ขับขี่ที่ตรวจพบ', 'เวลาประมวลผล', ms, 'ms', 'timer')
    left, right = st.columns([1, 1.4])
    with left.container(border=True):
        st.markdown('**:material/bar_chart: จำนวนต่อคลาส**')
        st.altair_chart(class_chart(counts), width='stretch')
    with right.container(border=True):
        st.markdown('**:material/list_alt: รายการที่ตรวจพบ**')
        if items:
            st.dataframe(pd.DataFrame([{
                'คลาส': d['cls'], 'Confidence': d['conf'],
                'x, y (px)': f"{d['xyxy'][0]}, {d['xyxy'][1]}",
                'ขนาด (px)': f"{d['xyxy'][2] - d['xyxy'][0]} × {d['xyxy'][3] - d['xyxy'][1]}",
            } for d in sorted(items, key=lambda d: -d['conf'])]), hide_index=True, width='stretch', column_config=CONF_COL)
        else:
            st.caption('ยังไม่มีรายการ')

else:
    with st.container(key='upl_video'):
        file = st.file_uploader('อัปโหลดวิดีโอ', type=['mp4', 'mov', 'webm', 'avi', 'mkv'], label_visibility='collapsed')
    if not file:
        how_to('upload', '1. อัปโหลดวิดีโอ', f'MP4, MOV, WEBM · ไม่เกิน {MAX_VIDEO_MB} MB และ {MAX_VIDEO_S} วินาที')
        st.stop()
    if file.size > MAX_VIDEO_MB * 1024 * 1024:
        st.error(f'ไฟล์ใหญ่เกิน {MAX_VIDEO_MB} MB', icon=':material/error:')
        st.stop()

    key = (file.file_id, conf)
    if st.session_state.get('video_key') != key:
        preview = st.empty()
        with preview.container():
            st.video(file)
            st.caption(':material/info: บน Cloud ใช้ CPU · วิดีโอ 1 นาทีอาจใช้เวลาหลายนาที')
            if not st.button('เริ่มประมวลผล', type='primary', icon=':material/play_arrow:', width='stretch'):
                st.stop()
        preview.empty()
        progress = st.progress(0.0, text='กำลังเริ่ม…')
        fd, path = tempfile.mkstemp(suffix=Path(file.name).suffix)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(file.getvalue())
            st.session_state.video_result = process_video(path, conf, progress)
            st.session_state.video_key = key
        except ValueError as e:
            st.error(str(e), icon=':material/error:')
            st.stop()
        finally:
            os.remove(path)
        progress.empty()

    res = st.session_state.video_result
    counts = Counter(t['cls'] for t in res['tracks'])
    banner(counts, len(res['tracks']))
    st.html(LEGEND)
    st.video(res['video'])
    st.download_button('ดาวน์โหลดวิดีโอผลลัพธ์ (MP4)', res['video'], f'{Path(file.name).stem}_result.mp4', 'video/mp4',
                       icon=':material/download:', type='primary')

    section('analytics', 'ผลการตรวจจับ')
    summary(res['tracks'], 'ผู้ขับขี่ที่นับได้', 'ความเร็วประมวลผล', f"{res['fps']:.1f}", 'FPS', 'speed')
    left, right = st.columns([1, 1.4])
    with left.container(border=True):
        st.markdown('**:material/bar_chart: จำนวนต่อคลาส**')
        st.altair_chart(class_chart(counts), width='stretch')
    with right.container(border=True):
        st.markdown('**:material/timeline: จำนวนคนในเฟรมตามเวลา**')
        tl = res['timeline'].rename_axis('วินาที').reset_index()
        st.altair_chart(alt.Chart(tl).mark_area(color='#0b57d0', opacity=0.2, line={'color': '#0b57d0'}).encode(
            x=alt.X('วินาที:Q', title='วินาที'), y=alt.Y('จำนวนคน:Q', title=None, axis=alt.Axis(tickMinStep=1)),
        ).properties(height=220), width='stretch')
    with st.container(border=True):
        st.markdown('**:material/list_alt: รายการที่ตรวจพบ**')
        if res['tracks']:
            st.dataframe(pd.DataFrame([{
                'Track ID': f"#{t['id']}", 'คลาส': t['cls'], 'Confidence': t['conf'],
                'พบครั้งแรก': fmt_time(t['t0']), 'พบครั้งสุดท้าย': fmt_time(t['t1']),
            } for t in sorted(res['tracks'], key=lambda t: t['t0'])]), hide_index=True, width='stretch', column_config=CONF_COL)
        else:
            st.caption('ยังไม่มีรายการ')

st.html('<div class="footer">Smart Helmet Detection · YOLO11n + ByteTrack · Image Processing and Computer Vision</div>')
