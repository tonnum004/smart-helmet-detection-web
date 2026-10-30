"""Smart Helmet Detection — Streamlit app (YOLO11n, best.pt)

รันในเครื่อง:  streamlit run streamlit_app/app.py
"""
import os
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent.parent / 'backend' / 'best.pt'
CLASSES = ['With Helmet', 'Without Helmet']
COLORS = {'With Helmet': (74, 163, 22), 'Without Helmet': (38, 38, 220)}  # BGR
MIN_CONF = 0.1        # ตรวจรูปที่ค่าต่ำแล้วค่อยกรองด้วย slider จะได้ไม่ต้องรันโมเดลใหม่
MAX_IMAGE_MB = 20
MAX_VIDEO_MB = 100
MAX_VIDEO_S = 60
SAMPLE_FPS = 5        # บน CPU ตรวจ ~5 เฟรม/วินาที แล้ววาดกรอบล่าสุดค้างไว้ในเฟรมที่ข้าม

st.set_page_config(page_title='Smart Helmet Detection', page_icon='🪖', layout='wide')


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
        cv2.rectangle(frame, (x1, ly), (x1 + tw + 2 * t, ly + th + 2 * t), color, -1)
        cv2.putText(frame, label, (x1 + t, ly + th + t), cv2.FONT_HERSHEY_SIMPLEX, scale,
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


def summary(items, count_label, speed_label, speed_value):
    counts = Counter(d['cls'] for d in items)
    n = len(items)
    rate = round(counts['With Helmet'] / n * 100) if n else 0
    c = st.columns(4)
    c[0].metric(count_label, n)
    c[1].metric('ไม่สวมหมวก', counts['Without Helmet'])
    c[2].metric('อัตราการสวมหมวก', f'{rate}%')
    c[3].metric(speed_label, speed_value)
    return counts


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
        fps = 30  # บางไฟล์ (เช่น WebM) รายงาน fps ผิด ใช้ timestamp ของเฟรมแทน
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

st.title('🪖 Smart Helmet Detection')
st.caption('ระบบตรวจจับการสวมหมวกกันน็อคจากรูปภาพและวิดีโอ · Image Processing and Computer Vision')

with st.sidebar:
    mode = st.radio('ประเภทข้อมูล', ['รูปภาพ', 'วิดีโอ'], horizontal=True)
    with st.expander('ตั้งค่าขั้นสูง'):
        conf = st.slider('Confidence threshold', 0.05, 0.95, 0.5, 0.05)
    st.divider()
    st.caption('โมเดล: YOLO11n · mAP50 0.912 (test set 149 รูป)\n\nคลาส: With Helmet, Without Helmet')

if mode == 'รูปภาพ':
    file = st.file_uploader('อัปโหลดรูป', type=['jpg', 'jpeg', 'png', 'webp'])
    if not file:
        st.info('อัปโหลดรูปเพื่อเริ่มตรวจจับ')
        st.stop()
    if file.size > MAX_IMAGE_MB * 1024 * 1024:
        st.error(f'ไฟล์ใหญ่เกิน {MAX_IMAGE_MB} MB')
        st.stop()
    img, dets, ms = detect_image(file.getvalue())
    if img is None:
        st.error('อ่านไฟล์รูปไม่ได้')
        st.stop()
    items = [d for d in dets if d['conf'] >= conf]
    result = draw(img.copy(), items)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), width='stretch')
    ok, png = cv2.imencode('.png', result)
    st.download_button('ดาวน์โหลดผลลัพธ์ (PNG)', png.tobytes(), f'{Path(file.name).stem}_result.png', 'image/png')

    st.subheader('ผลการตรวจจับ')
    counts = summary(items, 'ผู้ขับขี่ที่ตรวจพบ', 'เวลาประมวลผล (ms)', ms)
    left, right = st.columns([1, 1.4])
    left.markdown('**จำนวนต่อคลาส**')
    left.bar_chart(pd.DataFrame({'จำนวน': [counts[c] for c in CLASSES]}, index=CLASSES))
    right.markdown('**รายการที่ตรวจพบ**')
    right.dataframe(pd.DataFrame([{
        'คลาส': d['cls'], 'Confidence': round(d['conf'], 3),
        'x, y (px)': f"{d['xyxy'][0]}, {d['xyxy'][1]}",
        'ขนาด (px)': f"{d['xyxy'][2] - d['xyxy'][0]} × {d['xyxy'][3] - d['xyxy'][1]}",
    } for d in sorted(items, key=lambda d: -d['conf'])]), hide_index=True, width='stretch')

else:
    file = st.file_uploader('อัปโหลดวิดีโอ', type=['mp4', 'mov', 'webm', 'avi', 'mkv'])
    if not file:
        st.info(f'อัปโหลดวิดีโอ (ไม่เกิน {MAX_VIDEO_MB} MB และ {MAX_VIDEO_S} วินาที) เพื่อเริ่มตรวจจับ')
        st.stop()
    if file.size > MAX_VIDEO_MB * 1024 * 1024:
        st.error(f'ไฟล์ใหญ่เกิน {MAX_VIDEO_MB} MB')
        st.stop()

    key = (file.file_id, conf)
    if st.session_state.get('video_key') != key:
        preview = st.empty()
        with preview.container():
            st.video(file)
            if not st.button('เริ่มประมวลผล', type='primary'):
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
            st.error(str(e))
            st.stop()
        finally:
            os.remove(path)
        progress.empty()

    res = st.session_state.video_result
    st.video(res['video'])
    st.download_button('ดาวน์โหลดวิดีโอผลลัพธ์', res['video'], f'{Path(file.name).stem}_result.mp4', 'video/mp4')

    st.subheader('ผลการตรวจจับ')
    counts = summary(res['tracks'], 'ผู้ขับขี่ที่นับได้', 'ความเร็ว (FPS)', f"{res['fps']:.1f}")
    left, right = st.columns([1, 1.4])
    left.markdown('**จำนวนต่อคลาส**')
    left.bar_chart(pd.DataFrame({'จำนวน': [counts[c] for c in CLASSES]}, index=CLASSES))
    right.markdown('**จำนวนคนในเฟรมตามเวลา (วินาที)**')
    right.bar_chart(res['timeline'])
    st.markdown('**รายการที่ตรวจพบ**')
    st.dataframe(pd.DataFrame([{
        'Track ID': f"#{t['id']}", 'คลาส': t['cls'], 'Confidence': round(t['conf'], 3),
        'พบครั้งแรก': fmt_time(t['t0']), 'พบครั้งสุดท้าย': fmt_time(t['t1']),
    } for t in sorted(res['tracks'], key=lambda t: t['t0'])]), hide_index=True, width='stretch')
