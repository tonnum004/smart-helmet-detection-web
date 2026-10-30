"""Smart Helmet Detection API — FastAPI + YOLO11 (best.pt)

POST /predict/image  -> {kind, dets: [{cls, conf, x, y, w, h}], ms}
POST /predict/video  -> {kind, tracks: [{id, cls, conf, t0, t1, boxes: [[t, x, y, w, h], ...]}], duration, fps}
พิกัดทั้งหมดเป็นสัดส่วน 0–1 ของภาพ (x, y = มุมซ้ายบน)
"""
import os
import tempfile
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

MODEL = YOLO(Path(__file__).with_name('best.pt'))
MIN_CONF = 0.1          # ส่งกล่องความมั่นใจต่ำมาด้วย ให้หน้าเว็บกรองเองด้วย slider
MAX_IMAGE_MB = 20
MAX_VIDEO_MB = 100
MAX_VIDEO_S = 120
VIDEO_SAMPLE_FPS = 10   # ตรวจประมาณ 10 เฟรม/วินาที พอสำหรับ tracking และเร็วพอบน CPU

# shortcut: โมเดลตัวเดียวรันแบบ blocking ทีละคำขอ พอสำหรับเดโม ถ้ามีผู้ใช้พร้อมกันหลายคนให้ย้ายไป threadpool/คิว
app = FastAPI(title='Smart Helmet Detection API')
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r'https://cv-web[a-z0-9-]*\.vercel\.app|http://(localhost|127\.0\.0\.1)(:\d+)?',
    allow_methods=['GET', 'POST'],
    allow_headers=['*'],
)


@app.get('/')
def health():
    return {'status': 'ok', 'classes': list(MODEL.names.values())}


async def read_upload(file: UploadFile, kind: str, max_mb: int) -> bytes:
    if not (file.content_type or '').startswith(kind + '/'):
        raise HTTPException(415, f'ต้องเป็นไฟล์ {kind}')
    data = await file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(413, f'ไฟล์ใหญ่เกิน {max_mb} MB')
    return data


def to_box(xyxyn):
    x1, y1, x2, y2 = (float(v) for v in xyxyn)
    return round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)


@app.post('/predict/image')
async def predict_image(file: UploadFile = File(...)):
    data = await read_upload(file, 'image', MAX_IMAGE_MB)
    try:
        img = Image.open(BytesIO(data)).convert('RGB')
    except UnidentifiedImageError:
        raise HTTPException(400, 'อ่านไฟล์รูปไม่ได้')
    start = time.perf_counter()
    r = MODEL.predict(img, conf=MIN_CONF, imgsz=640, verbose=False)[0]
    ms = round((time.perf_counter() - start) * 1000)
    dets = []
    for xyxyn, c, k in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        x, y, w, h = to_box(xyxyn)
        dets.append({'cls': MODEL.names[int(k)], 'conf': round(c, 4), 'x': x, 'y': y, 'w': w, 'h': h})
    return {'kind': 'image', 'dets': dets, 'ms': ms}


@app.post('/predict/video')
async def predict_video(file: UploadFile = File(...)):
    data = await read_upload(file, 'video', MAX_VIDEO_MB)
    fd, path = tempfile.mkstemp(suffix=Path(file.filename or 'v.mp4').suffix or '.mp4')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps <= 0 or frames <= 0:
            raise HTTPException(400, 'อ่านไฟล์วิดีโอไม่ได้ (แนะนำ MP4/H.264)')
        duration = frames / fps
        if duration > MAX_VIDEO_S:
            raise HTTPException(413, f'วิดีโอยาวเกิน {MAX_VIDEO_S} วินาที')
        stride = max(1, round(fps / VIDEO_SAMPLE_FPS))

        boxes, confs, classes = defaultdict(list), defaultdict(list), defaultdict(Counter)
        start, n = time.perf_counter(), 0
        results = MODEL.track(path, conf=MIN_CONF, imgsz=640, vid_stride=stride,
                              tracker='bytetrack.yaml', stream=True, verbose=False)
        for i, r in enumerate(results):
            n += 1
            if r.boxes.id is None:
                continue
            t = round(i * stride / fps, 3)
            for tid, xyxyn, c, k in zip(r.boxes.id.tolist(), r.boxes.xyxyn.tolist(),
                                        r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                tid = int(tid)
                boxes[tid].append([t, *to_box(xyxyn)])
                confs[tid].append(c)
                classes[tid][MODEL.names[int(k)]] += 1
        elapsed = time.perf_counter() - start
    finally:
        os.remove(path)

    tracks = [{
        'id': tid,
        'cls': classes[tid].most_common(1)[0][0],   # คลาสที่เจอบ่อยสุดของคนนี้
        'conf': round(sum(confs[tid]) / len(confs[tid]), 4),
        't0': b[0][0],
        't1': b[-1][0],
        'boxes': b,
    } for tid, b in boxes.items()]
    return {'kind': 'video', 'tracks': tracks, 'duration': round(duration, 3),
            'fps': round(n / elapsed, 1) if elapsed else 0}
