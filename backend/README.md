# Smart Helmet Detection API

FastAPI + YOLO11n (`best.pt`) สำหรับเว็บ https://cv-web-theta-eight.vercel.app

- `GET /` — health check
- `POST /predict/image` — form field `file` (รูป ≤ 20 MB)
- `POST /predict/video` — form field `file` (วิดีโอ ≤ 100 MB, ≤ 120 วินาที) ตรวจ ~10 เฟรม/วินาที + ByteTrack

รันในเครื่อง:

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
```
