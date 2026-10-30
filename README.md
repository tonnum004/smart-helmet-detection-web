# Smart Helmet Detection — Web UI

เว็บสำหรับตรวจจับการสวมหมวกกันน็อคจากรูปภาพและวิดีโอ (วิชา Image Processing and Computer Vision)

Live: https://cv-web-theta-eight.vercel.app

## แอป

- **Streamlit (หลัก):** `streamlit_app/app.py` ใช้โมเดล `backend/best.pt` (YOLO11n, mAP50 0.912 บน test set) ตรวจรูปและวิดีโอ (ByteTrack นับคน)
  - รันในเครื่อง: `pip install -r streamlit_app/requirements.txt` แล้ว `streamlit run streamlit_app/app.py`
  - Deploy: Streamlit Community Cloud → main file `streamlit_app/app.py`, Python 3.12
- **เว็บ HTML (Vercel):** เรียก API ใน `backend/` ต้องรัน `uvicorn app:app --port 7860` ในโฟลเดอร์ `backend` บนเครื่องเดียวกัน (หรือใส่ `?api=<url>`)
- Dataset: [Helmet Detection_YOLOv8](https://universe.roboflow.com/learning-evidence/helmet-detection_yolov8) (CC BY 4.0) คลาส `With Helmet`, `Without Helmet`

## รันเว็บ HTML ในเครื่อง

```bash
npx serve .
```

push เข้า `main` แล้ว Vercel จะ deploy ให้อัตโนมัติ

## Train โมเดล (GPU ในเครื่อง, ตั้งค่าสำหรับ RTX 5060 8 GB)

ติดตั้งครั้งแรก (RTX 50 series ต้องใช้ PyTorch แบบ CUDA 12.8):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r training/requirements.txt notebook
```

**แบบ Colab + GPU ในเครื่อง:** เปิด [training/train_helmet_local.ipynb](https://colab.research.google.com/github/tonnum004/smart-helmet-detection-web/blob/main/training/train_helmet_local.ipynb) ใน Colab แล้วทำตามขั้นตอนใน cell แรก (Connect to a local runtime)

**แบบ script:** (จะขอ Roboflow API key ตอนดาวน์โหลด dataset ครั้งแรก)

```bash
python training/train.py
```

ได้ `~/helmet-detection/best.pt` และกราฟใน `~/helmet-detection/runs/yolo11n/` · ถ้าเจอ `CUDA out of memory` ใช้ `--batch 8`
