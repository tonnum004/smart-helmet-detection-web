# Smart Helmet Detection — Web UI

เว็บสำหรับตรวจจับการสวมหมวกกันน็อคจากรูปภาพและวิดีโอ (วิชา Image Processing and Computer Vision)

Live: https://cv-web-theta-eight.vercel.app

## สถานะ

- ตอนนี้ทำแค่ UX/UI ผลการตรวจจับเป็น **ข้อมูลจำลอง**
- ยังไม่ได้เชื่อมโมเดล: แทนฟังก์ชัน `detect()` ใน `app.js` ด้วยการเรียก API ของโมเดล (เช่น YOLO11) โดยคืนค่ารูปแบบเดิม
- ชื่อคลาสตั้งไว้ใน `CONFIG` บนสุดของ `app.js` (`With Helmet`, `Without Helmet`) ตรงกับ dataset [Helmet Detection_YOLOv8](https://universe.roboflow.com/learning-evidence/helmet-detection_yolov8)

## รันในเครื่อง

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
