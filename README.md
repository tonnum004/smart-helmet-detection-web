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

## Train โมเดล

เปิด notebook ใน Colab: [training/train_helmet.ipynb](https://colab.research.google.com/github/tonnum004/smart-helmet-detection-web/blob/main/training/train_helmet.ipynb)
