# 🪖 Smart Helmet Detection

ระบบตรวจจับการสวมหมวกกันน็อคของผู้ขับขี่จากรูปภาพและวิดีโอ — วิชา Image Processing and Computer Vision

**แอป:** https://smart-helmet.streamlit.app

## ความสามารถ

- **รูปภาพ:** ตรวจจับหัวผู้ขับขี่ แยก `With Helmet` / `Without Helmet` วาดกรอบ สรุปจำนวน อัตราการสวมหมวก ตาราง และดาวน์โหลดรูปผลลัพธ์
- **วิดีโอ:** ตรวจ ~5 เฟรม/วินาที + ติดตามแต่ละคนด้วย ByteTrack (นับคนไม่ซ้ำ) กราฟจำนวนคนตามเวลา และดาวน์โหลดวิดีโอผลลัพธ์
- ปรับ Confidence threshold ได้ใน "ตั้งค่าขั้นสูง" (ค่าเริ่มต้น 0.3 = ค่าที่ F1 สูงสุดบน valid set)

## Dataset

[Helmet Detection_YOLOv8 v3](https://universe.roboflow.com/learning-evidence/helmet-detection_yolov8) (CC BY 4.0) — 2 คลาส `With Helmet`, `Without Helmet`

| ส่วน | จำนวนรูป |
|---|---|
| train | 3,648 (จากรูปต้นฉบับ 1,737 รูป + augmentation ×3 ของ Roboflow) |
| valid | 372 |
| test | 149 |

Preprocessing จาก Roboflow: Auto-Orient, Resize 640×640 · Augmentation: Flip, Crop, Grayscale, Hue/Saturation/Brightness, Blur

**ข้อมูล train เพิ่ม (โมเดลในแอป)** — 3 ชุดจาก Roboflow Universe (CC BY 4.0) ที่ภาพคล้ายชุดหลัก (ถนนเอเชีย ตีกรอบเฉพาะหัว) ใช้เป็น train อย่างเดียว valid/test ยังเป็นของชุดหลัก

| ชุด | รูปที่ใช้ | รูปที่ซ้ำกับชุดหลัก (ตัดออก) |
|---|---|---|
| [test-deesf/helmet_full](https://universe.roboflow.com/test-deesf/helmet_full) | 826 | 4,156 |
| [syas-project/helmet-detection-7ckfh](https://universe.roboflow.com/syas-project/helmet-detection-7ckfh) | 2,109 | 637 |
| [onlytusik/helmet-detection-fk6is](https://universe.roboflow.com/onlytusik/helmet-detection-fk6is) | 6,557 | 1,317 |

ตัดรูปซ้ำด้วย dHash และตัดรูปที่เป็นภาพหมุน/ครอป หรือฉากเดียวกับ valid/test (YOLO embedding + ORB) อีก 105 รูป กันข้อมูลทดสอบรั่วเข้า train → รวมรูปเพิ่ม 9,387 รูป

## โมเดลและผลประเมิน (test set 149 รูป, RTX 5060)

| โมเดล | epochs (best) | เวลา train | mAP50 | mAP50-95 | Precision | Recall | F1 สูงสุด | ms/รูป (GPU) | ขนาด |
|---|---|---|---|---|---|---|---|---|---|
| **yolo11n + ข้อมูลเพิ่ม** (ใช้ในแอป) | 49 (34) | 64 นาที | **0.933** | **0.617** | 0.853 | **0.881** | **0.873** | 3.5 | **5.5 MB** |
| yolo11n | 50 (37) | ~30 นาที | 0.912 | 0.596 | 0.852 | 0.879 | 0.870 | **3.2** | **5.5 MB** |
| yolo11s | 50 (30) | 28 นาที | 0.896 | 0.575 | 0.846 | 0.845 | 0.848 | 7.8 | 19.2 MB |
| yolo11m | 60 (42) | 70 นาที | 0.917 | 0.574 | **0.874** | 0.847 | 0.865 | 10.9 | 40.5 MB |

ชุดหลักอย่างเดียว: โมเดลใหญ่ไม่ได้ให้ผลดีกว่าอย่างมีนัยสำคัญ (train ครั้งเดียวต่อโมเดล) yolo11n ให้ mAP50-95 และ F1 สูงสุด และเร็วที่สุด เหมาะกับ Streamlit Cloud ที่รันบน CPU
เพิ่มข้อมูลที่ภาพคล้ายชุดหลักทำให้ yolo11n ดีขึ้นทุกค่า (mAP50 +0.021, mAP50-95 +0.021) จึงใช้ตัวนี้ในแอป

## รันแอปในเครื่อง

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Deploy: Streamlit Community Cloud → main file `streamlit_app/app.py` (push เข้า `main` แล้วแอปอัปเดตเอง)

## Train โมเดลใหม่ (GPU ในเครื่อง, ตั้งค่าสำหรับ RTX 5060 8 GB)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r training/requirements.txt
python training/train.py                         # yolo11n
python training/train.py --model yolo11s.pt      # โมเดลอื่น

# โมเดลในแอป: ดาวน์โหลด + รวมข้อมูลเพิ่ม แล้ว train
python training/merge_datasets.py --set similar
python training/train.py --data ~/helmet-detection/datasets/data_similar.yaml --name yolo11n-similar
```

- ครั้งแรกจะขอ Roboflow API key เพื่อดาวน์โหลด dataset
- ผลลัพธ์อยู่ที่ `~/helmet-detection/`: `best_<โมเดล>.pt`, กราฟใน `runs/<โมเดล>/`, ตารางเทียบ `results_summary.csv`
- ถ้าเจอ `CUDA out of memory` ใช้ `--batch 8`
- ใช้โมเดลใหม่ในแอป: copy `best_<โมเดล>.pt` ไปทับ `streamlit_app/best.pt`

## โครงสร้าง

```
streamlit_app/   แอป Streamlit + best.pt (yolo11n + ข้อมูลเพิ่ม)
training/        script สำหรับ train และประเมินผล
packages.txt     แพ็กเกจระบบสำหรับ Streamlit Cloud
```
