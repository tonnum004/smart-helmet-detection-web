"""Train YOLO11 สำหรับ Smart Helmet Detection บน GPU ในเครื่อง (ตั้งค่าสำหรับ RTX 5060 8 GB)

ใช้งาน:
    python training/train.py                     # yolo11n, 50 epochs, batch 16
    python training/train.py --model yolo11s.pt  # แม่นขึ้น ช้าลง
    python training/train.py --batch 8           # ถ้าเจอ CUDA out of memory

Dataset: Helmet Detection_YOLOv8 v3 (CC BY 4.0)
https://universe.roboflow.com/learning-evidence/helmet-detection_yolov8
"""
import argparse
import os
import shutil
import sys
from getpass import getpass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / 'datasets' / 'helmet-detection_yolov8-3'
RUNS_DIR = HERE / 'runs'


def check_gpu():
    try:
        import torch
        ok = torch.cuda.is_available()
    except ImportError:
        ok = False
    if not ok:
        sys.exit(
            'ไม่เจอ GPU — ติดตั้ง PyTorch แบบ CUDA 12.8 ก่อน (RTX 50 series ต้องใช้ cu128 ขึ้นไป):\n'
            '  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128'
        )
    print(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}')


def download_dataset():
    if (DATA_DIR / 'data.yaml').exists():
        print(f'ใช้ dataset ที่มีอยู่แล้ว: {DATA_DIR}')
        return DATA_DIR / 'data.yaml'
    from roboflow import Roboflow
    # Private API Key จาก roboflow.com → Settings → API Keys (หรือตั้ง env ROBOFLOW_API_KEY)
    key = os.environ.get('ROBOFLOW_API_KEY') or getpass('Roboflow API key: ')
    project = Roboflow(api_key=key).workspace('learning-evidence').project('helmet-detection_yolov8')
    project.version(3).download('yolov11', location=str(DATA_DIR))
    return DATA_DIR / 'data.yaml'


def main():
    # Windows ที่ pipe output ใช้ cp1252 → พิมพ์ภาษาไทยไม่ได้
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yolo11n.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    check_gpu()
    data_yaml = download_dataset()

    from ultralytics import YOLO
    name = Path(args.model).stem
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        patience=15,      # หยุดก่อนถ้าไม่ดีขึ้น 15 epoch ติด
        device=0,
        workers=4,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
    )

    run = RUNS_DIR / name
    best = YOLO(run / 'weights' / 'best.pt')
    m = best.val(data=str(data_yaml), split='test', project=str(RUNS_DIR), name=f'{name}-test', exist_ok=True)
    print('\n=== ผลบน test set (ใส่รายงาน) ===')
    print(f'mAP50     : {m.box.map50:.3f}')
    print(f'mAP50-95  : {m.box.map:.3f}')
    print(f'Precision : {m.box.mp:.3f}')
    print(f'Recall    : {m.box.mr:.3f}')
    for i, c in enumerate(m.box.ap_class_index):
        print(f'{best.names[int(c)]:15s} AP50 = {m.box.ap50[i]:.3f}')

    shutil.copy(run / 'weights' / 'best.pt', HERE / 'best.pt')
    print(f'\nโมเดล: {HERE / "best.pt"}')
    print(f'กราฟ (results.png, confusion_matrix.png, F1 curve): {run}')


if __name__ == '__main__':  # จำเป็นบน Windows ไม่งั้น dataloader workers จะ error
    main()
