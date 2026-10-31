"""Train YOLO11 สำหรับ Smart Helmet Detection บน GPU ในเครื่อง (ตั้งค่าสำหรับ RTX 5060 8 GB)

ใช้งาน:
    python training/train.py                     # yolo11n, 50 epochs, batch 16
    python training/train.py --model yolo11s.pt  # แม่นขึ้น ช้าลง
    python training/train.py --batch 8           # ถ้าเจอ CUDA out of memory
    python training/train.py --eval-only         # ประเมินโมเดลที่ train ไว้แล้ว + อัปเดตตารางเทียบ

Dataset: Helmet Detection_YOLOv8 v3 (CC BY 4.0)
https://universe.roboflow.com/learning-evidence/helmet-detection_yolov8
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# เก็บ dataset/ผล train นอก repo: ไม่ขึ้น git และไม่หายถ้าย้าย/ลบโฟลเดอร์โปรเจกต์
WORK = Path.home() / 'helmet-detection'
DATA_DIR = WORK / 'datasets' / 'helmet-detection_yolov8-3'
RUNS_DIR = WORK / 'runs'


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
    key = os.environ.get('ROBOFLOW_API_KEY') or input('Roboflow API key (วางแล้วกด Enter): ').strip()
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
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--patience', type=int, default=15, help='หยุดก่อนถ้าไม่ดีขึ้นติดกันกี่ epoch')
    ap.add_argument('--eval-only', action='store_true', help='ไม่ train ใหม่ แค่ประเมินโมเดลที่ train ไว้แล้ว')
    args = ap.parse_args()

    check_gpu()
    if sys.platform == 'win32':
        # กันเครื่อง idle-sleep ระหว่าง train (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) คืนค่าเองเมื่อ process จบ
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    WORK.mkdir(exist_ok=True)
    os.chdir(WORK)  # yolo11n.pt ที่ดาวน์โหลดจะลงที่นี่ด้วย
    data_yaml = download_dataset()

    import pandas as pd
    from ultralytics import YOLO
    name = Path(args.model).stem + ('' if args.imgsz == 640 else f'-{args.imgsz}')  # แยก run ไม่ทับของ 640
    run = RUNS_DIR / name
    start = time.time()
    if not args.eval_only:
        YOLO(args.model).train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=0,
            workers=4,
            project=str(RUNS_DIR),
            name=name,
            exist_ok=True,
        )
    train_min = (time.time() - start) / 60

    best = YOLO(run / 'weights' / 'best.pt')
    m = best.val(data=str(data_yaml), split='test', imgsz=args.imgsz, project=str(RUNS_DIR), name=f'{name}-test', exist_ok=True)
    print('\n=== ผลบน test set (ใส่รายงาน) ===')
    print(f'mAP50     : {m.box.map50:.3f}')
    print(f'mAP50-95  : {m.box.map:.3f}')
    print(f'Precision : {m.box.mp:.3f}')
    print(f'Recall    : {m.box.mr:.3f}')
    for i, c in enumerate(m.box.ap_class_index):
        print(f'{best.names[int(c)]:15s} AP50 = {m.box.ap50[i]:.3f}')

    hist = pd.read_csv(run / 'results.csv')
    hist.columns = [c.strip() for c in hist.columns]
    f1 = m.box.f1_curve.mean(0)
    row = {
        'model': name,
        'epochs_run': len(hist),
        'best_epoch': int(hist.loc[hist['metrics/mAP50-95(B)'].idxmax(), 'epoch']),
        'train_min': None if args.eval_only else round(train_min, 1),
        'test_mAP50': round(m.box.map50, 3),
        'test_mAP50-95': round(m.box.map, 3),
        'test_precision': round(m.box.mp, 3),
        'test_recall': round(m.box.mr, 3),
        'best_f1': round(float(f1.max()), 3),
        'best_f1_conf': round(float(m.box.px[int(f1.argmax())]), 3),
        'gpu_ms_per_img': round(m.speed['inference'], 1),
        'size_mb': round((run / 'weights' / 'best.pt').stat().st_size / 1e6, 1),
    }
    summary = WORK / 'results_summary.csv'
    df = pd.read_csv(summary) if summary.exists() else pd.DataFrame(columns=list(row))
    df = pd.concat([df[df['model'] != name], pd.DataFrame([row])], ignore_index=True)
    df.to_csv(summary, index=False)
    print(f'\nตารางเทียบโมเดล ({summary}):')
    print(df.to_string(index=False))

    # เก็บแยกต่อโมเดล ไม่เขียนทับกัน
    shutil.copy(run / 'weights' / 'best.pt', WORK / f'best_{name}.pt')
    print(f'\nโมเดล: {WORK / f"best_{name}.pt"}')
    print(f'กราฟ (results.png, confusion_matrix.png, F1 curve): {run}')


if __name__ == '__main__':  # จำเป็นบน Windows ไม่งั้น dataloader workers จะ error
    main()
