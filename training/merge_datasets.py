"""รวม dataset หมวกกันน็อคจาก Roboflow เพิ่มเข้ากับชุดหลัก แล้วสร้าง data_<set>.yaml

ใช้งาน:  python training/merge_datasets.py --set similar   (ขอ Roboflow API key ถ้ายังไม่ได้ดาวน์โหลด)
         --set merged  = 3 ชุดแรกที่ลอง (ภาพต่างจากชุดหลักมาก ผลแย่ลง)
         --set similar = 3 ชุดที่ภาพคล้ายชุดหลัก (ถนนเอเชีย ตีกรอบเฉพาะหัว)

- แปลงคลาสของทุกชุดเป็น 2 คลาส: 0 = With Helmet, 1 = Without Helmet (คลาสอื่นตัดทิ้ง)
- รูปจากชุดเพิ่มทั้งหมดใช้เป็น train เท่านั้น ส่วน valid/test ยังเป็นของชุดหลัก จะได้เทียบกับโมเดลเดิมตรงๆ
- ตัดรูปชุดเพิ่มที่ซ้ำ/เกือบซ้ำกับชุดหลักออก (valid/test: กันข้อมูลทดสอบรั่วเข้า train, train: ไม่ได้ข้อมูลใหม่)
"""
import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

WORK = Path.home() / 'helmet-detection'
MAIN = WORK / 'datasets' / 'helmet-detection_yolov8-3'
EXTRA_DIR = WORK / 'datasets' / 'extra'

SETS = {  # (workspace, project) บน Roboflow Universe
    'merged': [
        ('cs-174-paper-1', 'helmet-rider-detection-y30qv'),
        ('naratipws', 'motorcycle-helmet-w75eo'),
        ('innovatech', 'motorcycle-helmet-q0wmd'),
    ],
    'similar': [
        ('test-deesf', 'helmet_full'),
        ('syas-project', 'helmet-detection-7ckfh'),
        ('onlytusik', 'helmet-detection-fk6is'),
    ],
}
WITH = {'with helmet', 'helmet', 'full-faced', 'half-faced'}
WITHOUT = {'without helmet', 'no-helmet', 'no helmet', 'invalid'}
IGNORE = {'helmetrider', 'nohelmetrider', 'person'}  # กรอบทั้งตัวคน ไม่ใช่หัว
DUP_BITS = 6  # dHash ต่างกันไม่เกินกี่ bit (จาก 64) ถือว่าเป็นรูปเดียวกัน


def class_map(names):
    m = {}
    for i, n in enumerate(names):
        k = n.strip().lower()
        if k in WITH:
            m[i] = 0
        elif k in WITHOUT:
            m[i] = 1
        elif k in IGNORE:
            m[i] = None
        else:
            sys.exit(f'ไม่รู้จักคลาส "{n}" — เพิ่มลง WITH/WITHOUT/IGNORE ก่อน')
    return m


def dhash(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view('>u8')[0])


def popcount(x):
    return np.array([bin(int(v)).count('1') for v in x])


def drop_rotated_copies(out, held_paths):
    """dHash จับรูปที่ถูกหมุน/ครอปไม่ได้ (Roboflow augmentation) → หาคู่ด้วย embedding ของ YOLO แล้วยืนยันด้วย ORB + homography
    ตัดทั้งรูปเดียวกันและเฟรมอื่นจากฉาก/มุมกล้องเดียวกันกับ valid/test ออก"""
    import torch
    from ultralytics import YOLO
    model = YOLO(WORK / 'best_yolo11n.pt')

    def embed(paths):
        e = [v for i in range(0, len(paths), 64) for v in model.embed([str(p) for p in paths[i:i + 64]], imgsz=320, verbose=False)]
        return torch.nn.functional.normalize(torch.stack(e).float(), dim=1)

    orb, bf, cache = cv2.ORB_create(1000), cv2.BFMatcher(cv2.NORM_HAMMING), {}

    def feats(p):
        if p not in cache:
            cache[p] = orb.detectAndCompute(cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (480, 480)), None)
        return cache[p]

    def inliers(a, b):
        (ka, da), (kb, db) = feats(a), feats(b)
        if da is None or db is None or len(ka) < 10 or len(kb) < 10:
            return 0
        good = [m for m, n in (x for x in bf.knnMatch(da, db, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 10:
            return 0
        _, mask = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                                     np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 8.0)
        return 0 if mask is None else int(mask.sum())

    extra = sorted((out / 'images').glob('*'))
    top_v, top_i = (embed(extra) @ embed(held_paths).T).topk(3, dim=1)
    dropped = 0
    for i in (top_v[:, 0] > 0.6).nonzero().flatten().tolist():
        if max(inliers(extra[i], held_paths[j]) for j in top_i[i].tolist()) >= 25:
            extra[i].unlink()
            (out / 'labels' / (extra[i].stem + '.txt')).unlink()
            dropped += 1
    print(f'\nตัดรูปที่เป็นภาพหมุน/ครอป หรือฉากเดียวกับ valid/test ชุดหลัก: {dropped} รูป')


def download(sources):
    from roboflow import Roboflow
    todo = [s for s in sources if not (EXTRA_DIR / s[1] / 'data.yaml').exists()]
    if not todo:
        return
    key = os.environ.get('ROBOFLOW_API_KEY') or input('Roboflow API key (วางแล้วกด Enter): ').strip()
    rf = Roboflow(api_key=key)
    for ws, proj in todo:
        project = rf.workspace(ws).project(proj)
        version = max(project.versions(), key=lambda v: int(str(v.version).split('/')[-1]))
        print(f'ดาวน์โหลด {ws}/{proj} version {version.version}')
        version.download('yolov11', location=str(EXTRA_DIR / proj))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', choices=SETS, default='similar')
    set_name = ap.parse_args().set
    sources = SETS[set_name]
    out = WORK / 'datasets' / f'extra_{set_name}'
    yaml_path = WORK / 'datasets' / f'data_{set_name}.yaml'
    download(sources)

    main_imgs = [p for s in ('train', 'valid', 'test') for p in (MAIN / s / 'images').glob('*')]
    held = np.array([h for h in map(dhash, main_imgs) if h is not None], dtype=np.uint64)
    print(f'hash รูปชุดหลัก: {len(held)} รูป')

    if out.exists():
        shutil.rmtree(out)
    (out / 'images').mkdir(parents=True)
    (out / 'labels').mkdir(parents=True)
    seen = set()
    for ws, proj in sources:
        root = EXTRA_DIR / proj
        names = yaml.safe_load((root / 'data.yaml').read_text(encoding='utf-8'))['names']
        names = list(names.values()) if isinstance(names, dict) else names
        cmap = class_map(names)
        print(f'\n{proj}: ' + ', '.join(f'{names[i]}→{v}' for i, v in cmap.items()))
        stats = Counter()
        for split in ('train', 'valid', 'test'):
            for img in sorted((root / split / 'images').glob('*')):
                h = dhash(img)
                if h is None:
                    stats['อ่านไม่ได้'] += 1
                    continue
                if len(held) and popcount(held ^ np.uint64(h)).min() <= DUP_BITS:
                    stats['ซ้ำกับชุดหลัก'] += 1
                    continue
                if h in seen:
                    stats['ซ้ำกันเอง'] += 1
                    continue
                seen.add(h)
                lbl = root / split / 'labels' / (img.stem + '.txt')
                lines = []
                for line in (lbl.read_text().splitlines() if lbl.exists() else []):
                    parts = line.split()
                    if len(parts) != 5:  # ข้าม polygon/บรรทัดเสีย
                        continue
                    c = cmap.get(int(parts[0]))
                    if c is not None:
                        lines.append(' '.join([str(c)] + parts[1:]))
                        stats[('With Helmet', 'Without Helmet')[c]] += 1
                name = f'{proj}_{img.name}'
                shutil.copy(img, out / 'images' / name)
                (out / 'labels' / (Path(name).stem + '.txt')).write_text('\n'.join(lines))
                stats['รูปที่ใช้'] += 1
        print('  ' + ', '.join(f'{k}: {v}' for k, v in stats.items()))

    drop_rotated_copies(out, sorted(p for s in ('valid', 'test') for p in (MAIN / s / 'images').glob('*')))
    yaml_path.write_text(yaml.safe_dump({
        'path': str(WORK / 'datasets'),
        'train': [str(MAIN / 'train' / 'images'), str(out / 'images')],
        'val': str(MAIN / 'valid' / 'images'),
        'test': str(MAIN / 'test' / 'images'),
        'nc': 2,
        'names': ['With Helmet', 'Without Helmet'],
    }, allow_unicode=True), encoding='utf-8')
    print(f'\nรูปเพิ่มทั้งหมด: {len(list((out / "images").glob("*")))} · data yaml: {yaml_path}')


if __name__ == '__main__':
    main()
