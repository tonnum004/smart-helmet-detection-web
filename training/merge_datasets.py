"""รวม dataset หมวกกันน็อคจาก Roboflow เพิ่มเข้ากับชุดหลัก แล้วสร้าง data_merged.yaml

ใช้งาน:  python training/merge_datasets.py   (ขอ Roboflow API key ถ้ายังไม่ได้ดาวน์โหลด)

- แปลงคลาสของทุกชุดเป็น 2 คลาส: 0 = With Helmet, 1 = Without Helmet (คลาสอื่นตัดทิ้ง)
- รูปจากชุดเพิ่มทั้งหมดใช้เป็น train เท่านั้น ส่วน valid/test ยังเป็นของชุดหลัก จะได้เทียบกับโมเดลเดิมตรงๆ
- ตัดรูปชุดเพิ่มที่ซ้ำ/เกือบซ้ำกับ valid/test ของชุดหลักออก (กันข้อมูลทดสอบรั่วเข้า train)
"""
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
OUT = WORK / 'datasets' / 'extra_merged'
YAML = WORK / 'datasets' / 'data_merged.yaml'

SOURCES = [  # (workspace, project) บน Roboflow Universe
    ('cs-174-paper-1', 'helmet-rider-detection-y30qv'),
    ('naratipws', 'motorcycle-helmet-w75eo'),
    ('innovatech', 'motorcycle-helmet-q0wmd'),
]
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


def download():
    from roboflow import Roboflow
    todo = [s for s in SOURCES if not (EXTRA_DIR / s[1] / 'data.yaml').exists()]
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
    download()

    held_out = [p for s in ('valid', 'test') for p in (MAIN / s / 'images').glob('*')]
    held = np.array([h for h in map(dhash, held_out) if h is not None], dtype=np.uint64)
    print(f'hash รูป valid/test ชุดหลัก: {len(held)} รูป')

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / 'images').mkdir(parents=True)
    (OUT / 'labels').mkdir(parents=True)
    seen = set()
    for ws, proj in SOURCES:
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
                    stats['ซ้ำกับ valid/test ชุดหลัก'] += 1
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
                shutil.copy(img, OUT / 'images' / name)
                (OUT / 'labels' / (Path(name).stem + '.txt')).write_text('\n'.join(lines))
                stats['รูปที่ใช้'] += 1
        print('  ' + ', '.join(f'{k}: {v}' for k, v in stats.items()))

    YAML.write_text(yaml.safe_dump({
        'path': str(WORK / 'datasets'),
        'train': [str(MAIN / 'train' / 'images'), str(OUT / 'images')],
        'val': str(MAIN / 'valid' / 'images'),
        'test': str(MAIN / 'test' / 'images'),
        'nc': 2,
        'names': ['With Helmet', 'Without Helmet'],
    }, allow_unicode=True), encoding='utf-8')
    print(f'\nรูปเพิ่มทั้งหมด: {len(list((OUT / "images").glob("*")))} · data yaml: {YAML}')


if __name__ == '__main__':
    main()
