"""Inventory labelled flights and reject leakage between training and validation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(dataset, *, hashes=False):
    """Check labels against actual files; flags describe evidence gaps, not flight success."""
    root = Path(dataset).resolve()
    labels_path, index_path = root / 'labels.json', root / 'index.csv'
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    if not labels:
        raise ValueError(f'{root}: no labels')
    names, fingerprints, positive, gates = set(), set(), 0, set()
    image_hashes = {}
    for lab in labels:
        name = lab['file']
        path = (root / 'frames' / name).resolve()
        if not path.is_relative_to(root / 'frames') or name in names:
            raise ValueError(f'{root}: duplicate or escaping frame path {name!r}')
        if not path.is_file():
            raise ValueError(f'{root}: missing frame {name}')
        names.add(name)
        if lab['visible'] not in (0, 1):
            raise ValueError(f'{root}: invalid visibility for {name}')
        if lab['visible']:
            if not np.isfinite([lab[k] for k in ('u', 'v', 'width_px')]).all() or lab['width_px'] <= 0:
                raise ValueError(f'{root}: invalid visible label for {name}')
            positive += 1
            if 'gate' in lab:
                gates.add(lab['gate'])
        if hashes:
            image_hashes[name] = sha256(path)
            fingerprints.add(image_hashes[name])
    index_hash = None
    if index_path.exists():
        with index_path.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        index_names = [r['file'] for r in rows]
        if len(set(index_names)) != len(index_names) or not names.issubset(index_names):
            raise ValueError(f'{root}: labels and frame index disagree')
        index_hash = sha256(index_path)
    capture_path = root / 'capture.json'
    capture = json.loads(capture_path.read_text(encoding='utf-8')) if capture_path.exists() else {}
    warnings = []
    if not capture.get('labels_reviewed'):
        warnings.append('No recorded visual review of labels; projection does not check occlusion.')
    if not capture.get('course_complete'):
        warnings.append('No recorded full-course completion; unlisted gates may be mislabeled as background.')
    return {
        'path': str(root), 'frames': len(labels), 'positive': positive,
        'negative': len(labels) - positive, 'labelled_gate_ids': sorted(gates, key=str),
        'labels_sha256': sha256(labels_path), 'index_sha256': index_hash,
        'images_sha256': hashlib.sha256(json.dumps(image_hashes, sort_keys=True).encode()).hexdigest() if hashes else None,
        'capture_sha256': sha256(capture_path) if capture_path.exists() else None,
        'course': capture.get('course'), 'source': capture.get('source', 'legacy/unknown'),
        'warnings': warnings,
    }, fingerprints


def audit_split(datasets, holdout=(), *, hashes=True):
    train, validation = [Path(p).resolve() for p in datasets], [Path(p).resolve() for p in holdout]
    if not train:
        raise ValueError('At least one training dataset is required')
    for split in (train, validation):
        if len(set(split)) != len(split):
            raise ValueError('A dataset is repeated within a split')
    if set(train) & set(validation):
        raise ValueError('Training and holdout contain the same dataset')
    report = {'schema': 1, 'split': 'whole_flights' if validation else 'inventory_only',
              'exact_image_overlap_checked': hashes, 'train': [], 'validation': []}
    train_images = set()
    for split, paths in (('train', train), ('validation', validation)):
        for path in paths:
            row, images = inventory(path, hashes=hashes)
            if split == 'train':
                train_images.update(images)
            elif images & train_images:
                raise ValueError(f'{path}: {len(images & train_images)} exact images also occur in training')
            report[split].append(row)
    return report
