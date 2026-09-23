"""Render a flight card from preserved run metadata, scores and HUD review.

The scorer cannot infer a race finish. A separate review names the finish
evidence; missing evidence stays unverified, and terminal impacts are retained.
Use before flight without a score, then write a separate result card afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def render_card(manifest, score=None, review=None):
    review = review or {}
    segments = []
    if score is not None:
        if not isinstance(score, dict):
            raise ValueError('Expected the scorer JSON object')
        for values in score.values():
            if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
                raise ValueError('Expected scorer lists of flight segments')
            segments.extend(values)
    impact = any(s.get('terminal_impact') or s.get('collisions') for s in segments)
    finish = review.get('finish_confirmed') is True
    evidence = review.get('finish_evidence', [])
    if finish and not evidence:
        raise ValueError('A confirmed finish must name saved game finish evidence')
    if finish:
        verdict = 'Game finish confirmed; contact criterion failed.' if impact else (
            'Game finish confirmed. Verify reset/intervention and clearance criteria from the review.')
    elif impact:
        verdict = 'Failed: impact recorded; no confirmed game finish.'
    elif score is not None:
        verdict = 'No confirmed game finish. Do not count as a completed race.'
    else:
        verdict = 'Preflight: result pending.'
    if manifest.get('runtime_route_oracle') is True or manifest.get('pilot_assistance') == 'oracle-route':
        verdict = 'Oracle collection only; ineligible for autonomous evaluation. '+verdict

    lines = [f"# {manifest.get('track', manifest.get('task', 'Flight'))}", '', verdict, '',
             f"- Exposure: {manifest.get('exposure', 'not recorded')}",
             f"- Runtime revision: `{manifest.get('base_revision', 'not recorded')}`",
             f"- Controller mode: {manifest.get('mode', 'not recorded')}",
             f"- Brain SHA-256: `{manifest.get('brain_sha256', 'not recorded')}`",
             f"- Time limit: {manifest.get('duration_s', 'not recorded')} s", '',
             '## Declared before flight', '',
             f"Prediction: {manifest.get('prediction', 'not recorded')}", '',
             f"Pass criterion: {manifest.get('criterion', 'not recorded')}", '',
             '## Measurements', '']
    if not segments:
        lines += ['No scorer measurements supplied.', '']
    for i, segment in enumerate(segments, 1):
        lines += [f'Flight segment {i}:', '']
        for key in ('airborne_s', 'speed_median', 'speed_p90', 'stop_reason'):
            value = segment.get(key, 'not recorded')
            if isinstance(value, float):
                value = f'{value:.3f}'
            lines.append(f'- {key}: {value}')
        lines += [f"- CSV contact estimates: {len(segment.get('collisions', []))}",
                  f"- Terminal impact: {json.dumps(segment.get('terminal_impact'))}", '']
    lines += ['## Game/video review', '',
              '```json', json.dumps(review, indent=2), '```', '',
              '## Preserved run contract', '',
              '```json', json.dumps(manifest, indent=2), '```', '']
    return '\n'.join(lines)


def write_card(manifest_path, out, score_path=None, review_path=None):
    paths = [Path(p) for p in (manifest_path, score_path, review_path) if p]
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    score = json.loads(Path(score_path).read_text(encoding='utf-8')) if score_path else None
    review = json.loads(Path(review_path).read_text(encoding='utf-8')) if review_path else None
    if review and review.get('finish_confirmed'):
        for item in review.get('finish_evidence', []):
            evidence = Path(item)
            if not evidence.is_file():
                raise ValueError(f'Missing saved finish evidence: {item}')
            paths.append(evidence)
    card = render_card(manifest, score, review)
    card += '\n## Source files\n\n'
    for p in paths:
        with p.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        card += f'- `{p.as_posix()}` — SHA-256 `{digest}`\n'
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        f.write(card)
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--score')
    p.add_argument('--review')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    print(write_card(a.manifest, a.out, a.score, a.review))


if __name__ == '__main__':
    main()
