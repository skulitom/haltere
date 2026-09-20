"""Read one course's bundled ghost-pool recording from a local Liftoff install.

Optional UnityPy dependency; no game code is executed and no game files are written.
The register wire layout was checked against Steam build 25118475 (Unity 2022.3).
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
from pathlib import Path

from .replay import export_replay, read_replay


class _Reader:
    def __init__(self, data, offset=0):
        self.data, self.offset = data, offset

    def number(self, fmt='i'):
        value = struct.unpack_from('<' + fmt, self.data, self.offset)[0]
        self.offset += struct.calcsize(fmt)
        return value

    def string(self):
        size = self.number()
        if size < 0 or self.offset + size > len(self.data):
            raise ValueError('Invalid serialized string length')
        value = self.data[self.offset:self.offset + size].decode('utf-8')
        self.offset = (self.offset + size + 3) // 4 * 4
        return value

    def content_id(self):
        return {'str': self.string(), 'version': self.number(), 'type': self.number()}


def read_register(raw):
    reader = _Reader(raw, 28)  # Unity MonoBehaviour base fields before m_Name
    if reader.string() != 'RecordingRegister':
        raise ValueError('Unexpected bot recording register layout')
    count = reader.number()
    if not 0 <= count <= 100000:
        raise ValueError('Invalid recording register count')
    rows = []
    for _ in range(count):
        rows.append(dict(filename=reader.string(), name=reader.string(), environment=reader.string(),
                         gamemode=reader.number(), race_time_s=reader.number('f'),
                         stunt_score=reader.number(), track_id=reader.content_id(),
                         race_id=reader.content_id()))
    if reader.offset != len(raw):
        raise ValueError('Unrecognized trailing register data; game format may have changed')
    return rows


def _register_bundle(catalog, aa_dir):
    """Resolve RecordingRegister using the compact Addressables catalogue."""
    kb, bb, eb = (base64.b64decode(catalog[k]) for k in
                  ('m_KeyDataString', 'm_BucketDataString', 'm_EntryDataString'))
    reader, buckets = _Reader(bb), []
    for _ in range(reader.number()):
        offset, count = reader.number(), reader.number()
        entries = [reader.number() for _ in range(count)]
        key = None
        if kb[offset] in (0, 1):
            size = struct.unpack_from('<i', kb, offset + 1)[0]
            key = kb[offset + 5:offset + 5 + size].decode('ascii' if kb[offset] == 0 else 'utf-16le')
        buckets.append((key, entries))
    records = [struct.unpack_from('<7i', eb, 4 + 28 * i)
               for i in range(struct.unpack_from('<i', eb)[0])]
    pending = [i for key, entries in buckets if key == 'RecordingRegister' for i in entries]
    seen, bundles = set(), set()
    while pending:
        i = pending.pop()
        if i in seen:
            continue
        seen.add(i)
        record = records[i]
        location = catalog['m_InternalIds'][record[0]]
        prefix, separator, suffix = location.partition('#')
        if separator and prefix.isdigit():
            location = catalog['m_InternalIdPrefixes'][int(prefix)] + suffix
        if location.endswith('.bundle'):
            location = location.replace('{UnityEngine.AddressableAssets.Addressables.RuntimePath}', str(aa_dir))
            path = Path(location.replace('\\', '/')).resolve()
            if not path.is_relative_to(aa_dir.resolve()):
                raise ValueError('Catalogue bundle is outside the local Addressables directory')
            bundles.add(path)
        if record[2] >= 0:
            pending.extend(buckets[record[2]][1])
    if len(bundles) != 1:
        raise ValueError('Expected one local recording bundle; catalogue layout may have changed')
    return bundles.pop()


def extract_bot_route(game_dir, out, *, race_id, recording_key=None):
    if Path(out).exists():
        raise FileExistsError(f'Refusing existing output: {out}')
    try:
        import UnityPy
    except ImportError as exc:
        raise RuntimeError('Bot extraction needs the optional dependency: pip install "haltere[replays]"') from exc
    game_dir = Path(game_dir).resolve()
    data_dir = game_dir / 'Liftoff_Data'
    # Read the installed engine version instead of guessing for stripped bundles.
    resources = UnityPy.load(str(data_dir / 'resources.assets'))
    version = next(iter(resources.files.values())).unity_version
    UnityPy.config.FALLBACK_UNITY_VERSION = version
    del resources
    aa_dir = data_dir / 'StreamingAssets' / 'aa'
    catalog_bytes = (aa_dir / 'catalog.json').read_bytes()
    bundle = _register_bundle(json.loads(catalog_bytes), aa_dir)
    env = UnityPy.load(str(bundle))
    register = next((o for o in env.objects
                     if o.type.name == 'MonoBehaviour' and o.peek_name() == 'RecordingRegister'), None)
    if register is None:
        raise ValueError('No RecordingRegister in the resolved bundle')
    matches = [r for r in read_register(register.get_raw_data()) if r['race_id']['str'] == race_id]
    if not matches:
        raise ValueError(f'No bundled recordings for race {race_id}')
    matches.sort(key=lambda r: (r['race_time_s'], r['filename']))
    selected = (next((r for r in matches if r['filename'] == recording_key), None)
                if recording_key else matches[len(matches) // 2])
    if selected is None:
        raise ValueError('Recording key is not in the requested race pool')
    asset = next((o for o in env.objects
                  if o.type.name == 'TextAsset' and o.peek_name() == selected['filename']), None)
    if asset is None:
        raise ValueError('Selected recording is not in the register bundle; format may have changed')
    raw = asset.parse_as_object().m_Script.encode('utf-8', errors='surrogateescape')
    _, meta = read_replay(raw)
    if (meta['race_id'] != race_id or meta['environment'] != selected['environment']
            or meta['track_id'] != selected['track_id']['str']
            or meta['race_version'] != str(selected['race_id']['version'])):
        raise ValueError('Recording content disagrees with register metadata')
    return export_replay(raw, out, source=f'{bundle}::{selected["filename"]}', provenance={
        'kind': 'bundled_bot_recording_pool', 'game_dir': str(game_dir), 'unity_version': version,
        'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
        'recording_key': selected['filename'], 'selected_register_entry': selected,
        'matching_race_recordings': len(matches),
        'selection': 'explicit key' if recording_key else 'median reported race time',
        'matching_time_range_s': [matches[0]['race_time_s'], matches[-1]['race_time_s']],
    })
