"""The beacon detector: what it must find, what it must refuse, and that a bad recovery is visible."""
import numpy as np
import pytest

from haltere.vision.beacon import BeaconParams, gates_from_beacons, marker_mask, markers_in_frame
from haltere.vision.camera import Camera, world_to_body

cv2 = pytest.importorskip('cv2')

BEACON = (40, 235, 130)          # emissive green cone
FOLIAGE = (120, 150, 60)         # sunlit leaves: green but yellow-green, R close to G
SHADE = (30, 45, 25)             # forest shade
MAGENTA = (230, 60, 210)         # the other ground light-strip colour
TENT = (150, 150, 155)           # camp tent / countdown ring: grey, what fools a human at thumbnail size


def _scene(w=640, h=360):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = SHADE
    img[60:200, 0:200] = FOLIAGE
    return img


def test_the_colour_test_takes_the_beacon_and_leaves_foliage_magenta_and_tents():
    img = _scene()
    img[250:260, 300:312] = BEACON          # 120 px, compact
    img[240:300, 380:440] = MAGENTA
    img[240:300, 60:120] = TENT
    m = marker_mask(img)
    assert m[250:260, 300:312].all()
    assert m.sum() == 120                   # and nothing else in the frame passed
    hits = markers_in_frame(img)
    assert len(hits) == 1
    u, v, n = hits[0]
    assert n == 120 and abs(u - 305.5) < 1.0 and abs(v - 254.5) < 1.0


def test_the_hud_is_masked_out_by_position():
    """The leaderboard's green tiles pass the colour test perfectly and never move, so a detector that
    trusted colour alone would triangulate a point from a user-interface element."""
    img = _scene()
    img[100:140, 580:620] = BEACON          # right-hand leaderboard
    img[10:40, 300:340] = BEACON            # top plate
    assert not marker_mask(img).any()
    assert markers_in_frame(img) == []


def test_a_ground_light_strip_is_refused_on_size_and_on_shape():
    """The strips are the beacon's colour, so only size and shape can reject them."""
    img = _scene()
    img[300:316, 40:540] = BEACON           # standing next to one: too big, and its centroid slides
    assert marker_mask(img).sum() > BeaconParams().max_px
    assert markers_in_frame(img) == []
    small = _scene()
    small[300:304, 40:160] = BEACON         # a distant strip: within max_px, still a sliver
    assert BeaconParams().min_px < marker_mask(small).sum() < BeaconParams().max_px
    assert markers_in_frame(small) == [], 'elongation must still reject it'


def test_chroma_noise_is_below_the_floor():
    img = _scene()
    img[250:252, 300:302] = BEACON          # 4 px
    assert marker_mask(img).any()
    assert markers_in_frame(img) == []


def test_the_pieces_of_one_marker_are_merged_before_they_are_measured():
    """The beacon renders as a cone, a shade and a bright core, and the colour test cuts it into
    pieces a pixel or two apart. Measuring them separately drops each below the floor and loses the
    detection - on pine1 that is the difference between finding the beacon in 214 frames and in 134."""
    img = _scene()
    img[250:256, 300:306] = BEACON          # 36 px
    img[258:264, 300:306] = BEACON          # 36 px, two rows of background between them
    hits = markers_in_frame(img, BeaconParams(min_px=50))
    assert len(hits) == 1 and hits[0][2] == 72


def test_every_admissible_marker_is_emitted_not_just_the_biggest():
    """Picking one blob per frame threw the beacon away in half the frames it appeared in on pine1
    (at best 49% of the picked blobs were the beacon). Sorting them out is the RANSAC's job."""
    img = _scene()
    img[250:262, 400:412] = BEACON          # 144 px
    img[300:308, 120:128] = BEACON          # 64 px
    hits = markers_in_frame(img)
    assert [h[2] for h in hits] == [144, 64], 'largest first, both kept'


def _fly(cam, points, path, noise_px=0.0, seed=0, yaw_bias_deg=12.0, also=()):
    """Fly `path` leg by leg, looking roughly at `points[i]` on leg i (a pilot chasing the marker, not
    staring at it), and return (index rows, detections). Points in `also` are visible throughout, like
    a static light-strip. Only projections that land inside the frame become detections: a detector
    sees what is in the picture and nothing else."""
    rng = np.random.default_rng(seed)
    rows, det = [], []
    for i, leg in enumerate(path):
        tgt = np.asarray(points[i], float)
        for pos in leg:
            pos = np.asarray(pos, float)
            yaw = np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0]) + np.radians(yaw_bias_deg)
            q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
            f = f'{len(rows):06d}.jpg'
            rows.append({'file': f, 'px': pos[0], 'py': pos[1], 'pz': pos[2],
                         'qw': q[0], 'qx': q[1], 'qy': q[2], 'qz': q[3]})
            for p in [tgt, *(np.asarray(a, float) for a in also)]:
                px, ok = cam.project_body(world_to_body(p[None], pos, q))
                u, v = px[0]
                if ok[0] and 0 <= u < cam.width and 0 <= v < cam.height:
                    det.append({'file': f, 'u': u + rng.normal(0, noise_px),
                                'v': v + rng.normal(0, noise_px), 'n': 100})
    return rows, det


def _dataset(tmp_path, rows):
    import csv
    d = tmp_path / 'ds'
    (d / 'frames').mkdir(parents=True)
    with open(d / 'index.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return d


def test_ransac_splits_the_rays_into_one_cluster_per_checkpoint(tmp_path):
    """The marker steps from gate to gate as the flight progresses, so a single least-squares
    intersection of every ray would answer with a point that is not any gate."""
    cam = Camera(640, 360, 300.0, 30.0)
    g0 = np.array([22.0, 4.0, 2.0])
    g1 = np.array([26.0, -9.0, 3.0])
    leg0 = [(x, 0.0, 1.5) for x in np.arange(0.0, 14.0, 1.0)]
    leg1 = [(22.0 + x, 4.0 - 1.4 * x, 1.8) for x in np.arange(0.0, 8.0, 0.6)]
    rows, det = _fly(cam, [g0, g1], [leg0, leg1], noise_px=1.0, seed=1)
    gates = gates_from_beacons(_dataset(tmp_path, rows), det, cam, min_inliers=6, verbose=False)
    assert len(gates) == 2
    assert [g['gate'] for g in gates] == [0, 1]              # flight order
    assert np.linalg.norm(np.array(gates[0]['pos']) - g0) < 0.4
    assert np.linalg.norm(np.array(gates[1]['pos']) - g1) < 0.6
    assert all(g['rms'] < 0.5 and g['n'] >= 6 for g in gates)
    assert gates[0]['first_frame'] < gates[1]['first_frame']


def test_a_static_strip_seen_all_flight_forms_its_own_cluster_and_does_not_move_the_gate(tmp_path):
    """The strips are not rejected by colour or by RANSAC - they are real static objects and come back
    as their own clusters. What must not happen is their rays leaking into the checkpoint's."""
    cam = Camera(640, 360, 300.0, 30.0)
    gate = np.array([22.0, 4.0, 2.0])
    strip = np.array([25.0, 10.0, 2.2])      # a second static green marker, 6.7 m off the gate
    leg = [(x, 0.0, 1.5) for x in np.arange(0.0, 16.0, 0.5)]
    rows, det = _fly(cam, [gate], [leg], noise_px=0.5, seed=5, also=[strip])
    gates = gates_from_beacons(_dataset(tmp_path, rows), det, cam, min_inliers=6, verbose=False)
    near_gate = [g for g in gates if np.linalg.norm(np.array(g['pos']) - gate) < 0.4]
    near_strip = [g for g in gates if np.linalg.norm(np.array(g['pos']) - strip) < 0.4]
    assert len(near_gate) == 1 and len(near_strip) == 1, [np.round(g['pos'], 1).tolist() for g in gates]
    assert near_gate[0]['rms'] < 0.3 and near_strip[0]['rms'] < 0.3
    assert not set(d['file'] + str(round(d['u'])) for d in near_gate[0]['obs']) & \
               set(d['file'] + str(round(d['u'])) for d in near_strip[0]['obs'])


def test_outlier_rays_are_left_out_rather_than_dragging_the_answer(tmp_path):
    cam = Camera(640, 360, 300.0, 30.0)
    gate = np.array([22.0, 4.0, 2.0])
    leg = [(x, 0.0, 1.5) for x in np.arange(0.0, 16.0, 0.8)]
    rows, det = _fly(cam, [gate], [leg], noise_px=0.5, seed=2)
    clean = len(det)
    for d in det[:4]:                                        # four blobs that slipped the shape gate
        det.append({'file': d['file'], 'u': d['u'] - 160.0, 'v': d['v'] + 40.0, 'n': 100})
    gates = gates_from_beacons(_dataset(tmp_path, rows), det, cam, min_inliers=6, verbose=False)
    assert gates, 'the good rays still form a gate'
    assert np.linalg.norm(np.array(gates[0]['pos']) - gate) < 0.3
    assert gates[0]['n'] >= clean - 2 and gates[0]['rms'] < 0.3


def test_a_recovery_with_too_little_support_is_refused_not_reported(tmp_path):
    """Silence is the required behaviour: a confidently wrong gate poisons every frame labelled from it."""
    cam = Camera(640, 360, 300.0, 30.0)
    gate = np.array([22.0, 4.0, 2.0])
    leg = [(x, 0.0, 1.5) for x in np.arange(0.0, 4.0, 1.0)]
    rows, det = _fly(cam, [gate], [leg], seed=3)
    assert gates_from_beacons(_dataset(tmp_path, rows), det, cam, min_inliers=8, verbose=False) == []


def test_the_inlier_rays_round_trip_through_the_hand_read_pipeline(tmp_path):
    """The recovery must be re-runnable and editable by the same command a human's readings go through."""
    from haltere.vision.beacon import save_observations
    from haltere.vision.triangulate import gates_from_observations, load_observations
    cam = Camera(640, 360, 300.0, 30.0)
    gate = np.array([22.0, 4.0, 2.0])
    rows, det = _fly(cam, [gate], [[(x, 0.0, 1.5) for x in np.arange(0.0, 14.0, 1.0)]], seed=4)
    ds = _dataset(tmp_path, rows)
    found = gates_from_beacons(ds, det, cam, min_inliers=6, verbose=False)
    save_observations(found, tmp_path / 'obs.json')
    g = gates_from_observations(ds, load_observations(tmp_path / 'obs.json'), cam, verbose=False)
    assert len(g) == 1 and np.linalg.norm(np.array(g[0]['pos']) - gate) < 0.2


def test_the_gates_json_carries_no_ray_lists(tmp_path):
    from haltere.vision.beacon import strip_obs
    cam = Camera(640, 360, 300.0, 30.0)
    rows, det = _fly(cam, [np.array([22.0, 4.0, 2.0])],
                     [[(x, 0.0, 1.5) for x in np.arange(0.0, 14.0, 1.0)]], seed=7)
    found = gates_from_beacons(_dataset(tmp_path, rows), det, cam, min_inliers=6, verbose=False)
    assert found and 'obs' in found[0]
    assert all('obs' not in g for g in strip_obs(found))


def test_nothing_that_flies_imports_the_beacon():
    """Steering to the game's own next-checkpoint marker would not be flying by sight, it would be
    reading the answer off the screen, and every by-sight number measured that way would be a lie.
    The module is for ground truth only, and the CLI is the only place allowed to reach it."""
    import pathlib
    allowed = {pathlib.Path('haltere/cli.py'), pathlib.Path('haltere/vision/beacon.py')}
    offenders = [str(p) for p in pathlib.Path('haltere').rglob('*.py')
                 if p not in allowed and 'beacon' in p.read_text(encoding='utf-8')]
    assert offenders == [], f'these modules reach for the beacon: {offenders}'
