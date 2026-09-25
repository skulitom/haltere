import numpy as np
import pytest

from haltere.obstacles import splits
from haltere.obstacles.splits import (ENVIRONMENTS, FOLDS, LOEO_FOLDS, OPEN_ENVS, SEALED_ENVS, SealedAccessError,
                                      env_code_mask, env_side, fold_envs, heldback_flight, load_folds_config,
                                      sampling_weights)


def test_environment_codes_are_unique_and_frozen():
    codes = [e.code for e in ENVIRONMENTS]
    assert len(set(codes)) == len(codes)
    assert max(codes) < splits.UNKNOWN_ENV_CODE
    # Codes are written into the frame-store index: these must never change.
    assert splits.ENV_CODE['Straw Bale'] == 0
    assert splits.ENV_CODE['Minus Two'] == 4
    assert splits.ENV_CODE['Pine Valley'] == 5
    assert splits.ENV_CODE['Hall 26'] == 12
    assert set(SEALED_ENVS) == {'The Green', 'Hall 26'}


def test_f12_holds_out_both_target_environments_and_forest():
    f = FOLDS['F12']
    assert set(f.test) == {'Minus Two', 'Pine Valley', 'Autumn Fields'}
    assert set(f.inner_val) == {'Hangar C03', 'Hannover', 'Paris', 'The Pit'}
    assert set(FOLDS['F3'].test) == {'Straw Bale'}
    assert set(FOLDS['F4'].test) == {'Drawing Board box course', 'Drawing Board loop v2', 'Drawing Board empty arena'}
    assert set(FOLDS['F5'].test) == {'Hangar C03', 'Hannover', 'Paris', 'The Pit'}


def test_loeo_folds_hold_out_every_open_environment_exactly_once():
    held = [e for k in LOEO_FOLDS for e in FOLDS[k].test]
    assert sorted(held) == sorted(OPEN_ENVS)


@pytest.mark.parametrize('name', list(FOLDS))
def test_fold_sides_are_disjoint_and_never_contain_sealed(name):
    f = FOLDS[name]
    train, test, iv, it = set(f.train), set(f.test), set(f.inner_val), set(f.inner_train)
    assert not train & test
    assert iv <= train and not iv & test
    assert it | iv == train and not it & iv
    for side in (train, test, iv, it):
        assert not side & set(SEALED_ENVS)
    assert train | test == set(OPEN_ENVS)


@pytest.mark.parametrize('name', list(FOLDS))
def test_related_layouts_and_domains_stay_on_one_side(name):
    for group in (splits.DRAWING_BOARD, splits.FOREST):
        sides = {env_side(name, e) for e in group}
        assert len(sides) == 1, (name, group, sides)


def test_sealed_set_needs_the_explicit_flag():
    with pytest.raises(SealedAccessError):
        fold_envs('F12', 'sealed')
    assert set(fold_envs('F12', 'sealed', sealed_final=True)) == set(SEALED_ENVS)
    assert env_side('F12', 'The Green') == 'sealed'
    with pytest.raises(SealedAccessError):
        splits.require_sealed(False)


def test_all_fold_uses_flight_holdback():
    f = FOLDS['ALL']
    assert f.test == () and f.flight_heldback
    assert set(fold_envs('ALL', 'inner_val')) == set(OPEN_ENVS)
    assert heldback_flight('x/y') == heldback_flight('x/y')
    keys = [f'run-{i}/flight' for i in range(4000)]
    frac = np.mean([heldback_flight(k) for k in keys])
    assert 0.08 < frac < 0.12


def test_env_code_mask_selects_fold_sides():
    codes = np.array([splits.ENV_CODE[n] for n in ('Minus Two', 'Straw Bale', 'Paris', 'Pine Valley')], np.uint8)
    assert env_code_mask(codes, 'F12', 'test').tolist() == [True, False, False, True]
    assert env_code_mask(codes, 'F12', 'inner_val').tolist() == [False, False, True, False]
    assert env_code_mask(codes, 'F12', 'inner_train').tolist() == [False, True, False, False]


def test_sampling_weights_apply_asset_caps():
    counts = {'Straw Bale': 195000, 'Drawing Board box course': 42000, 'Drawing Board loop v2': 1300,
              'Drawing Board empty arena': 1, 'Hangar C03': 1100, 'Hannover': 5300, 'Paris': 2000, 'The Pit': 19000}
    w = sampling_weights(counts)
    assert abs(sum(w.values()) - 1) < 1e-9
    assert w['Straw Bale'] <= 0.30 + 1e-9
    db = sum(v for k, v in w.items() if k.startswith('Drawing Board'))
    assert db <= 0.20 + 1e-9
    assert w['The Pit'] > w['Hannover'] > w['Paris'] > w['Hangar C03']
    assert sampling_weights({}) == {}


def test_folds_config_matches_code_and_inventory_assignment():
    cfg = load_folds_config()          # raises if folds.json drifted from splits.py
    assert cfg['primary_fold'] == 'F12'
    envs = set(OPEN_ENVS) | set(SEALED_ENVS)
    usable = [s for s in cfg['sources'].values() if s['usable']]
    assert len(usable) > 200
    for sid, s in cfg['sources'].items():
        assert s['env'] is None or s['env'] in envs, sid
        if s['usable']:
            assert s['flight'] is not None, sid
            assert cfg['flights'][s['flight']] == s['env'], sid
    per_env = {e: sum(1 for s in usable if s['env'] == e) for e in envs}
    for e in ('Minus Two', 'Pine Valley', 'Straw Bale', 'Drawing Board box course'):
        assert per_env[e] > 0, e
    for e in SEALED_ENVS:
        assert per_env[e] > 0   # present in the assignment; the store builder skips them without --sealed-final
