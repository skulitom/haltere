"""The fast race-cue pilot as the runner deploys it, for offline rollouts (training data and surrogate gates only).

`deployed_pilot_kwargs(contract)` returns the FastRaceCue keyword arguments that
``haltere.liftoff.visual_brain`` builds for ``--pilot-profile fast --looming-brake --obstacle-stack on
--descent-view on`` and a motor contract, loaded through the runner's own loaders (frozen, hash-checked
declarations of the rule versions this code implements), plus a record of every declaration behind them:

- lag-aware turns (configs/obstacles/lag_turn.json, the contract's entry);
- the gap aim (configs/obstacles/gap_pilot.json);
- the wall-pilot rules: turn first at a wall with the contract's stopping model, ceiling guard and, from
  wall-pilot version 5, the clearance brake's sink floor (configs/obstacles/wall_pilot.json);
- the vertical guard (configs/obstacles/vertical_guard.json);
- the view-keeping descent (configs/pilot/descent_view.json) and, from its version 2, its contact support;
- with ``motor_assist=True`` (``--motor-assist on``, off by default), the motor-assist entry of the contract
  (configs/pilot/motor_assist.json; none for the fast PD).

The surrogate has no looming or gap samples, so the gap aim, the ceiling guard and the vertical guard stay idle
there; the lag-aware turns and the view-keeping descent act, and turn-first acts on a governor cap (for example a
`SyntheticCaps` object in the place of the looming governor). Nothing is per course.
"""
from __future__ import annotations

from ..vision.datasets import sha256

CONTRACTS = ('fast_velocity_brain_v1', 'fast_velocity_pd_v1')


def deployed_pilot_kwargs(contract='fast_velocity_brain_v1', *, stack=True, descent_view=True, motor_assist=False):
    """(FastRaceCue kwargs, declarations record) of the deployed pilot for a motor contract (see the module doc)."""
    from ..liftoff import fast_race_cue as frc
    from ..liftoff import visual_brain as vb
    if contract not in CONTRACTS:
        raise ValueError(f'Unknown fast motor contract {contract!r}')
    kwargs, record = {}, {}

    def note(name, path, declaration, digest):
        record[name] = dict(path=str(path), version=declaration.get('version'), sha256=digest, file_sha256=sha256(path))

    if stack:
        from ..liftoff.gap_aim import GapAimConfig
        from ..liftoff.gap_stack import GAP_PILOT_PATH, load_gap_pilot
        lag, digest = vb.load_lag_turn_declaration(vb.LAG_TURN_DECLARATION)
        lag_config = frc.lag_turn_for_contract(lag, contract)
        if lag_config is None:
            raise ValueError(f'The lag-turn declaration has no entry for {contract}')
        note('lag_turn', vb.LAG_TURN_DECLARATION, lag, digest)
        wall, digest = vb.load_wall_pilot()
        note('wall_pilot', vb.WALL_PILOT_DECLARATION, wall, digest)
        vertical, digest = vb.load_vertical_guard()
        note('vertical_guard', vb.VERTICAL_GUARD_DECLARATION, vertical, digest)
        gap, digest = load_gap_pilot(GAP_PILOT_PATH)
        note('gap_pilot', GAP_PILOT_PATH, gap, digest)
        kwargs.update(lag_turn=lag_config, gap_aim=GapAimConfig.from_dict(gap['pilot']),
                      vertical_guard=frc.vertical_guard_config(vertical), **frc.wall_pilot_configs(wall, contract))
        record['motor_contract'] = contract
    if descent_view:
        view, digest = vb.load_descent_view()
        note('descent_view', vb.DESCENT_VIEW_DECLARATION, view, digest)
        kwargs['descent_view'] = frc.descent_view_config(view)
        contact = frc.contact_support_config(view)
        if contact is not None:
            # descent view version 2: contact support, as the runner adds it (it reads the pad calibration, which the
            # surrogate's pilots receive from the checkpoint)
            kwargs['contact_support'] = contact
    if motor_assist:
        assist, digest = vb.load_motor_assist()
        note('motor_assist', vb.MOTOR_ASSIST_DECLARATION, assist, digest)
        config = frc.motor_assist_for_contract(assist, contract)
        if config is not None:
            kwargs['motor_assist'] = config
    return kwargs, record
