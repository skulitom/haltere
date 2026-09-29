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
- the view-keeping descent (configs/pilot/descent_view.json) and, from its version 2, its contact support
  (``contact_support='off'|'shadow'`` as the runner's --contact-support; version 3);
- with ``motor_assist=True`` (``--motor-assist on``, off by default), the motor-assist entry of the contract
  (configs/pilot/motor_assist.json, version 4 from round 6; none for the fast PD); ``motor_assist=3`` takes the kept
  round-5 declaration (motor_assist_v3.json), with which the brain-12 training rollouts and gates were made;
- with ``sighted_descent='on'|'shadow'`` (``--sighted-descent``; off by default), the sighted descent
  (configs/pilot/sighted_descent.json; it limits the view rule's steep late to the ring's sighted line of sight);
- with ``stale_evidence=True``, the stale-evidence rule (configs/obstacles/stale_evidence.json; the looming governor's
  cap follows the ray of its evidence). The runner adds it only with ``--stale-evidence on`` inside the obstacle stack
  (version 2; off by default); it is off here by default too, so the brain-11 and brain-12 records built on this
  module stay reproducible. The surrogate has no looming samples, so it is idle there;
- with ``early_brake=True`` (``--early-brake on``, off by default, inside the obstacle stack), the early-brake entry of
  the contract (configs/obstacles/early_brake.json; none for the fast PD): the looming governor engages as early as the
  contract's stopping model needs. Off by default here too; idle in the surrogate (no looming samples);
- with ``marker_jump='on'|'shadow'`` (``--marker-jump on|shadow``, off by default), the marker-jump rule
  (configs/pilot/marker_jump.json: a checkpoint marker that jumps after a gap in the readings is held until confirmed;
  shadow computes it without holding). Off here by default too;
- with ``ring_lead='on'|'shadow'`` (``--ring-lead on|shadow``, off by default, inside the obstacle stack), the ring-lead
  entry of the contract (configs/pilot/ring_lead.json; none for the fast PD): while the in-view ring's line of sight
  swings, the lag-aware turn leads the ring bearing; shadow computes it without changing any request. Off here by
  default too.

The surrogate has no looming or gap samples, so the gap aim, the ceiling guard and the vertical guard stay idle
there; the lag-aware turns and the view-keeping descent act, and turn-first acts on a governor cap (for example a
`SyntheticCaps` object in the place of the looming governor). Nothing is per course.
"""
from __future__ import annotations

from ..vision.datasets import sha256

CONTRACTS = ('fast_velocity_brain_v1', 'fast_velocity_pd_v1')


def deployed_pilot_kwargs(contract='fast_velocity_brain_v1', *, stack=True, descent_view=True, motor_assist=False,
                          stale_evidence=False, contact_support='on', marker_jump=None, early_brake=False,
                          sighted_descent='off', ring_lead=None):
    """(FastRaceCue kwargs, declarations record) of the deployed pilot for a motor contract (see the module doc)."""
    from ..liftoff import fast_race_cue as frc
    from ..liftoff import visual_brain as vb
    if contract not in CONTRACTS:
        raise ValueError(f'Unknown fast motor contract {contract!r}')
    if contact_support not in frc.CONTACT_SUPPORT_MODES:
        raise ValueError(f'Unknown contact-support mode {contact_support!r}')
    if sighted_descent not in frc.SIGHTED_DESCENT_MODES or (sighted_descent != 'off' and not descent_view):
        raise ValueError(f'Unknown sighted-descent mode {sighted_descent!r}, or on/shadow without the descent view')
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
        if contact is not None and contact_support != 'off':
            # descent view version 2: contact support, as the runner adds it (it reads the pad calibration, which the
            # surrogate's pilots receive from the checkpoint); version 3 in shadow: computed, no climb
            kwargs['contact_support'] = contact
            if contact_support == 'shadow':
                kwargs['contact_apply'] = False
        record['descent_view']['contact_support'] = None if contact is None else contact_support
        if sighted_descent != 'off':
            # --sighted-descent on|shadow (off by default: not passed, the pilot exactly as before)
            sighted, digest = vb.load_sighted_descent()
            note('sighted_descent', vb.SIGHTED_DESCENT_DECLARATION, sighted, digest)
            record['sighted_descent']['mode'] = sighted_descent
            kwargs['sighted_descent'] = frc.sighted_descent_config(sighted)
            if sighted_descent == 'shadow':
                kwargs['sighted_apply'] = False
    if stack and stale_evidence:
        stale, digest = vb.load_stale_evidence()
        note('stale_evidence', vb.STALE_EVIDENCE_DECLARATION, stale, digest)
        kwargs.update(frc.stale_evidence_configs(stale))
    if stack and early_brake:
        early, digest = vb.load_early_brake()
        note('early_brake', vb.EARLY_BRAKE_DECLARATION, early, digest)
        config = frc.early_brake_for_contract(early, contract)
        if config is not None:
            kwargs['early_brake'] = config
    if motor_assist:
        # True: the declaration the runner flies (version 4 from round 6); an int: that kept version (3: the round-5
        # declaration motor_assist_v3.json, with which the brain-12 records were made), hash-checked like the runner's
        path = vb.MOTOR_ASSIST_DECLARATION
        if motor_assist is not True:
            import json
            from ..liftoff.gap_stack import config_sha256
            if int(motor_assist) != frc.MOTOR_ASSIST_VERSION:
                path = path.with_name(f'motor_assist_v{int(motor_assist)}.json')
            assist = json.loads(path.read_text(encoding='utf-8'))
            digest = config_sha256(assist)
            if assist.get('frozen') is not True or assist.get('sha256') != digest or assist['version'] != int(motor_assist):
                raise ValueError(f'{path} is not the frozen motor-assist declaration version {motor_assist}')
        else:
            assist, digest = vb.load_motor_assist()
        note('motor_assist', path, assist, digest)
        config = frc.motor_assist_for_contract(assist, contract)
        if config is not None:
            kwargs['motor_assist'] = config
    if marker_jump is not None:
        if marker_jump not in frc.MARKER_JUMP_MODES:
            raise ValueError(f'Unknown marker-jump mode {marker_jump!r}')
        declaration, digest = vb.load_marker_jump()
        note('marker_jump', vb.MARKER_JUMP_DECLARATION, declaration, digest)
        kwargs.update(marker_jump=frc.marker_jump_config(declaration), marker_jump_apply=marker_jump == 'on')
    if ring_lead is not None:
        if ring_lead not in frc.RING_LEAD_MODES or not stack:
            raise ValueError(f'Unknown ring-lead mode {ring_lead!r}, or the ring lead without the stack lag-aware turn')
        declaration, digest = vb.load_ring_lead()
        note('ring_lead', vb.RING_LEAD_DECLARATION, declaration, digest)
        record['ring_lead']['mode'] = ring_lead
        config = frc.ring_lead_for_contract(declaration, contract)
        if config is not None:
            kwargs.update(ring_lead=config, ring_lead_apply=ring_lead == 'on')
    return kwargs, record
