# Flight cards

One card per flight, written **before** it and appended to after. A prediction written after the
flight does not exist. Rules: `docs/generalisation.md`.

Name: `<date>_<track>_<n>.md`. Template:

```markdown
# 2026-09-16 pinevalley 1
- **Rung**: Z4 (unseen map, unseen gate type)   <!-- Z0-Z4, see docs/generalisation.md -->
- **Track**: Pine Valley / "1 - Forest For The Trees"  (pool: dev)
- **Detector**: runs/gatenet10/best.pt  sha256 <first 12>
- **Command**: <the exact line, copy-pasteable>
- **Course parameters typed on the command line**: none   <!-- must be "none" on a sealed track -->
- **Prediction**: <what you expect, as a number, before flying>
- **Pass criterion**: <decided now, not after>

## Result
- HUD checkpoints: n/m, splits ...
- Pilot's own `n_passes`: ...  (precision/recall against the HUD: ...)
- `liftoff score --gates ""`: speed, shake, contacts
- Verdict against the criterion: pass / fail
- What it changed: <if this was a sealed track and anything here changes the code, the track
  becomes dev - record that>
```
