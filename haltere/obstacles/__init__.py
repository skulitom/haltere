"""Obstacle clearance bench and learned clearance fan (milestone M1 onwards).

The package separates code that a live flight may run from code that only runs
offline on recordings. Offline labels use hindsight (future frames, whole-flight
triangulation, impact annotations, box colliders). They are training and
evaluation targets only and must never reach a runtime process.

Runtime-safe modules (causal; may be imported by the flight stack):

- ``contract``  image, camera, grid and fan geometry shared by runtime and offline code
- ``overlays``  per-frame HUD, ring, ghost-trail and propeller masks, input validity channel
- ``model``     ClearanceNet definition and its output contract
- ``runtime``   (M4) the clearance process
- ``governor``  (M3) ClearanceFanGovernor

Offline-only modules (never imported by runtime modules):

- ``splits``    environments, folds, sealed-set guard, per-run environment assignment
- ``store``     frame store format, writer and reader (runs/obstacle-store-v1)
- ``timing``    per-video clock refinement
- ``labels``    offline label schema and builders (hindsight, colliders, impacts, tube, teacher)
- ``evaluate``  held-out harness E1-E8, baselines B0-B4, thresholds freeze, scoring ledger
- ``baselines`` pretrained-depth baselines B2-B4 (GPU runners, B3 calibration, B4 ceiling), model inference
- ``leaks``     E8 perturbations (ring paint/removal, HUD-glyph scramble, ghost trails)
- ``split_looming``  B1, the lateral study's split-field looming cue (vendored)
- ``lateral_cache``  the lateral study's caches as a pseudo store (reproduction of prior numbers)
- ``train``     ClearanceNet training
- ``thermal``   flight lock, GPU temperature and thread limits for heavy jobs

``tests/test_obstacle_label_isolation.py`` proves that no runtime module can reach
``haltere.obstacles.labels`` through any import chain. The labels package also
refuses to import in a process that sets ``HALTERE_RUNTIME_PROCESS=1``.

This ``__init__`` must stay free of submodule imports.
"""

SCHEMA_VERSION = 1

LABEL_PACKAGE = 'haltere.obstacles.labels'
RUNTIME_ENV_FLAG = 'HALTERE_RUNTIME_PROCESS'

# Modules that a live flight may import. Listed even before they exist (M3/M4);
# the isolation test checks each one that is present.
RUNTIME_MODULES = (
    'haltere.obstacles',
    'haltere.obstacles.contract',
    'haltere.obstacles.overlays',
    'haltere.obstacles.model',
    'haltere.obstacles.runtime',
    'haltere.obstacles.governor',
)

# Packages whose every module is part of the flight stack.
RUNTIME_PACKAGES = (
    'haltere.liftoff',
    'haltere.vision',
    'haltere.brain',
    'haltere.sim',
)

OFFLINE_MODULES = (
    'haltere.obstacles.splits',
    'haltere.obstacles.store',
    'haltere.obstacles.store_build',
    'haltere.obstacles.timing',
    'haltere.obstacles.labels',
    'haltere.obstacles.evaluate',
    'haltere.obstacles.baselines',
    'haltere.obstacles.leaks',
    'haltere.obstacles.split_looming',
    'haltere.obstacles.lateral_cache',
    'haltere.obstacles.train',
    'haltere.obstacles.thermal',
)
