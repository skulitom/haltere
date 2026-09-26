"""Runtime code must never reach the offline label package (hindsight labels are not runtime inputs)."""
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import haltere
from haltere.obstacles import LABEL_PACKAGE, OFFLINE_MODULES, RUNTIME_ENV_FLAG, RUNTIME_MODULES, RUNTIME_PACKAGES

PACKAGE_DIR = Path(haltere.__file__).resolve().parent


def discover(package_dir: Path, package: str) -> dict:
    """{module name: (path, is_package)} for every .py file under a package directory."""
    mods = {}
    for path in package_dir.rglob('*.py'):
        rel = path.relative_to(package_dir.parent).with_suffix('')
        parts = list(rel.parts)
        is_pkg = parts[-1] == '__init__'
        if is_pkg:
            parts = parts[:-1]
        mods['.'.join(parts)] = (path, is_pkg)
    return mods


def _parents(name: str):
    parts = name.split('.')
    return ['.'.join(parts[:i]) for i in range(1, len(parts))]


def imports_of(name: str, path: Path, is_pkg: bool, known: set, root: str) -> tuple[set, set]:
    """Modules of ``root`` imported anywhere in the file (incl. lazy imports) and string references."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    base = name if is_pkg else name.rpartition('.')[0]
    found, strings = set(), set()
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = base
                for _ in range(node.level - 1):
                    pkg = pkg.rpartition('.')[0]
                mod = f'{pkg}.{node.module}' if node.module else pkg
            else:
                mod = node.module or ''
            targets = [mod] + [f'{mod}.{a.name}' for a in node.names if f'{mod}.{a.name}' in known]
        elif isinstance(node, ast.Call) and node.args:
            # importlib.import_module('...') / __import__('...') with a literal module name
            fn = node.func
            fname = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, 'id', '')
            arg = node.args[0]
            if (fname in ('import_module', '__import__') and isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str) and arg.value.startswith(root + '.')):
                strings.add(arg.value)
        for t in targets:
            if t == root or t.startswith(root + '.'):
                found.add(t)
                found.update(p for p in _parents(t) if p.startswith(root))
    return found, strings


def import_graph(package_dir: Path, root: str):
    mods = discover(package_dir, root)
    known = set(mods)
    graph, strings = {}, {}
    for name, (path, is_pkg) in mods.items():
        graph[name], strings[name] = imports_of(name, path, is_pkg, known, root)
        graph[name].update(p for p in _parents(name) if p in known)   # importing a module runs its parents
    return graph, strings


def reachable(graph: dict, starts) -> dict:
    """Reachable module -> import chain from a start module."""
    chain = {s: [s] for s in starts if s in graph}
    todo = list(chain)
    while todo:
        m = todo.pop()
        for n in graph.get(m, ()):
            if n not in chain:
                chain[n] = chain[m] + [n]
                todo.append(n)
    return chain


def runtime_roots(graph: dict) -> list:
    roots = [m for m in graph if any(m == p or m.startswith(p + '.') for p in RUNTIME_PACKAGES)]
    roots += [m for m in RUNTIME_MODULES if m in graph]
    return roots


def violations(graph, strings, roots, label_package):
    chains = reachable(graph, roots)
    bad = {m: c for m, c in chains.items() if m == label_package or m.startswith(label_package + '.')}
    bad_strings = {m: s for m in chains for s in strings.get(m, ()) if s.startswith(label_package)}
    return bad, bad_strings


def test_runtime_modules_cannot_reach_offline_labels():
    graph, strings = import_graph(PACKAGE_DIR, 'haltere')
    roots = runtime_roots(graph)
    assert 'haltere.liftoff.visual_brain' in roots and 'haltere.obstacles.contract' in roots
    assert LABEL_PACKAGE in graph      # the scanner sees the label package
    bad, bad_strings = violations(graph, strings, roots, LABEL_PACKAGE)
    assert not bad, {m: ' -> '.join(c) for m, c in bad.items()}
    assert not bad_strings, bad_strings


def test_scanner_detects_direct_lazy_and_transitive_label_imports(tmp_path):
    pkg = tmp_path / 'fakepkg'
    (pkg / 'obstacles' / 'labels').mkdir(parents=True)
    (pkg / 'liftoff').mkdir()
    files = {
        '__init__.py': '',
        'obstacles/__init__.py': '',
        'obstacles/labels/__init__.py': 'X = 1\n',
        'obstacles/labels/tube.py': 'Y = 2\n',
        'obstacles/helper.py': 'def f():\n    from .labels import tube\n    return tube\n',
        'obstacles/clean.py': 'import numpy\n',
        'liftoff/__init__.py': '',
        'liftoff/runner.py': 'from ..obstacles import helper\n',
        'liftoff/ok.py': 'from ..obstacles import clean\n',
        'liftoff/dyn.py': 'import importlib\ndef f():\n    return importlib.import_module("fakepkg.obstacles.labels.tube")\n',
        'liftoff/named.py': 'NAME = "fakepkg.obstacles.labels"  # a constant is not an import\n',
    }
    for rel, src in files.items():
        (pkg / rel).write_text(textwrap.dedent(src), encoding='utf-8')
    graph, strings = import_graph(pkg, 'fakepkg')
    bad, bad_strings = violations(graph, strings, ['fakepkg.liftoff.runner'], 'fakepkg.obstacles.labels')
    assert 'fakepkg.obstacles.labels.tube' in bad
    assert bad['fakepkg.obstacles.labels.tube'][:2] == ['fakepkg.liftoff.runner', 'fakepkg.obstacles.helper']
    bad, bad_strings = violations(graph, strings, ['fakepkg.liftoff.ok'], 'fakepkg.obstacles.labels')
    assert not bad and not bad_strings
    bad, bad_strings = violations(graph, strings, ['fakepkg.liftoff.dyn'], 'fakepkg.obstacles.labels')
    assert bad_strings
    bad, bad_strings = violations(graph, strings, ['fakepkg.liftoff.named'], 'fakepkg.obstacles.labels')
    assert not bad and not bad_strings


def _python(code, env_extra=None):
    env = dict(os.environ, PYTHONPATH=str(PACKAGE_DIR.parent), OMP_NUM_THREADS='2', **(env_extra or {}))
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=120)


def test_importing_runtime_obstacle_modules_loads_no_label_code():
    code = ('import sys, haltere.obstacles, haltere.obstacles.contract, haltere.obstacles.model, '
            'haltere.obstacles.overlays\n'
            f'print(sorted(m for m in sys.modules if m.startswith("{LABEL_PACKAGE}")))')
    r = _python(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '[]'


def test_label_package_refuses_runtime_processes():
    r = _python(f'import {LABEL_PACKAGE}', {RUNTIME_ENV_FLAG: '1'})
    assert r.returncode != 0
    assert 'must not be imported by a runtime process' in r.stderr
    r = _python(f'import {LABEL_PACKAGE}; print("ok")')
    assert r.returncode == 0 and 'ok' in r.stdout, r.stderr


OFFLINE_CHECKED = {'haltere.obstacles.store', 'haltere.obstacles.splits', 'haltere.obstacles.evaluate',
                   'haltere.obstacles.train', 'haltere.obstacles.timing', 'haltere.obstacles.store_build',
                   'haltere.obstacles.gap_cue_eval', 'haltere.obstacles.leaks', 'haltere.obstacles.thermal',
                   'haltere.obstacles.gap_bench', 'haltere.obstacles.free_space_eval'}
# The free-space planner's pilot side (haltere.liftoff.corridor_aim) is checked once it exists on the branch.
OPTIONAL_RUNTIME = ('haltere.liftoff.corridor_aim',)


@pytest.mark.parametrize('module', ('haltere.obstacles.contract', 'haltere.obstacles.model', 'haltere.obstacles.overlays',
                                    'haltere.vision.gap_cue', 'haltere.vision.relative_depth',
                                    'haltere.vision.free_space',
                                    'haltere.liftoff.gap_stack', 'haltere.liftoff.gap_aim',
                                    'haltere.liftoff.camera_replay', 'haltere.liftoff.camera_process',
                                    'haltere.liftoff.fast_race_cue', 'haltere.liftoff.visual_brain')
                         + OPTIONAL_RUNTIME)
def test_runtime_obstacle_modules_import_only_runtime_safe_code(module):
    graph, _ = import_graph(PACKAGE_DIR, 'haltere')
    if module in OPTIONAL_RUNTIME and module not in graph:
        pytest.skip(f'{module} is not on this branch')
    assert module in graph
    assert OFFLINE_CHECKED <= set(OFFLINE_MODULES)
    chain = reachable(graph, [module])
    assert not set(chain) & OFFLINE_CHECKED
    assert not any(m == LABEL_PACKAGE or m.startswith(LABEL_PACKAGE + '.') for m in chain)


def test_free_space_runtime_modules_load_no_label_torch_or_offline_code():
    """The planner is numpy only at import (OpenCV only when tracks are computed); the depth-process worker and
    the controller-side PLAN_FIELDS reader load no label code, no torch and no offline evaluation or bench."""
    code = ('import sys, haltere.vision.free_space, haltere.liftoff.gap_stack, haltere.liftoff.camera_process\n'
            'from haltere.liftoff.camera_process import PLAN_FIELDS, PLAN_KINDS, plan_sample\n'
            'assert PLAN_FIELDS == haltere.vision.free_space.PLAN_FIELDS\n'
            f'print(sorted(m for m in sys.modules if m.startswith("{LABEL_PACKAGE}") '
            'or m.split(".")[0] in ("torch", "cv2", "transformers") '
            'or m in ("haltere.obstacles.free_space_eval", "haltere.obstacles.gap_cue_eval", '
            '"haltere.obstacles.gap_bench", "haltere.obstacles.store")))')
    r = _python(code, {RUNTIME_ENV_FLAG: '1'})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '[]'


def test_no_runtime_module_reaches_the_free_space_evaluation():
    graph, _ = import_graph(PACKAGE_DIR, 'haltere')
    assert 'haltere.obstacles.free_space_eval' in graph
    chains = reachable(graph, runtime_roots(graph))
    assert 'haltere.obstacles.free_space_eval' not in chains
    assert {'haltere.vision.free_space', 'haltere.liftoff.gap_stack', 'haltere.liftoff.camera_process'} <= set(chains)


def test_free_space_code_names_no_course_or_route_inputs():
    """The planner's runtime modules take no course, route, gate-geometry or label inputs (source scan)."""
    banned = ('course_pool', 'bot_routes', 'collection_route', 'qualification_route', 'challenge_courses',
              'section_geometry', 'gate_memory', 'oracle', 'hindsight', 'tti_s', 'impact')
    for rel in ('vision/free_space.py',):
        src = (PACKAGE_DIR / rel).read_text(encoding='utf-8')
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {a.arg for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) for a in n.args.args + n.args.kwonlyargs}
        imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
            {n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        for b in banned:
            assert not any(b in x for x in names | imports), (rel, b)


def test_gap_cue_runtime_modules_load_no_label_torch_or_offline_code():
    """The gap cue is numpy only; the depth wrapper imports torch only when a model is built."""
    code = ('import sys, haltere.vision.gap_cue, haltere.vision.relative_depth\n'
            f'print(sorted(m for m in sys.modules if m.startswith("{LABEL_PACKAGE}") '
            'or m.split(".")[0] in ("torch", "cv2", "transformers") '
            'or m in ("haltere.obstacles.gap_cue_eval", "haltere.obstacles.store")))')
    r = _python(code, {RUNTIME_ENV_FLAG: '1'})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '[]'


def test_no_runtime_module_reaches_the_gap_cue_evaluation():
    graph, _ = import_graph(PACKAGE_DIR, 'haltere')
    assert 'haltere.obstacles.gap_cue_eval' in graph
    chains = reachable(graph, runtime_roots(graph))
    assert 'haltere.obstacles.gap_cue_eval' not in chains
    assert 'haltere.vision.gap_cue' in chains and 'haltere.vision.relative_depth' in chains


def test_wired_gap_cue_modules_load_no_label_torch_or_offline_code():
    """The flight-side gap modules import numpy only at module level (torch/cv2/transformers only when a
    worker is built); none reaches labels, the gap-cue evaluation or the bench."""
    code = ('import sys, haltere.liftoff.gap_stack, haltere.liftoff.gap_aim, haltere.liftoff.camera_replay\n'
            f'print(sorted(m for m in sys.modules if m.startswith("{LABEL_PACKAGE}") '
            'or m.split(".")[0] in ("torch", "cv2", "transformers") '
            'or m in ("haltere.obstacles.gap_cue_eval", "haltere.obstacles.gap_bench", "haltere.obstacles.store")))')
    r = _python(code, {RUNTIME_ENV_FLAG: '1'})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == '[]'


def test_no_runtime_module_reaches_the_gap_bench():
    graph, _ = import_graph(PACKAGE_DIR, 'haltere')
    assert 'haltere.obstacles.gap_bench' in graph
    chains = reachable(graph, runtime_roots(graph))
    assert 'haltere.obstacles.gap_bench' not in chains
    assert {'haltere.liftoff.gap_stack', 'haltere.liftoff.gap_aim', 'haltere.obstacles.overlays'} <= set(chains)
