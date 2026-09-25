"""Read-only, machine-wide workload check before a measured flight.

Checks every Windows session, not just Anode. Known training/benchmark commands
block even while waiting for work; other busy compute processes block on CPU
use; a running game blocks because the virtual gamepad is machine-wide (the pad
bridge also unplugs itself when one starts, see game_guard), unless Anode keeps
the seat's pads inside it, where the user's game cannot read them; the report
then lists the game instead. This is a snapshot, not an operating-system
reservation. Recheck after the flight and keep runtime failures separate from
navigation outcomes.
"""
from __future__ import annotations

import json
import ntpath
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

from .game_guard import anode_isolation, game_processes, library_games, own_session, pads_stay_in_seat


_INVENTORY = r"""
Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,SessionId,Name,ExecutablePath,CommandLine,UserModeTime,KernelModeTime | ConvertTo-Json -Compress
"""
_COMPUTE = re.compile(r'^(python(?:w|\d+(?:\.\d+)?)?|pypy\d*|pytest|torchrun|accelerate|uv|node|java|dotnet|ffmpeg|cargo|rustc|cmake|ninja|msbuild|wsl|docker)(?:\.exe)?$', re.I)
_WORK = re.compile(r'(?:^|[\\/._-])(train(?:ing)?|bench(?:mark)?s?|pytest|evaluate|evaluation|quality-measurement|bptt)(?:$|[\\/._-])', re.I)


def workload_reason(process):
    """Classify executable/argument tokens; never match a shell's script text."""
    name = process.get('Name', '')
    if not _COMPUTE.fullmatch(name):
        return None
    command = process.get('CommandLine') or ''
    tokens = [a or b for a, b in re.findall(r'"([^"]*)"|(\S+)', command)]
    # The executable's parent directory can contain 'training' incidentally.
    for token in [Path(name).stem, *tokens[1:]]:
        if _WORK.search(token) or token.lower() in {'train', 'test', 'bench', 'benchmark'}:
            return 'training, evaluation, benchmark or test command'
        if token.lower() == 'haltere.liftoff.visual_brain':
            return 'another visual flight controller'
    return None


def inventory():
    if sys.platform != 'win32':
        raise RuntimeError('Flight workload inventory requires Windows; no unchecked fallback')
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', _INVENTORY],
                            capture_output=True, text=True, timeout=20, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    data = json.loads(result.stdout)
    return data if isinstance(data, list) else [data]


def classify(before, after, elapsed, own_pid):
    """Return safe-to-publish blockers without recording unrelated arguments."""
    by_pid = {int(p['ProcessId']): p for p in after}
    excluded = set()
    pid = own_pid
    while pid in by_pid and pid not in excluded:
        excluded.add(pid)
        pid = int(by_pid[pid].get('ParentProcessId', 0))
    prior = {int(p['ProcessId']): p for p in before}
    blockers = []
    for pid, process in by_pid.items():
        if pid in excluded:
            continue
        reason = workload_reason(process)
        cpu_cores = None
        if pid in prior and elapsed > 0:
            def ticks(p):
                return sum(int(p.get(key) or 0) for key in ('UserModeTime', 'KernelModeTime'))
            cpu_cores = max(0., (ticks(process)-ticks(prior[pid]))/1e7/elapsed)
        if not reason and _COMPUTE.fullmatch(process.get('Name', '')) and cpu_cores is not None and cpu_cores >= .5:
            reason = 'other compute process using at least half a CPU core'
        if reason:
            blockers.append(dict(pid=pid, session=process.get('SessionId'), executable=process['Name'],
                                 reason=reason, cpu_cores=cpu_cores))
    return blockers


def project_workload_pids(processes, allowed_projects):
    """Resolve explicitly authorized projects from absolute Python script paths.

    A venv executable, output path, project-name substring or shell script text
    is insufficient. Include only connected compute ancestors/descendants of a
    matching script, stopping at shell/service boundaries. Resolve each snapshot
    anew so PID rotation cannot silently expand an old numeric exception.
    """
    roots=[]
    for value in allowed_projects:
        root=ntpath.normcase(ntpath.normpath(str(value))).rstrip('\\')
        if not ntpath.isabs(root) or not ntpath.splitdrive(root)[0] or len(root)<=3:
            raise ValueError('Workload project exceptions require absolute Windows project directories')
        roots.append(root)
    by_pid={int(p['ProcessId']):p for p in processes}
    authorized={}
    for pid,p in by_pid.items():
        if not re.fullmatch(r'(python(?:w|\d+(?:\.\d+)?)?|pypy\d*)(?:\.exe)?',p.get('Name',''),re.I):
            continue
        tokens=[a or b for a,b in re.findall(r'"([^"]*)"|(\S+)',p.get('CommandLine') or '')]
        # Python flags such as -c/-m/-X can contain arbitrary path-like text;
        # only a directly executed script is used as the identity seed.
        if len(tokens)<2 or tokens[1].startswith('-'):
            continue
        script=ntpath.normcase(ntpath.normpath(tokens[1]))
        if ntpath.isabs(script) and script.endswith(('.py','.pyw')):
            for root in roots:
                if script.startswith(root+'\\'):
                    authorized[pid]=root
                    break
    seeds=dict(authorized)
    for pid,root in seeds.items():
        seen={pid}
        parent=int(by_pid[pid].get('ParentProcessId',0))
        while parent in by_pid and parent not in seen and _COMPUTE.fullmatch(by_pid[parent].get('Name','')):
            seen.add(parent);authorized[parent]=root
            parent=int(by_pid[parent].get('ParentProcessId',0))
    changed=True
    while changed:
        changed=False
        for pid,p in by_pid.items():
            parent=int(p.get('ParentProcessId',0))
            if pid not in authorized and parent in authorized and _COMPUTE.fullmatch(p.get('Name','')):
                authorized[pid]=authorized[parent];changed=True
    return authorized


def check_workloads(allowed_pids=(), allowed_projects=()):
    start = time.monotonic()
    before = inventory()
    time.sleep(1.)
    after = inventory()
    elapsed = time.monotonic()-start
    blockers = classify(before, after, elapsed, os.getpid())
    own = own_session()
    games = game_processes(after, own)+library_games(after, own)
    # Anode keeping the pads in its seat means a game on the desktop cannot read the flight's commands.
    seat_only = pads_stay_in_seat(anode_isolation())
    if not seat_only:
        blockers += [dict(g, reason=f"game running ({g['reason']}); the virtual gamepad would reach it", cpu_cores=None)
                     for g in games]
    project_pids=project_workload_pids(after,allowed_projects)
    # Explicit operator exceptions still fail if their measured CPU work rises.
    # Keep both the permission and observed load visible in the flight record.
    exceptions = [b for b in blockers if (b['pid'] in allowed_pids or b['pid'] in project_pids)
                  and b['cpu_cores'] is not None and b['cpu_cores'] < .5]
    blockers = [b for b in blockers if b not in exceptions]
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), passed=not blockers,
                blockers=blockers, operator_exception_pids=list(allowed_pids), exceptions=exceptions,
                pads_seat_only=seat_only, games_beside_seat_only_pads=games if seat_only else [],
                operator_exception_projects=list(allowed_projects),
                resolved_project_workloads=[dict(pid=pid,project=root) for pid,root in sorted(project_pids.items())],
                scope='all Windows sessions; known jobs, busy compute executables and games',
                limitation='snapshot; does not reserve resources or identify all possible GPU jobs')


def require_quiet(log_path, allowed_pids=(), allowed_projects=()):
    """Persist a refusal before opening capture, pads or the flight CSV."""
    path = Path(log_path).with_suffix('.preflight.json')
    if path.exists():
        raise FileExistsError(f'Preserve the previous preflight record; use a new log path: {path}')
    report = check_workloads(allowed_pids,allowed_projects)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        pids = ', '.join(str(p['pid']) for p in report['blockers'])
        raise RuntimeError(f'Preflight refused: competing workloads (PIDs {pids}); see {path}')
    return report


if __name__ == '__main__':
    print(json.dumps(check_workloads(), indent=2))
