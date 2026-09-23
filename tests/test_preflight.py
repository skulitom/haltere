from haltere.liftoff.preflight import classify, workload_reason


def process(pid, command, name='python.exe', parent=0, session=1, ticks=0):
    return dict(ProcessId=pid, ParentProcessId=parent, SessionId=session, Name=name,
                CommandLine=command, UserModeTime=ticks, KernelModeTime=0)


def test_cross_session_training_and_benchmarks_block_even_when_idle():
    rows = [process(1, 'python -m haltere.liftoff.visual_brain candidate.pt', session=5),
            process(2, 'python -m haltere.train.motor_tracking', session=1),
            process(3, 'python research/quality-measurement/study/rebuy.py run', session=2),
            process(4, 'python runs/motor10/benchmark.py', session=3)]
    assert {b['pid'] for b in classify(rows, rows, 1., 1)} == {2, 3, 4}


def test_services_and_shell_command_text_do_not_false_positive():
    for row in [process(1, 'python litharness-mcp.exe --profile read'),
                process(2, 'python -m http.server 4173'),
                process(3, 'powershell -Command "Get-Content training.py"', name='powershell.exe'),
                process(4, '"C:/training-tools/python.exe" -m http.server')]:
        assert workload_reason(row) is None


def test_busy_unknown_compute_blocks_and_own_launcher_is_excluded():
    before = [process(1, 'python -m haltere.liftoff.visual_brain model.pt', parent=2),
              process(2, 'python -m haltere.liftoff.visual_brain model.pt'),
              process(3, 'python mystery.py'), process(4, 'python idle.py')]
    after = [dict(p, UserModeTime=8_000_000 if p['ProcessId'] == 3 else 0) for p in before]
    blockers = classify(before, after, 1., 1)
    assert [b['pid'] for b in blockers] == [3]
    assert 'CommandLine' not in blockers[0]


def test_operator_exception_is_recorded_but_cannot_exempt_busy_job(monkeypatch):
    from haltere.liftoff import preflight
    rows = [process(900001, 'python benchmark.py')]
    monkeypatch.setattr(preflight, 'inventory', lambda: rows)
    monkeypatch.setattr(preflight.time, 'sleep', lambda _: None)
    report = preflight.check_workloads([900001])
    assert report['passed'] and report['exceptions'][0]['pid'] == 900001
    assert report['operator_exception_pids'] == [900001]
    monkeypatch.setattr(preflight, 'classify', lambda *args: [dict(pid=900001, cpu_cores=1.)])
    report = preflight.check_workloads([900001])
    assert not report['passed'] and not report['exceptions']


def test_project_authorization_tracks_only_verified_script_compute_family():
    from haltere.liftoff.preflight import project_workload_pids
    rows=[process(10,'powershell run.ps1',name='powershell.exe'),
          process(11,'uv run python research/benchmark.py',name='uv.exe',parent=10),
          process(12,'"C:/DEV/LitHarness/.venv/python.exe" research/benchmark.py',parent=11),
          process(13,'python C:/DEV/LitHarness/research/benchmark.py',parent=12),
          process(14,'python worker.py',parent=13),
          process(15,'python C:/DEV/LitHarness-other/benchmark.py'),
          process(16,'python C:/DEV/LitHarness/../Haltere/benchmark.py'),
          process(17,'python other.py --output C:/DEV/LitHarness/report.py'),
          process(18,'"C:/DEV/LitHarness/python.exe" other.py'),
          process(19,'python -c "C:/DEV/LitHarness/benchmark.py"'),
          process(20,'powershell C:/DEV/LitHarness/script.py',name='powershell.exe',parent=13),
          process(21,'python other.py',parent=20)]
    assert set(project_workload_pids(rows,['C:/DEV/LitHarness']))=={11,12,13,14}
    assert project_workload_pids(rows,[])=={}


def test_project_authorization_still_rejects_busy_or_unknown_load(monkeypatch):
    from haltere.liftoff import preflight
    rows=[process(900001,'python C:/DEV/LitHarness/research/benchmark.py')]
    monkeypatch.setattr(preflight,'inventory',lambda:rows)
    monkeypatch.setattr(preflight.time,'sleep',lambda _:None)
    result=preflight.check_workloads(allowed_projects=['C:/DEV/LitHarness'])
    assert result['passed'] and result['resolved_project_workloads'][0]['pid']==900001
    for load in [None,.5,1.]:
        monkeypatch.setattr(preflight,'classify',lambda *args:[dict(pid=900001,cpu_cores=load)])
        result=preflight.check_workloads(allowed_projects=['C:/DEV/LitHarness'])
        assert not result['passed'] and not result['exceptions']
