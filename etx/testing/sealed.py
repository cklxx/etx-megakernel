"""Bounded sealed-session probes; synthetic CPU work is never device evidence."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from etx.api.types import Configuration
from etx.experiments import sealed


@contextmanager
def synthetic_sealed_stages(*, write_decision=True, exit_code=0, forbid=False):
    """Replace stage dispatch only, retaining real bounded child/log execution.

    The fixed synthetic decision never admits a device result. No native build,
    correctness, certificate, or performance evidence is manufactured here.
    ContextVar tokens restore both nested and exceptional uses of this seam.
    """
    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        raise ValueError('synthetic exit code must be in 0..255')
    calls = []

    def run(argv, log, *, seconds):
        if forbid:
            raise AssertionError('must not spawn')
        calls.append(list(argv))
        log = Path(log)
        if '-preflight.log' in log.name:
            script = "import sys,json; print(json.dumps({'status':'PASS','synthetic_cpu_only':True}),flush=True); "
        else:
            script = "import sys; print('SYNTHETIC_CPU_ONLY stdout',flush=True); print('SYNTHETIC_CPU_ONLY stderr',file=sys.stderr,flush=True); "
        if write_decision:
            record = log.name.split('-')[0]
            decision = log.parent / (record + '.json')
            script += ('from pathlib import Path; import json; Path(' + repr(str(decision)) +
                       ").write_text(json.dumps({'synthetic_cpu_only':True,'admitted':False,'complete':True})); ")
        script += 'sys.exit(' + str(exit_code) + ')'
        return sealed.run_logged([sys.executable, '-u', '-c', script], log,
                                 seconds=min(seconds, 2))

    token = sealed._stage_runner.set(run)
    try:
        yield calls
    finally:
        sealed._stage_runner.reset(token)


@contextmanager
def interrupted_sealed_export():
    """Inject one archive-open fault without replacing the export lifecycle."""
    with patch.object(sealed.tarfile, 'open',
                      side_effect=OSError('SYNTHETIC interrupted export')):
        yield


def sealed_child_probe(kind, *, log, marker=None):
    """Run a fixed CPU child through production logging/deadline handling."""
    if kind == 'raw-stdout-stderr':
        script = 'import sys;print("raw stdout");print("raw stderr",file=sys.stderr)'
        seconds = 3
    elif kind == 'exit7':
        script = 'raise SystemExit(7)'
        seconds = 3
    elif kind == 'short-deadline':
        script = 'import time;time.sleep(2)'
        seconds = .02
    elif kind == 'stdout-stderr':
        script = "import sys;print('SYNTHETIC stdout');print('SYNTHETIC stderr',file=sys.stderr)"
        seconds = 2
    elif kind == 'deadline':
        if marker is None:
            raise ValueError('deadline probe requires a caller-owned marker')
        grandchild = ("import time;from pathlib import Path;time.sleep(.3);Path(" +
                      repr(str(marker)) + ").write_text('SYNTHETIC orphan')")
        script = ('import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",' +
                  repr(grandchild) + ']);print("SYNTHETIC CPU deadline child",flush=True);time.sleep(10)')
        seconds = .08
    else:
        raise ValueError('unknown sealed CPU child probe')
    return sealed.run_logged([sys.executable, '-u', '-c', script], Path(log), seconds=seconds)


def sealed_cleanup_interrupt_probe(*, log, marker):
    """Deliver SIGTERM at real cleanup entry; a bounded CPU grandchild must die.

    The signal is injected in a separate worker, never into the test caller.
    The child and grandchild are synthetic and terminate on their own even if
    the regression prevents cleanup. Only the sweep entry is instrumented;
    descendant discovery and killing still use the production implementation.
    """
    import tempfile
    import time

    log = Path(log).resolve()
    marker = Path(marker).resolve()
    if marker.exists():
        raise ValueError('cleanup probe marker must be unused')
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('xb'):
        pass
    grandchild = ("import time;from pathlib import Path;"
                  "print('SYNTHETIC ready',flush=True);time.sleep(.3);Path(" +
                  repr(str(marker)) + ").write_text('SYNTHETIC orphan')")
    child = ('import subprocess,sys;p=subprocess.Popen([sys.executable,"-c",' +
             repr(grandchild) + '],stdout=subprocess.PIPE);'
             'assert p.stdout.readline()==b"SYNTHETIC ready\\n"')
    script = '''
import os, signal, sys
sys.path.insert(0, sys.argv[1])
from etx.contracts import log_worker
from etx.session import proc
original_sweep = proc.sweep
def interrupt_at_sweep(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGTERM)
    return original_sweep(*args, **kwargs)
proc.sweep = interrupt_at_sweep
log_worker.main(sys.argv[2], os.getppid())
'''
    with tempfile.TemporaryDirectory(prefix='etx-sealed-cleanup-') as directory:
        root = Path(directory)
        (root / 'request.json').write_text(json.dumps(dict(
            argv=[sys.executable, '-c', child], log=str(log), seconds=2)))
        subprocess.run([sys.executable, '-I', '-c', script, str(sealed.ROOT), directory],
                       capture_output=True, text=True, timeout=5, check=True)
        response = json.loads((root / 'response.json').read_text())
        time.sleep(.4)
        return Configuration(dict(response=response, descendant_survived=marker.exists()))


def sealed_contract_probe(kind, *, value=None, output=None, record=None, path=None):
    """Pure binding/order and fail-closed evidence seams; no stage dispatch."""
    if kind == 'routing-decode':
        from etx.experiments.sealed_routing import decode_records
        return Configuration(decode_records(value['raw'], batch=value['batch']))
    if kind == 'routing-compatibility':
        from etx.experiments import sealed_routing as provider
        from examples.experiment_program import routing
        from examples.experiment_program.routing import __main__ as cli
        return Configuration(dict(decode=routing.decode_records is provider.decode_records,
                                  roles=cli.ROLES is provider.ROLES,
                                  config=cli.config is provider.config,
                                  gates=cli.gates is provider.gates))
    if kind == 'routing-column-dependency':
        from examples.experiment_program.columns.runner import verify_dependency
        return Configuration(verify_dependency(path))
    if kind == 'routing-isolated-decision':
        # Cold import with examples explicitly unavailable proves this evidence
        # path no longer reaches a CLI, even indirectly. No stage is dispatched.
        script = '''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class NoExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'examples' or fullname.startswith('examples.'):
            raise ImportError('example import forbidden: ' + fullname)
sys.meta_path.insert(0, NoExamples())
from etx.experiments.sealed import decision_for_dependencies
print(json.dumps(decision_for_dependencies('G04', json.loads(sys.argv[2]), sys.argv[3])))
'''
        result = subprocess.run([sys.executable, '-I', '-c', script, str(sealed.ROOT),
                                 json.dumps(value), str(path)],
                                capture_output=True, text=True, timeout=10, check=True)
        return Configuration(json.loads(result.stdout))
    if kind == 'binding':
        return Configuration(sealed.bind_output(value, output))
    if kind == 'order':
        return Configuration({'order': sealed.execution_order(value)})
    if kind == 'decision':
        if record not in sealed.MODULES:
            raise ValueError('unknown sealed record')
        return Configuration(sealed.decision_for_dependencies(record, value, path))
    if kind == 'source-scope':
        from etx.artifacts.sealed_source import snapshot, ROOT
        return Configuration(dict(root=str(ROOT), sources=snapshot()))
    raise ValueError('unknown sealed contract probe')


def sealed_resume_probe(state, config_sha, revision):
    """Exercise identity, sealed-byte and incomplete-state resume gates only."""
    sealed.verify_resume(state, config_sha, revision)


def sealed_dependency_probe(record, decisions, options):
    """Exercise a registered dependency gate without executing any stage."""
    if record not in sealed.MODULES:
        raise ValueError('unknown sealed record')
    return sealed.eligible(record, decisions, options)


def sealed_cli_list_probe():
    """Run only the real disabled-record CLI listing with a bounded CPU child."""
    result = subprocess.run([sys.executable, '-m', 'examples.experiment_program', 'list'],
                            cwd=sealed.ROOT, text=True, capture_output=True,
                            timeout=10, check=True)
    return Configuration({'records': json.loads(result.stdout)})


def sealed_provenance_probe(sources, *, root, revision=None):
    """Inspect provenance against an explicit non-Git fixture root only."""
    from etx.artifacts.sealed import source_provenance
    root = Path(root)
    if not root.is_dir() or (root / '.git').exists():
        raise ValueError('provenance probe requires a non-Git fixture directory')
    return Configuration(source_provenance(sources, revision, root=root))


def sealed_historical_replay_probe():
    """Replay only the fixed retained record; never claim fresh performance."""
    from examples.experiment_program.local_acceptance import historical_replay
    return Configuration(historical_replay())
