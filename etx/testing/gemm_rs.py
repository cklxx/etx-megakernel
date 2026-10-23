"""CPU-only GEMM/Reduce-Scatter provider conformance; no device qualification."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from etx.api.types import Configuration


def _encoded(value):
    from etx.passes.plan import PassOptions, recorded_options

    def normal(item):
        if isinstance(item, PassOptions):
            # Options as records store them: default-off emission switches are absent, as before they existed.
            return normal(recorded_options(item))
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            return {field.name: normal(getattr(item, field.name)) for field in dataclasses.fields(item)}
        if isinstance(item, dict):
            return {str(key): normal(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [normal(value) for value in item]
        if isinstance(item, (set, frozenset)):
            return sorted(normal(value) for value in item)
        return item
    return json.dumps(normal(value), sort_keys=True, default=str, indent=2).encode()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def provider_conformance(*, kind='library'):
    """Fingerprint declarations and 16 full plans/source sets against retained bytes.

    ``isolated`` additionally forbids importing examples in a fresh interpreter.
    The result exposes immutable signatures, not compiler objects. Ambient pass
    options remain ambient, just as they are for the original graph provider.
    """
    if kind not in ('library', 'legacy', 'isolated'):
        raise ValueError('unknown GEMM provider probe')
    root = Path(__file__).resolve().parents[2]
    if kind == 'isolated':
        script = '''
import importlib.abc, json, sys
sys.path.insert(0, sys.argv[1])
class NoExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'examples' or fullname.startswith('examples.'):
            raise ImportError('example import forbidden: ' + fullname)
sys.meta_path.insert(0, NoExamples())
from etx.testing.gemm_rs import provider_conformance
print(json.dumps(provider_conformance().to_dict()))
'''
        result = subprocess.run([sys.executable, '-I', '-c', script, str(root)],
                                capture_output=True, text=True, timeout=60, check=True)
        return Configuration(json.loads(result.stdout))
    if kind == 'legacy':
        spec = importlib.util.spec_from_file_location('gemm_provider_probe', root / 'examples/gemm_rs.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        from etx.operators import gemm_rs as mod

    from etx.contracts.emission import emit_payloads
    from etx.frontends import collect_sources
    from etx.passes import compile_graph, PassOptions

    signature = dict(constants={name: getattr(mod, name) for name in ('T', 'BM', 'BN')},
                     bindings=[mod.bindings(), mod.bindings(M=256, K=128, N=256, extra=7)],
                     runtime=[mod.runtime(None), mod.runtime({}, object())], graphs={}, plans={})
    # Preserve even the historical world!=2 declarations, not a new validity policy.
    for world in (0, 1, 2, 3):
        signature['graphs'][str(world)] = _digest(_encoded(mod.build(world)))
    for arch in ('gfx90a', 'gfx942'):
        for size, overrides in (('default', {}), ('small', dict(M=256, K=128, N=256))):
            bindings = mod.bindings(**overrides)
            for mode in (None, 'static', 'dynamic', 'hybrid'):
                name = f'{arch}-{size}-{mode or "auto"}'
                plan = compile_graph(mod.build(), arch, bindings, mod.runtime(bindings),
                                     PassOptions(force_mode=mode))
                files, _, _ = emit_payloads(plan, runtime_headers=False)
                hashes = {path: _digest(text.encode()) for path, text in files.items()}
                hashes['plan-full.json'] = _digest(_encoded(plan))
                signature['plans'][name] = dict(files=hashes, sources=collect_sources(plan.graph))
    return Configuration(dict(backend='cpu', plans=len(signature['plans']),
                              signature=_digest(_encoded(signature)), device_qualification=False))
