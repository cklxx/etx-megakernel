"""Package-owned native inputs for registered operators, not linked builds."""
import json
from pathlib import Path


NATIVE_INPUTS = 'etx_native_inputs.json'


def _units(request):
    name = request.workload.config.get('provider')
    resource = {'splitk_sum': 'splitk.hip', 'moe_layer': 'moe.hip', 'gemm_rs': 'gemm_rs.hip'}.get(name)
    if request.workload.kind != 'graph' or resource is None:
        return ()
    defines = {}
    if name == 'moe_layer':
        from etx.operators.moe_layer import provider
        dimensions = provider(small=request.workload.config.get('small', False))
        defines = {key: getattr(dimensions, key) for key in ('D', 'H', 'S', 'FF')}
    return (dict(source='examples/tiles/' + resource, language='hip', defines=defines),)


def _recipe(units):
    return json.dumps(dict(schema_version=1, kind='native-tile-inputs', units=units),
                      sort_keys=True, indent=2) + '\n'


def operator_sources(request, dependencies):
    """Capture known provider inputs without resolving arbitrary checkout paths."""
    units = _units(request)
    if not units:
        return {}, tuple(dependencies)
    if set(dependencies) != {unit['source'] for unit in units}:
        raise ValueError('registered operator native dependency inventory changed')
    resources = Path(__file__).resolve().parents[1] / 'operators' / 'tiles'
    files = {unit['source']: (resources / Path(unit['source']).name).read_text() for unit in units}
    files[NATIVE_INPUTS] = _recipe(units)
    for unit in units:
        host = unit['source'].replace('/tiles/', '/hosts/')
        files[host] = (resources.parent / 'hosts' / Path(host).name).read_text()
    return files, ()


def checked_native_units(artifact):
    """Validate the recorded recipe without reading current package tile bodies."""
    units = _units(artifact.planned.request)
    if not units:
        if NATIVE_INPUTS in artifact.files:
            raise ValueError('native inputs require a registered operator')
        return ()
    if (artifact.files.get(NATIVE_INPUTS) != _recipe(units) or
            any(unit['source'] not in artifact.files for unit in units)):
        raise ValueError('native inputs disagree with the provider declaration')
    return units


def checked_host(artifact):
    units = checked_native_units(artifact)
    expected_ranks = 2 if artifact.planned.request.workload.config.get('provider') == 'gemm_rs' else 1
    if len(units) != 1 or len(artifact.rank_sources) != expected_ranks:
        raise ValueError('host linking requires the registered operator rank inventory')
    host = units[0]['source'].replace('/tiles/', '/hosts/')
    if host not in artifact.files:
        raise ValueError('missing captured operator host')
    request = artifact.planned.request
    if expected_ranks > 1 and request.options.get('inline_tiles', False):
        raise ValueError('multi-rank inline tile definitions cannot be linked as separate translation units')
    if (request.workload.config.get('provider') == 'moe_layer' and
            request.bindings.get('seed', 12345) != 12345):
        raise ValueError('legacy MoE host requires seed 12345')
    return host, units
