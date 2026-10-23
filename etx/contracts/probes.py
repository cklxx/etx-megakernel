"""Native probe dispatch: calibration and the fresh-handoff mechanism probe."""
import json
from pathlib import Path

from etx.api.probes import NativeProbeRequest


def run_native_probe(request):
    if not isinstance(request, NativeProbeRequest):
        raise TypeError('run_native_probe requires NativeProbeRequest')
    a = request.arguments.to_dict()
    if request.probe == 'calibration':
        from etx.experiments import calibration
        return calibration.run(request.root, arch=a.get('arch', 'gfx942'), output=a.get('output'),
                               compiler=a.get('compiler', 'hipcc'))
    from etx.experiments import fresh_handoff as probe
    if request.operation == 'plan':
        print(json.dumps(probe.matrix(), indent=2))
        return 0
    if request.operation == 'run':
        out = Path(a['out'])
        probe.run(Path(a['binary']), out, a['budget_seconds'], a['case_seconds'], Path(request.root))
        result = probe.analyze(out)
        (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        return 0 if result['mechanism_gate'] == 'pass' else 1
    result = probe.analyze(Path(a['directory']))
    print(json.dumps(result, indent=2))
    return 0 if result['mechanism_gate'] == 'pass' else 1
