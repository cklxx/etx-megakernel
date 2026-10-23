"""Bridge sealed lifecycle requests to the shared library implementation."""
from etx.api.sealed import SealedPreparationRequest, SealedSessionRequest
from etx.api.types import Configuration


def prepare_sealed_session(request):
    if not isinstance(request, SealedPreparationRequest):
        raise TypeError('prepare_sealed_session requires SealedPreparationRequest')
    from etx.experiments.sealed_stage import prepare
    return Configuration(prepare(request.inventory.to_dict(), request.destination, request.output))


def run_sealed_session(request, *, render=False):
    if not isinstance(request, SealedSessionRequest):
        raise TypeError('run_sealed_session requires SealedSessionRequest')
    if type(render) is not bool:
        raise ValueError('render must be a boolean')
    from etx.experiments.sealed import run
    result = run(request)
    if render:
        import json
        if request.action == 'export':
            print(result['sha256'])
        else:
            print(json.dumps(result['records'] if request.action == 'list' else result, indent=2))
    return Configuration(result)
