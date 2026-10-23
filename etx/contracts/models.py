"""Model planning provider: a declaration file, loaded by explicit path with explicit configuration.

A model declaration is a Python module exposing build(), bindings(**overrides) and runtime(bindings),
the protocol `etx compile <graph.py>` already uses. Here it is reached only through a
WorkloadSpec(kind='model'), so callers never import model code or rely on ambient state:

- `declaration` is an absolute path and `declaration_sha256` its content hash, so the plan's
  identity covers the declaration bytes and a changed file is refused rather than silently planned;
- `environment` is the complete ETX_* configuration the declaration reads; ambient ETX_* variables
  are removed while it loads and builds, then restored;
- modules first imported while loading are dropped afterwards, so one request cannot leak cached
  configuration into the next.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import threading


_LOCK = threading.Lock()
FIELDS = {'declaration', 'declaration_sha256', 'environment'}


def model_inputs(request):
    """Return (graph, bindings, runtime) for a kind='model' PlanRequest."""
    config = request.workload.config.to_dict()
    unknown = set(config) - FIELDS
    if unknown:
        raise ValueError(f'unknown model configuration fields: {sorted(unknown)}')
    path = config.get('declaration')
    if type(path) is not str or not Path(path).is_absolute():
        raise ValueError('model declaration must be an absolute path')
    path = Path(path)
    if not path.is_file() or path.suffix != '.py':
        raise ValueError(f'model declaration is not a Python file: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if config.get('declaration_sha256') != digest:
        raise ValueError(f'model declaration {path} has sha256 {digest}, not the requested '
                         f'{config.get("declaration_sha256")!r}')
    environment = config.get('environment', {})
    if any(type(key) is not str or not key.startswith('ETX_') or type(value) is not str
           for key, value in environment.items()):
        raise ValueError('model environment takes ETX_* string variables only')
    with _LOCK:
        saved = dict(os.environ)
        loaded = set(sys.modules)
        added_path = str(_repository(path))
        try:
            for key in [key for key in os.environ if key.startswith('ETX_')]:
                del os.environ[key]
            os.environ.update(environment)
            # Declarations import their own package (examples.<model>) from the repository root.
            sys.path.insert(0, added_path)
            module = _load(path)
            graph = module.build()
            bindings = module.bindings(**request.bindings.to_dict())
            runtime = module.runtime(bindings)
            if request.runtime.to_dict():
                raise ValueError('model runtime state comes from the declaration, not the request')
        finally:
            sys.path.remove(added_path)
            for name in set(sys.modules) - loaded:
                del sys.modules[name]
            os.environ.clear()
            os.environ.update(saved)
    return graph, bindings, runtime


def _repository(path):
    # examples/<model>/<file>.py -> the directory holding `examples`; otherwise the file's directory.
    for parent in path.parents:
        if (parent / 'examples').is_dir() and path.is_relative_to(parent / 'examples'):
            return parent
    return path.parent


def _load(path):
    name = 'etx_model_declaration_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Ambient variables a compiler legitimately needs; everything else stays out of build receipts.
COMPILER_ENVIRONMENT = ('PATH', 'HOME', 'TMPDIR', 'LD_LIBRARY_PATH', 'LANG', 'LC_ALL')
COMPILER_PREFIXES = ('ROCM_', 'HIP_', 'HIPCC_', 'HCC_', 'HSA_', 'AMD_', 'LLVM_', 'CLANG_', 'CCACHE_')


def compiler_command(compiler, argv, root, out, readable=()):
    """(command, environment, container) running `argv` (argv[0] the tool name) with a host tool or a local image.

    With 'docker:<image>' the repository (read-only), `out` (writable) and `readable` directories are mounted at
    their own absolute paths, so command lines and outputs match the host compiler's.
    """
    import shutil
    import uuid
    environment = {key: value for key, value in os.environ.items()
                   if key in COMPILER_ENVIRONMENT or key.startswith(COMPILER_PREFIXES)}
    if compiler.startswith('docker:'):
        docker = shutil.which('docker')
        if docker is None:
            raise ValueError('docker is not installed')
        container = 'etx-check-' + uuid.uuid4().hex
        mounts = ['-v', f'{root}:{root}:ro', '-v', f'{out}:{out}']
        for path in readable:
            mounts += ['-v', f'{path}:{path}:ro']
        command = [docker, 'run', '--name', container, '--pull=never', '--network=none',
                   '--user', f'{os.getuid()}:{os.getgid()}', *mounts, '-w', str(root), compiler[len('docker:'):], *argv]
        return command, {key: value for key, value in environment.items() if key in ('PATH', 'HOME')}, container
    tool = shutil.which(compiler if compiler != 'hipcc' else argv[0])
    if tool is None:
        raise ValueError(f'compiler {compiler!r} not found on PATH')
    return [tool, *argv[1:]], environment, None


class _Steps:
    """Ordered build steps on the shared executor; the first failure stops the build with its exit code."""

    def __init__(self, request, logs):
        self.request, self.logs, self.records, self.failed = request, logs, [], None

    def python(self, name, args, *, exit_code):
        """Run a declaration module (`python -m examples...`) from the repository root, as the legacy shells did."""
        import sys
        environment = {key: value for key, value in os.environ.items()
                       if key in ('PATH', 'HOME', 'TMPDIR') or key.startswith('ETX_')}
        return self.process(name, (sys.executable, *args), environment, exit_code=exit_code)

    def process(self, name, argv, environment, *, exit_code=None, container=None):
        from etx.api.execution import ProcessRequest
        from etx.contracts.execution import execute
        process = ProcessRequest(tuple(argv), self.request.root, self.request.deadline_s,
                                 environment=environment, silence_s=self.request.deadline_s)
        try:
            result = execute(process, output=self.logs / name)
        finally:
            if container is not None:
                removed = execute(ProcessRequest((argv[0], 'rm', '--force', container), self.request.root, 30,
                                                 environment=environment), output=self.logs / (name + '-cleanup'))
                if removed.status != 'complete':
                    raise RuntimeError(f'compiler container cleanup not confirmed; see {self.logs}')
        code = 0 if result.status == 'complete' else (exit_code if exit_code is not None else result.returncode or 1)
        self.records.append(dict(step=name, kind='process', argv=list(argv), status=result.status,
                                 returncode=result.returncode, exit=code, log=str(self.logs / name / 'output.log')))
        if code and self.failed is None:
            self.failed = self.records[-1]
        return result

    def text(self, name, text):
        """An in-process step (planning) whose console output the translator may print."""
        path = self.logs / name / 'output.log'
        path.parent.mkdir(parents=True)
        path.write_text(text + '\n')
        self.records.append(dict(step=name, kind='text', status='complete', exit=0, log=str(path)))


def build_model(request):
    """Prepare, plan, gate and link a model build in ordered, logged steps; never executes `run`."""
    import json
    import shutil
    import uuid
    from etx.api.models import ModelBuildRequest
    from etx.api.types import Configuration
    from etx.contracts.compiler import collect_sources
    from etx.contracts.emission import artifact_from, emit_payloads
    from etx.contracts.planning import compile_request, snapshot
    if not isinstance(request, ModelBuildRequest):
        raise TypeError('build_model requires ModelBuildRequest')
    provider = _provider(request.provider)
    root, out = Path(request.root), Path(request.output)
    if not root.is_dir() or not Path(request.host).is_file():
        raise ValueError('model build root or host source is missing')
    # Refusals that need no output happen before anything is created.
    request = provider.refine(request)
    out.mkdir(parents=True, exist_ok=True)
    steps = _Steps(request, out / 'build-logs' / uuid.uuid4().hex)
    planned = artifact = None
    request = provider.prepare(request, steps)
    if steps.failed is None:
        compiled = compile_request(request.plan)
        planned = snapshot(request.plan, compiled)
        artifact = artifact_from(planned, compiled)
        # The historical `etx compile` inventory, rewritten in place like the legacy builds.
        files, sources, _ = emit_payloads(compiled, all_rank_headers=False, runtime_headers=False)
        for name, text in files.items():
            (out / name).write_text(text, encoding='utf-8')
        steps.text('plan', '\n'.join([compiled.summary(), f'sources to link: {collect_sources(compiled.graph)}',
                                      f'wrote {request.output}/']))
        provider.check(request, steps)
    if steps.failed is None:
        # Paths reach the compiler exactly as requested (no normalisation): device code objects depend on them.
        argv = ['hipcc', *request.flags, *(arg for path in request.includes for arg in ('-I', path))]
        if request.resource_usage:
            argv.append('-Rpass-analysis=kernel-resource-usage')
        argv += [f'{request.output}/{source}' for source in sources] + list(request.extra_sources)
        argv += [request.host, '-o', f'{request.output}/run']
        readable = [path for path in request.includes
                    if not (Path(path).is_relative_to(root) or Path(path).is_relative_to(out))]
        command, environment, container = compiler_command(request.compiler, argv, root, out, readable)
        steps.process('link', command, environment, container=container)
    executable = out / 'run'
    link = next((step for step in steps.records if step['step'] == 'link'), None)
    built = steps.failed is None and link is not None and executable.is_file()
    plan_step = next((step for step in steps.records if step['step'] == 'plan'), None)
    record = dict(schema_version=1, kind='model-build', provider=request.provider,
                  plan_identity=planned.identity if planned else None,
                  source_identity=artifact.identity if artifact else None, compiler=request.compiler,
                  steps=steps.records, argv=link['argv'] if link else None,
                  process=link['status'] if link else None, returncode=link['returncode'] if link else None,
                  log=link['log'] if link else None,
                  summary=Path(plan_step['log']).read_text().rstrip('\n') if plan_step else None,
                  executable=f'{request.output}/run' if built else None,
                  executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest() if built else None,
                  exit=0 if built else (steps.failed['exit'] if steps.failed else 1),
                  status='EXECUTABLE' if built else 'FAILED', device_execution=False)
    (steps.logs / 'model-build.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    return Configuration(record)


class _HostLink:
    """Ranks and one host translation unit, linked with the requested flags; no model-specific steps."""

    @staticmethod
    def refine(request):
        return request

    @staticmethod
    def prepare(request, steps):
        return request

    @staticmethod
    def check(request, steps):
        pass


def _provider(name):
    if name == 'host-link':
        return _HostLink
    if name == 'dsv2lite':
        from etx.models import dsv2lite
        return dsv2lite
    raise ValueError(f'unknown model build provider {name!r}')


def experiment_session_arms(provider, root):
    """Arms each arm-taking operation of `provider` accepts, as declared by repository `root`."""
    if provider != 'dsv2lite':
        raise ValueError(f'unknown experiment session provider {provider!r}')
    from etx.models.dsv2lite import session
    return session.allowed_arms(Path(root))


def run_experiment_session(request):
    from etx.api.sessions import ExperimentSessionRequest
    if not isinstance(request, ExperimentSessionRequest):
        raise TypeError('run_experiment_session requires ExperimentSessionRequest')
    allowed = experiment_session_arms(request.provider, request.root)
    if request.operation in allowed and request.arm not in allowed[request.operation]:
        raise ValueError(f'{request.operation} arm must be one of {allowed[request.operation]}')
    from etx.models.dsv2lite import session
    session.run(request.operation, request.arm, stamped=request.stamped,
                reference_dir=Path(request.reference_dir) if request.reference_dir else None,
                logs=request.logs, fleet=request.inputs['fleet'], data=request.inputs['data'], root=request.root)
    return 0


def prepare_model_assets(request):
    from types import SimpleNamespace
    from etx.api.assets import AssetPreparationRequest
    if not isinstance(request, AssetPreparationRequest):
        raise TypeError('prepare_model_assets requires AssetPreparationRequest')
    from etx.models.llm import prep
    prep.prepare(Path(request.root), SimpleNamespace(**request.arguments.to_dict()))
    return 0
