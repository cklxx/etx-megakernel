"""Translator for model build shells: `python -m etx.tools.model_build <recipe.json> <argument> [compiler flags]`.

The recipe (examples/<model>/build.json) declares the model file, its configuration, planner options,
build provider and variant knobs, host, include directories, compiler flags and the historical
console contract; this command maps the shell's arguments and environment onto one
etx.external.build_model request and prints what the shell printed.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from etx.external import BuildRefused, ModelBuildRequest, PlanRequest, build_model, environment_options, model


# Variables this translator consumes itself; they never reach the declaration.
CONTROL = ('ETX_OUT', 'ETX_PYTHON', 'ETX_MODEL_COMPILER')


def request(recipe, argument, extra, environ, root):
    if recipe.get('schema_version') != 1:
        raise SystemExit('unsupported build recipe')
    kind = recipe['argument']
    # The shell resolved a packed directory (`cd && pwd`) but passed a fleet path through verbatim.
    value = os.path.abspath(argument) if kind == 'packed' else argument
    if kind == 'packed' and not os.path.isdir(value):
        raise SystemExit(f'packed model dir not found: {value}')
    fill = lambda text, **more: text.format(**{kind: value}, name=os.path.basename(value), **more)
    out = environ.get(recipe['output_override']) or str(root / fill(recipe['output']))
    # An absolute ETX_OUT is used verbatim, as the legacy shells did; relative ones resolve against the caller.
    out = out if os.path.isabs(out) else os.path.abspath(out)
    declaration = root / environ.get(recipe.get('declaration_override', ''), recipe['declaration'])
    environment = {key: fill(text) for key, text in recipe.get('environment', {}).items()}
    environment.update({key: environ[key] for key in recipe.get('forward_environment', ()) if key in environ})
    prefixes = tuple(recipe.get('forward_prefixes', ()))
    environment.update({key: text for key, text in environ.items()
                        if prefixes and key.startswith(prefixes) and key not in CONTROL})
    options = dict(recipe['options'])
    # The build environment's planner/emission switches, as the legacy shells' compiler inherited them.
    switches = environment_options(environ)
    for key, variable in (('coherence', 'ETX_COHERENCE'), ('prefetch_isolate_poller', 'ETX_PREFETCH_ISOLATE_POLLER'),
                          ('inbody_emit', 'ETX_INBODY_EMIT')):
        if variable in environ:
            options[key] = switches[key]
    workload = model(Path(recipe['declaration']).parent.name, config=dict(
        declaration=str(declaration), declaration_sha256=hashlib.sha256(declaration.read_bytes()).hexdigest(),
        environment=environment))
    includes = tuple(fill(path, out=out) if '{out}' in path else str(root) if path == '.' else str(root / path)
                     for path in recipe.get('includes', ()))
    variant = {name: environ.get(variable, '') for name, variable in recipe.get('variant_environment', {}).items()}
    if recipe.get('provider', 'host-link') != 'host-link':
        variant[kind] = value
    return out, value, ModelBuildRequest(
        PlanRequest(workload, options=options), str(root), out, str(root / recipe['host']),
        (*recipe.get('flags', ()), *extra), includes, recipe.get('resource_usage', False),
        environ.get('ETX_MODEL_COMPILER', 'hipcc'), provider=recipe.get('provider', 'host-link'), variant=variant)


def resource_lines(log):
    # Historical filter: grep -E "Function Name|VGPRs:|ScratchSize|VGPRs Spill|error" | sed | paste - - - -
    keep = [re.sub(r' \[-Rpass.*', '', re.sub(r'.*remark: +', '', line))
            for line in log.splitlines() if re.search(r'Function Name|VGPRs:|ScratchSize|VGPRs Spill|error', line)]
    return ['\t'.join((keep[i:i + 4] + [''] * 4)[:4]) for i in range(0, len(keep), 4)]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        raise SystemExit('usage: python -m etx.tools.model_build <recipe.json> <argument> [compiler flags]')
    root = Path(__file__).resolve().parents[2]
    recipe = json.loads(Path(argv[0]).read_text())
    out, value, build = request(recipe, argv[1], argv[2:], os.environ, root)
    try:
        result = build_model(build)
    except BuildRefused as refusal:
        print(refusal, file=sys.stderr)
        return 2
    verbose = os.environ.get(recipe.get('summary_verbose', ''), '0') == '1'
    for step in result['steps']:
        text = Path(step['log']).read_text(errors='replace')
        if step['step'] == 'plan':
            for line in text.splitlines():
                if verbose or re.search(recipe['summary'], line):
                    print(line)
        elif step['step'] == 'link' and recipe.get('link_output') == 'resource-usage':
            for line in resource_lines(text):
                print(line)
        else:
            sys.stdout.write(text)
        sys.stdout.flush()
    if result['status'] != 'EXECUTABLE':
        message = recipe.get('failure_message')
        if message:
            print(message.format(out=out))
        return recipe.get('failure_exit') or result['exit'] or 1
    print(recipe['built_message'].format(out=out, **{recipe['argument']: value}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
