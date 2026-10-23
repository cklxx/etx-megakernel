"""etx command line.

  python -m etx compile examples/moe_layer.py --arch gfx942 --out build/moe --sim
  python -m etx explain examples/moe_layer.py --arch gfx942
  python -m etx archs
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from etx.external import (CompilerRequest, compiler_command, PlanRequest, WorkloadSpec,
                          plan, emit, save_sources, load_sources, build, load_build,
                          ArtifactRunRequest, run_artifact, read_artifact_run, environment_options)


def _request(args):
    return CompilerRequest(args.cmd, graph=getattr(args, 'graph', None),
                           target=getattr(args, 'arch', 'gfx942'),
                           output=getattr(args, 'out', 'build/out'),
                           bindings={key: int(value) for key, value in
                                     (item.split('=') for item in getattr(args, 'bind', []))},
                           options={} if args.cmd == 'archs' else dict(
                               event_elimination=not args.no_event_elim,
                               prefetch=not args.no_prefetch,
                               split_resource_classes=args.split_classes,
                               force_mode=args.force_mode,
                               inline_tiles=args.inline_tiles, relay=args.relay,
                               # The command line is where the build environment's switches enter.
                               **environment_options(os.environ).to_dict()),
                           simulate=getattr(args, 'sim', False), seed=getattr(args, 'seed', 0))


def cmd_compile(args):
    return compiler_command(_request(args))


def cmd_explain(args):
    return compiler_command(_request(args))


def cmd_archs(args):
    return compiler_command(_request(args))


def cmd_sources(args):
    fields = json.loads(Path(args.request).read_text())
    fields['workload'] = WorkloadSpec.from_dict(fields['workload'])
    artifact = emit(plan(PlanRequest(**fields)))
    save_sources(artifact, args.out)
    print(json.dumps(dict(identity=artifact.identity, status=artifact.status), sort_keys=True))
    return 0


def cmd_build(args):
    result = build(load_sources(args.sources), output=args.out, docker=args.docker,
                   image=args.image, deadline_s=args.deadline)
    print(json.dumps(dict(identity=result.identity, status=result.status), sort_keys=True))
    return 0 if result.status == 'EXECUTABLE' else 1


def cmd_inspect_build(args):
    result = load_build(args.directory)
    print(json.dumps(dict(identity=result.identity, **result.to_dict()), sort_keys=True))
    return 0 if result.status == 'EXECUTABLE' else 1


def cmd_run(args):
    request = ArtifactRunRequest(**json.loads(Path(args.request).read_text()))
    result = run_artifact(request, output=args.out, execute_native=args.execute)
    print(json.dumps(dict(identity=result.identity, **result.to_dict()), sort_keys=True))
    return result.process.returncode if result.process.returncode is not None else 1


def cmd_inspect_run(args):
    request = ArtifactRunRequest(**json.loads(Path(args.request).read_text()))
    result = read_artifact_run(args.directory, request=request)
    print(json.dumps(dict(identity=result.identity, **result.to_dict()), sort_keys=True))
    return 0 if result.process.status == 'complete' else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="etx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("compile", cmd_compile), ("explain", cmd_explain)):
        s = sub.add_parser(name)
        s.add_argument("graph")
        s.add_argument("--arch", default="gfx942")
        s.add_argument("--out", default="build/out")
        s.add_argument("--bind", action="append", default=[], help="SYM=value overrides")
        s.add_argument("--sim", action="store_true")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--no-event-elim", action="store_true")
        s.add_argument("--no-prefetch", action="store_true")
        s.add_argument("--split-classes", action="store_true")
        s.add_argument("--force-mode", choices=["static", "dynamic", "hybrid"], default=None)
        s.add_argument("--inline-tiles", action="store_true", help="#include the tile sources into the kernel TU so bodies can inline")
        s.add_argument("--relay", choices=["auto", "on", "off"], default="off", help="per-domain relay of DEVICE-scope counters (P5); off by default")
        s.set_defaults(fn=fn)
    s = sub.add_parser('sources', help='emit a relocatable source artifact from a declarative PlanRequest JSON')
    s.add_argument('request')
    s.add_argument('--out', required=True)
    s.set_defaults(fn=cmd_sources)
    s = sub.add_parser('build', help='compile and link captured sources offline, without execution')
    s.add_argument('sources')
    s.add_argument('--out', required=True)
    s.add_argument('--docker', required=True)
    s.add_argument('--image', required=True)
    s.add_argument('--deadline', type=float, default=90)
    s.set_defaults(fn=cmd_build)
    s = sub.add_parser('inspect-build', help='validate a retained build and its compiler evidence')
    s.add_argument('directory')
    s.set_defaults(fn=cmd_inspect_build)
    s = sub.add_parser('run', help='run a validated artifact with explicit native execution opt-in')
    s.add_argument('request')
    s.add_argument('--out', required=True)
    s.add_argument('--execute', action='store_true')
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser('inspect-run', help='validate artifact-bound process evidence, not device qualification')
    s.add_argument('directory')
    s.add_argument('--request', required=True)
    s.set_defaults(fn=cmd_inspect_run)
    s = sub.add_parser("archs")
    s.set_defaults(fn=cmd_archs)
    a = ap.parse_args(argv)
    return a.fn(a)
