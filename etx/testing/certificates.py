"""Synthetic recorded sessions and import certificates (vLLM-kernel Qwen) for report/certificate tests; not evidence."""
import json


def _cert():
    from examples.vllm_llm import import_certificate
    return import_certificate


def write(d, name, body, gpu="0"):
    (d / f"{name}.log").write_text(f"# job {name} gpu={gpu} cwd=/x\n"
                                  "# echo 'RESULT matched=32 gen=32'; echo 'CHECK-DSL OK'\n" + body + "\n")


def etx(ms, arm="fused", matched=32, count=31):
    mode = "ETX megakernel (vLLM kernels imported)" if arm == "fused" else "unfused (same imported kernels, one launch per grid, HIP graph)"
    # A faster warm-up must not influence the final measured run.
    steps = "  step 0: 1.000 ms wall\n" + "".join(f"  step {t}: 1.000 ms wall\n" for t in range(1024, 1055))
    steps += "run 0: 32/32 generated tokens match HF greedy\n  step 0: 2.000 ms wall\n"
    steps += "".join(f"  step {t}: {ms:.3f} ms wall\n" for t in range(1024, 1024 + count))
    return ("ETX_ENV ETX_UNFUSED_SYNC= ETX_UNFUSED_NOGRAPH=\n" + steps
            + f"run 1: {matched}/32 generated tokens match HF greedy\nmode: {mode}\n"
            + f"RESULT model=/m requests=1 matched={matched} gen=32 median_ms={ms - .15:.3f} wall_ms={ms:.3f} matched_each={matched}")


def certificate_envelope(identity, body):
    tag = _cert().digest(identity)
    return f'BUILD-IDENTITY phase=before sha256={tag}\n{body}\nBUILD-IDENTITY phase=after sha256={tag}\n'


def certified_import_session(path, count=97):
    path.mkdir(exist_ok=True)
    plan = '# fixture\ntotal 100 parts 5 maxlen 1072 requests 1\n'
    plan += ''.join(f'{n} rms_norm_2d {n // 2} in_norm 1 1 1 0 1 0 0 0 {8 if n == 0 else 1} 112 -1 0 1\n' for n in range(count))
    (path / 'vl_plan.txt').write_text(plan)
    (path / 'imports.json').write_text(json.dumps({'rms_norm_2d': {'block': [256, 1, 1]}}))
    keys = ['executable', 'plan', 'imports', 'import_metadata', 'plan_header', 'plan_json',
            'weights_manifest', 'golden', 'compiler_output', 'device_source/megakernel_d0.hip',
            'device_source/dsl_tiles.hip', 'device_source/imp_tiles.hip']
    identity = dict(schema=1, check_enabled=True, compiler='AMD clang version fixture',
                    files={k: 'a' * 64 for k in keys})
    identity['files']['plan'] = _cert().sha(path / 'vl_plan.txt')
    identity['files']['import_metadata'] = _cert().sha(path / 'imports.json')
    (path / 'build_identity.json').write_text(json.dumps(identity))
    expected = _cert().expected_instances(plan, json.loads((path / 'imports.json').read_text())) if count else []
    for pos in (0, 1040):
        body = ''.join('CHECK-INSTANCE ' + json.dumps(dict(r, position=pos, bytes=128, result='identical', first_byte=-1)) + '\n' for r in expected)
        body += f'CHECK-IMPORTED checked={count} planned={count} replicas={sum(r["replicas"] for r in expected)} position={pos}\nCHECK OK: 0 launch(es) differ'
        (path / f'check_p{pos}.txt').write_text(certificate_envelope(identity, body))
    for name, arm in [('gate_chkx', 'fused'), ('gate_unf', 'graph')]:
        write(path, name, certificate_envelope(identity, etx(5, arm) + '\nCHECK-DSL OK: fixture\nGATE-OK'))
    for rnd in range(1, 6):
        for arm, mode, ms in [('etx', 'fused', 4.8), ('etxu', 'graph', 5.2)]:
            write(path, f'{arm}_r{rnd}', certificate_envelope(identity, etx(ms, mode)))
    return path
