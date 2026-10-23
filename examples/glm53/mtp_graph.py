"""Offline prerequisites for the GLM MTP draft graph (not yet executable).

The pinned module8 checkpoint contains only main-model layers. Do not use the
verify graph or ordinary decode as a surrogate MTP oracle: neither consumes the
MTP embedding/hidden-state fusion weights. This preflight deliberately refuses
qualification until an independently loadable HF head and staged MTP weights
exist. It performs no downloads, checkpoint writes, or GPU operations.
"""
from __future__ import annotations

import json

from examples.glm53 import module_gate


class MTPUnavailable(RuntimeError):
    """The local gate cannot support an honest MTP correctness comparison."""


def prerequisites():
    """Return actionable, independently checked blockers for the pinned gate.

    Use the generator's tensor inventory without materializing its matrices.
    Check HF's actual configuration API rather than inferring support from the
    release's num_nextn_predict_layers metadata.
    """
    from transformers import AutoConfig

    cfg = module_gate.module_config('module8')
    text = cfg['text_config']
    depth = text['num_hidden_layers']
    names = {name for i in range(depth)
             for name, _, _, _ in module_gate.native_tensors(text, i)}
    names.update(name for name, _ in module_gate.outer_tensors(text))
    prefix = f'model.language_model.layers.{depth}.'
    blockers = []
    if not any(name.startswith(prefix) for name in names):
        blockers.append(
            f'module8 generator has no MTP layer {depth} tensors; '
            'fusion norms/projection and the draft decoder weights are absent')
    config = AutoConfig.for_model(**cfg)
    try:
        config.get_mtp_config()
    except (ValueError, AttributeError) as exc:
        blockers.append(f'HF {type(config.get_text_config()).__name__}: {exc}')
    return dict(gate='module8', main_layers=depth,
                declared_mtp_layers=text.get('num_nextn_predict_layers'),
                status='BLOCKED' if blockers else 'PREREQUISITES-ONLY',
                blockers=blockers, qualified=False,
                required_comparison=dict(positions=32, k=5, reference='HF MTP head'))


def require_prerequisites():
    """Refuse to draft from the main model when MTP inputs are unavailable."""
    report = prerequisites()
    if report['blockers']:
        raise MTPUnavailable('; '.join(report['blockers']))
    return report


def main():
    report = prerequisites()
    print(json.dumps(report, indent=2))
    return 2 if report['blockers'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
