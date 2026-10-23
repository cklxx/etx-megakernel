"""GLM-5.3 CPU fixtures for E2E tests: a staged synthetic checkpoint and a reference record; not model evidence."""
import json
from pathlib import Path


def glm_cpu_stage(out):
    """Stage the CPU module-gate fixture checkpoint at `out` (needs torch and transformers)."""
    from examples.glm53 import cpu_check
    cpu_check.fixture(Path(out))
    return Path(out)


def glm_module_checkpoint(out):
    """Write the module-gate checkpoint (the release's dtype rules at module sizes) at `out` (needs torch)."""
    from examples.glm53 import module_gate
    module_gate.checkpoint(Path(out))
    return Path(out)


def glm_reference_record(tmp_path, staged, batch=2, **changes):
    """A synthetic HF reference record (+ zero logits file) matching the pinned release's identity."""
    from examples.glm53 import reference as reference_module
    from examples.glm53.prepare import metadata
    cfg, _ = metadata()
    record = dict(reference_module.identity(), backend='hf', batch=batch,
                  requests=[dict(prompt_ids=[1, 2, r], generated_ids=[4] * 32) for r in range(batch)],
                  provenance=dict(quantization='FP8 block dequantize -> BF16',
                                  index_sha256=staged['snapshot_inputs']['index_sha256'],
                                  source_shards=staged['snapshot_inputs']['shards']))
    record.update(changes)
    path = Path(tmp_path) / 'glm53-reference.json'
    path.write_text(json.dumps(record))
    with path.with_suffix('.logits').open('wb') as fh:
        fh.truncate(32 * batch * cfg['text_config']['vocab_size'] * 4)
    return path
