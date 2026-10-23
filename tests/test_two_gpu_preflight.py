import json
from pathlib import Path

import pytest

from examples.vllm_llm import batch_correctness as bc
from etx.testing import qwen_staged_model as staged_model
from tools import two_gpu_preflight as pf

ROOT = Path(__file__).resolve().parent.parent








@pytest.fixture(scope='module')
def glm_fixture(tmp_path_factory):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from examples.glm53 import cpu_check
    out = tmp_path_factory.mktemp('glm-preflight') / 'stage'
    cpu_check.fixture(out)
    return out






def reference(tmp_path, staged, batch=2, **changes):
    from examples.glm53 import reference as reference_module
    from examples.glm53.prepare import metadata
    cfg, _ = metadata()
    record = dict(reference_module.identity(), backend='hf', batch=batch,
                  requests=[dict(prompt_ids=[1, 2, r], generated_ids=[4] * 32) for r in range(batch)],
                  provenance=dict(quantization='FP8 block dequantize -> BF16',
                                  index_sha256=staged['snapshot_inputs']['index_sha256'],
                                  source_shards=staged['snapshot_inputs']['shards']))
    record.update(changes)
    path = tmp_path / 'glm53-reference.json'
    path.write_text(json.dumps(record))
    with path.with_suffix('.logits').open('wb') as fh:
        fh.truncate(32 * batch * cfg['text_config']['vocab_size'] * 4)
    return path


def staged():
    return dict(snapshot_inputs=dict(index_sha256='a' * 64, shards={'model-00001.safetensors': 'b' * 64}))


def test_reference_layout_passes_and_carries_the_batch(tmp_path):
    from examples.glm53.prepare import metadata
    summary = pf.glm_reference(reference(tmp_path, staged()), staged(), 2)
    assert summary['batch'] == 2 and summary['steps'] == 32 and summary['prompt'] == 3
    vocab = metadata()[0]['text_config']['vocab_size']
    assert summary['logits_bytes'] == 32 * 2 * vocab * 4


@pytest.mark.parametrize('change, match', [
    (dict(backend='etx-pp2'), 'backend'),
    (dict(batch=3), 'batch must be'),
    (dict(provenance=dict(quantization='BF16')), 'FP8 numerical semantics'),
    (dict(revision='0' * 40), 'revision'),
])
def test_reference_identity_changes_fail_closed(tmp_path, change, match):
    path = reference(tmp_path, staged(), **change)
    with pytest.raises(pf.Fail, match=match):
        pf.glm_reference(path, staged(), None)


def test_reference_must_match_the_staged_shards_and_batch(tmp_path):
    path = reference(tmp_path, staged())
    with pytest.raises(pf.Fail, match='shard identity differs'):
        pf.glm_reference(path, dict(snapshot_inputs=dict(index_sha256='a' * 64, shards={'x': 'c' * 64})), None)
    with pytest.raises(pf.Fail, match='ETX_GLM_BATCH expects 1'):
        pf.glm_reference(path, staged(), 1)


def test_reference_requires_distinct_equal_length_prompts(tmp_path):
    from examples.glm53 import reference as reference_module
    from examples.glm53.prepare import metadata
    cfg, _ = metadata()
    staged_record = staged()
    base = dict(reference_module.identity(), backend='hf', batch=2,
                provenance=dict(quantization='FP8 block dequantize -> BF16',
                                index_sha256=staged_record['snapshot_inputs']['index_sha256'],
                                source_shards=staged_record['snapshot_inputs']['shards']))
    for requests, match in [
        ([dict(prompt_ids=[1, 2], generated_ids=[4] * 32)] * 2, 'distinct prompts'),
        ([dict(prompt_ids=[1, 2], generated_ids=[4] * 32),
          dict(prompt_ids=[1, 2, 3], generated_ids=[4] * 32)], 'unequal lengths'),
    ]:
        path = tmp_path / 'ref.json'
        path.write_text(json.dumps(dict(base, requests=requests)))
        with path.with_suffix('.logits').open('wb') as fh:
            fh.truncate(32 * 2 * cfg['text_config']['vocab_size'] * 4)
        with pytest.raises(pf.Fail, match=match):
            pf.glm_reference(path, staged_record, None)


def test_reference_logits_size_is_checked(tmp_path):
    path = reference(tmp_path, staged())
    path.with_suffix('.logits').write_bytes(b'x')
    with pytest.raises(pf.Fail, match='expected'):
        pf.glm_reference(path, staged(), None)
    path.with_suffix('.logits').unlink()
    with pytest.raises(pf.Fail, match='missing'):
        pf.glm_reference(path, staged(), None)
