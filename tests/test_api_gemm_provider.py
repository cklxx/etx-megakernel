"""Provider parity against independently captured pre-extraction bytes."""
import pytest

from etx.testing import gemm_provider_conformance as provider_conformance


# Original examples/gemm_rs.py SHA256:
# 5163d915bba03b879630e1e2b05828dd1e44343abc324b994fde360c18351551
# Local and desktop pre-extraction captures agree on all graph/full-plan/source
# bytes: gfx90a/gfx942, default/small bindings, auto/static/dynamic/hybrid modes.
BASELINE = '9f45e7b5911bcb6b398c393529ce0a82f3cd09209d9f260a2e1b057296556add'


@pytest.mark.parametrize('kind', ['library', 'legacy', 'isolated'])
def test_gemm_provider_preserves_pre_extraction_bytes(kind, monkeypatch):
    monkeypatch.setenv('ETX_COHERENCE', 'declared')
    result = provider_conformance(kind=kind)
    assert result['signature'] == BASELINE
    assert result['plans'] == 16
    assert result['backend'] == 'cpu'
    assert result['device_qualification'] is False


def test_gemm_provider_probe_rejects_unknown_kind():
    with pytest.raises(ValueError, match='unknown GEMM provider probe'):
        provider_conformance(kind='unknown')
