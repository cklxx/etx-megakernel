"""CPU acceptance policy conformance through supported testing probes."""
from etx.testing import acceptance_results, verify_acceptance_archive


def test_external_and_unexpected_skips_are_distinguished(tmp_path):
    xml = tmp_path / 'results.xml'
    xml.write_text('''<testsuites><testsuite>
      <testcase classname="tests.test_ref_numpy" name="test_ref_numpy_matches_hf"><skipped message="checkpoint"/></testcase>
      <testcase classname="tests.test_cpu_numerical" name="test_complete_acceptance_suite"><skipped message="unexpected"/></testcase>
      <testcase classname="tests.test_bench_sim" name="ok"/>
      <testcase classname="tests.test_bench_sim" name="bad"><failure/></testcase>
    </testsuite></testsuites>''')
    result = acceptance_results(xml).to_dict()
    assert result['passed'] == 1 and len(result['skipped']) == 2
    assert len(result['unexpected_skips']) == 1 and len(result['failed']) == 1


def test_vllm_source_skip_is_external_but_other_skips_and_failures_are_not(tmp_path):
    xml = tmp_path / 'results.xml'
    xml.write_text('''<testsuites><testsuite>
      <testcase classname="tests.test_vlbest_session" name="test_g0_local_source_baseline"><skipped message="set ETX_VLLM_SOURCE_ROOT to inject a vLLM source checkout"/></testcase>
      <testcase classname="tests.test_vlbest_session" name="test_g0_other"><skipped message="unexpected"/></testcase>
    </testsuite></testsuites>''')
    result = acceptance_results(xml).to_dict()
    assert len(result['skipped']) == 2  # External checks remain pending, never counted as passes.
    assert result['passed'] == 0 and result['failed'] == []
    assert result['unexpected_skips'] == [
        {'test': 'tests.test_vlbest_session.test_g0_other', 'reason': 'unexpected'}]
    xml.write_text('''<testsuites><testsuite>
      <testcase classname="tests.test_vlbest_session" name="test_g0_local_source_baseline"><failure message="invalid source baseline"/></testcase>
    </testsuite></testsuites>''')
    result = acceptance_results(xml).to_dict()
    assert result['failed'] == ['tests.test_vlbest_session.test_g0_local_source_baseline']
    assert result['skipped'] == [] and result['passed'] == 0


def test_interrupted_or_truncated_junit_cannot_be_success(tmp_path):
    xml = tmp_path / 'results.xml'
    xml.write_text('<testsuites><testsuite><testcase><error message="interrupted"/></testcase>'
                   '<testcase name="completed"/></testsuite></testsuites>')
    result = acceptance_results(xml)
    assert result['passed'] == 1
    assert result['failed'] == ['malformed testcase 0: missing name']
    xml.write_text('<testsuites><testsuite>')
    result = acceptance_results(xml)
    assert result['passed'] == 0 and 'unreadable pytest results' in result['failed'][0]


def test_archive_members_must_match_pinned_inventory(tmp_path):
    import hashlib
    import pytest
    member = tmp_path / 'plan.json'
    member.write_text('{"tasks": 1}\n')
    index = tmp_path / 'SHA256SUMS'
    index.write_text(hashlib.sha256(member.read_bytes()).hexdigest() + '  ./plan.json\n')
    pin = hashlib.sha256(index.read_bytes()).hexdigest()
    assert str(member) in verify_acceptance_archive(tmp_path, pin).to_dict()
    member.write_text('{"tasks": 999}\n')
    with pytest.raises(ValueError, match='member checksum'):
        verify_acceptance_archive(tmp_path, pin).to_dict()
    member.unlink()
    with pytest.raises(ValueError, match='missing file'):
        verify_acceptance_archive(tmp_path, pin).to_dict()
    index.write_text('rewritten inventory\n')
    with pytest.raises(ValueError, match='inventory checksum'):
        verify_acceptance_archive(tmp_path, pin).to_dict()


def test_archive_rejects_symlink_alias(tmp_path):
    import hashlib
    import pytest
    member = tmp_path / 'plan.json'
    member.write_text('{"tasks": 1}\n')
    index = tmp_path / 'SHA256SUMS'
    index.write_text(hashlib.sha256(member.read_bytes()).hexdigest() + '  ./plan.json\n')
    pin = hashlib.sha256(index.read_bytes()).hexdigest()
    assert str(member) in verify_acceptance_archive(tmp_path, pin).to_dict()
    alias = tmp_path / 'alias.json'
    alias.symlink_to(member)    # resolves onto a listed member, so a resolved-only inventory misses it
    with pytest.raises(ValueError, match='symlink'):
        verify_acceptance_archive(tmp_path, pin).to_dict()
    alias.unlink()
    (tmp_path / 'outside').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match='symlink'):
        verify_acceptance_archive(tmp_path, pin).to_dict()
