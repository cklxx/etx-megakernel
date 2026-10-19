"""tools/check_known_failures.py: new failures exit 1, known ones pass, stale entries are reported."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ckf", ROOT / "tools/check_known_failures.py")
ckf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ckf)

JUNIT = """<?xml version="1.0"?><testsuites><testsuite>
<testcase classname="tests.test_a" name="test_known[x]"><failure message="m"/></testcase>
<testcase classname="tests.test_a.TestC" name="test_err"><error message="m"/></testcase>
<testcase classname="tests.test_b" name="test_ok"/>
<testcase classname="tests.test_b" name="test_skip"><skipped/></testcase>
</testsuite></testsuites>"""


def _list(tmp_path, *ids):
    p = tmp_path / "known.txt"
    p.write_text("# local runs only\n" + "".join(f"{i} reason\n" for i in ids))
    return p


def test_junit_ids_and_outcomes(tmp_path):
    r = tmp_path / "r.xml"
    r.write_text(JUNIT)
    bad, ok = ckf.results(r)
    assert bad == {"tests/test_a.py::test_known[x]", "tests/test_a.py::TestC::test_err"}
    assert ok == {"tests/test_b.py::test_ok"}


def test_new_failure_exits_1_known_exits_0(tmp_path, capsys):
    r = tmp_path / "r.xml"
    r.write_text(JUNIT)
    assert ckf.main([str(r), "--list", str(_list(tmp_path, "tests/test_a.py::test_known[x]"))]) == 1
    assert "NEW    tests/test_a.py::TestC::test_err" in capsys.readouterr().out
    both = _list(tmp_path, "tests/test_a.py::test_known[x]", "tests/test_a.py::TestC::test_err")
    assert ckf.main([str(r), "--list", str(both)]) == 0


def test_stale_entries_reported_and_strict(tmp_path, capsys):
    r = tmp_path / "r.log"
    r.write_text("FAILED tests/test_a.py::test_x - boom\nPASSED tests/test_b.py::test_ok\n")
    lst = _list(tmp_path, "tests/test_a.py::test_x", "tests/test_b.py::test_ok")
    assert ckf.main([str(r), "--list", str(lst)]) == 0
    assert "STALE  tests/test_b.py::test_ok" in capsys.readouterr().out
    assert ckf.main([str(r), "--list", str(lst), "--strict"]) == 1


def test_committed_list_says_local_only():
    head = (ROOT / "tools/known_env_failures.txt").read_text().splitlines()[:6]
    assert any("local runs only" in l.lower() for l in head) and any("not" in l.lower() and "CI" in l for l in head)
