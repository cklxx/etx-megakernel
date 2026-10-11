"""examples/vllm_llm/tile_ab.py: the tile-flag session's verdicts from synthetic job logs."""
from pathlib import Path

from examples.vllm_llm import tile_ab as ta


def _run(path: Path, ms: float, matched: int = 32) -> None:
    path.write_text(f"# job {path.stem} gpu=0 cwd=/x\n# grep 'RESULT .*matched=32 gen=32'\n"
                    f"run 0: {matched}/32 generated tokens match HF greedy\n"
                    f"RESULT model=/m requests=1 matched={matched} gen=32 median_ms={ms:.3f} wall_ms={ms + 0.2:.3f} "
                    f"tokens_per_s=200 matched_each={matched}\n")


def _session(d: Path, gains: dict, bits: dict, ctl_ident: bool = True, rounds: int = 3) -> None:
    for r in range(1, rounds + 1):
        for f, g in gains.items():
            _run(d / f"ab{r}_{f}_ctl.log", 4.830 + 0.005 * r)
            _run(d / f"ab{r}_{f}_x.log", 4.830 + 0.005 * r + g)
    lines = [f"BITCMP ctl2 {'identical' if ctl_ident else 'DIFFERS resid'}"]
    lines += [f"BITCMP {f} {'identical' if ok else 'DIFFERS logits'}" for f, ok in bits.items()]
    (d / "bitcmp.log").write_text("\n".join(lines) + "\n")


def test_verdicts(tmp_path, capsys):
    _session(tmp_path, {"all": -0.150, "wsum": -0.010, "scalar": +0.050, "gu1": -0.060, "rbatch": -0.080},
             {"all": True, "wsum": True, "scalar": True, "gu1": True, "rbatch": False})
    assert ta.main([str(tmp_path)]) == 0
    out = {l.split()[0]: l.split()[-1] for l in capsys.readouterr().out.splitlines() if l.split() and l.split()[0] in
           ("all", "wsum", "scalar", "gu1", "rbatch")}
    assert out == {"all": "KEEP", "wsum": "NEUTRAL", "scalar": "SLOWER", "gu1": "KEEP", "rbatch": "REJECT"}


def test_header_lines_are_not_results(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("# job x gpu=0\n# RESULT model=/m requests=1 matched=0 gen=32 median_ms=1.0 wall_ms=1.0\n")
    assert ta.result(p) is None


def test_incomplete_and_mixed(tmp_path):
    _session(tmp_path, {"gu1": -0.060}, {"gu1": True}, rounds=2)
    prs = ta.pairs(tmp_path, "gu1")
    assert len(prs) == 2 and ta.verdict(prs, True, True) == "INCOMPLETE"
    c, x = (4.8, 5.0, True), (4.72, 4.95, True)
    mixed = [(1, c, x), (2, c, x), (3, c, (4.81, 5.0, True))]   # mean -0.050 but one pair slower
    assert ta.verdict(mixed, True, True) == "MIXED"
    # a wrong token count rejects; a nondeterministic control leaves bit identity unproven
    assert ta.verdict([(1, c, (4.7, 4.9, False))] * 3, True, True) == "REJECT"
    assert ta.verdict([(1, c, x)] * 3, False, False) == "KEEP-UNVERIFIED"


def test_trace_bodies(tmp_path):
    p = tmp_path / "tr_ctl.log"
    p.write_text("  type                    tasks      wait      body first-ready last-ready  last-done\n"
                 "  qkv_0                     256     14.44      7.78        13.6        2.6       10.9\n"
                 "  qkv_1                     256      5.17     11.26       105.5        2.6       13.6\n"
                 "  topk_0_x                    8      9.29      7.75        62.4        0.5        8.2\n"
                 "  lm_head                   256      4.91    157.79      4538.5        2.6      160.3\n")
    b = ta.trace_bodies(p)
    assert abs(b["qkv"] - (7.78 + 11.26) / 2) < 1e-9 and b["lm_head"] == 157.79 and b["topk"] == 7.75


def test_trace_bodies_required_families(tmp_path):
    p = tmp_path / "tr_off.log"
    p.write_text("  type                    tasks      wait      body first-ready last-ready  last-done\n"
                 "  topk_0_x                     8      9.59      8.02        61.2        0.3        8.5\n"
                 "  attn_0                     20     24.22      5.59        28.8        0.7        6.2\n"
                 "  reduce_0                    4     12.97      5.96        35.9        0.4        6.3\n"
                 "  prep_0                      4     22.54      3.97        24.1        0.3        4.2\n"
                 "  moe_up_0                  192     13.44     17.12        68.7        1.0       18.5\n"
                 "  moe_down_0                256     13.62      9.77        87.3        1.9       11.7\n")
    assert ta.trace_bodies(p) == {"topk": 8.02, "attn": 5.59, "reduce": 5.96,
                                  "prep": 3.97, "moe_up": 17.12, "moe_down": 9.77}


def test_trace_bodies_stems_and_filter(tmp_path):
    p = tmp_path / "tr_ctl.log"
    # Numbered types are unrestricted; the bare-name whitelist stays unchanged.
    rows = [(f"{stem}{suffix}", body) for stem in ta.TYPES + ("topk", "in_norm", "custom_phase")
            for suffix, body in (("", 3.0), ("_0", 6.0), ("_1_x", 9.0))]
    rows += [("prologue", 100.0), ("argmax", 100.0), ("topk_2_y", 100.0), ("topk_2_x_more", 100.0)]
    p.write_text("".join(f"  {name:22}      8      1.00 {body:9.2f}         1.0        0.3        8.5\n"
                         for name, body in rows))
    expected = {stem: 6.0 for stem in ta.TYPES}
    expected.update(topk=7.5, in_norm=7.5, custom_phase=7.5)
    assert ta.trace_bodies(p) == expected
    assert ta.trace_bodies(tmp_path / "missing.log") == {}
