"""Same-machine interleaved A/B measurement with an A/A control (docs/EXPERIMENTS.md, measurement rules).

The script runs three arms on one machine: A (the baseline), B (the candidate) and C (a second copy of A, the
control). First, every arm runs once as a warm-up. The warm-up values do not count. Then every round runs each arm
once. The rounds cycle through all 6 orders of A, B and C (ORDERS), so each arm is in each position equally often and,
over the whole run sequence, each arm runs directly after each other arm equally often. --rounds must be a multiple
of 6.
Each run prints one timing value. A regular expression with exactly one group reads the value from the run output.

For each round the script computes two paired gains (lower is better):
  gain    = A / B - 1     the gain of the candidate
  control = A / C - 1     the gain of A over itself; this is the noise of the machine and of the order
p is the one-sided exact sign-test p-value of the gains in the observed direction (the sign of the median gain) (6 positive gains out
of 6 give p = 0.5^6 = 0.0156). The verdict uses only these paired gains, never a number from another session or
another machine:
  WIN      every gain > 0, median gain > max(--min-gain, max |control|) and p <= 0.05 (--min-gain default 1 %)
  LOSS     every gain < 0, -median gain > max(--min-gain, max |control|) and p <= 0.05
  NO-DIFF  otherwise
  INCOMPLETE  a run fails or times out, a run prints no value, a value is not a positive number, the identity
              command fails, or the machine identity changes. The script stops at the first such problem and keeps
              the completed runs.

The machine identity is read at the start, before every round and at the end. By default it is the local host
name and the GPU serial numbers (rocm-smi or nvidia-smi). When the runs go over ssh to a GPU host, give
--identity-cmd, for example: --identity-cmd "ssh gpu1 'hostname; rocm-smi --showserial --json'".

The commands run with a shell (shell=True). They are free text from the operator; do not pass untrusted input.
The exit code is 0 for WIN, LOSS and NO-DIFF, 1 for INCOMPLETE and 2 for bad arguments.

Output, in --out:
  runs/warm_<arm>.log, runs/r<round>_<arm>.log   the full output of each run
  ab.log    ABMACHINE/ABWARN lines, one ABWARM line per warm-up run, one ABRUN line per run, then one ABREPORT
            line; tools/extract_key.py reads it. The file is rewritten after every run.
  ab.json   the same numbers
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
from pathlib import Path

ARMS = ("A", "B", "C")
# The 6 orders of 3 arms. Each arm is in each position twice. Inside the rounds each ordered pair of different
# arms is adjacent twice. The last arm of each order is the first arm of the next order (also from BCA back to
# ABC), so the round boundaries add only A->A, B->B and C->C. Over the whole run sequence the 6 cross-arm pairs
# are therefore equally frequent.
ORDERS = (("A", "B", "C"), ("C", "A", "B"), ("B", "A", "C"), ("C", "B", "A"), ("A", "C", "B"), ("B", "C", "A"))
KILL_WAIT = 10.0  # seconds to wait for the output pipe after the process group is killed
ALPHA = 0.05


class IdentityError(RuntimeError):
    pass


def local_identity() -> str:
    """Host name and GPU serial numbers of this machine. A failed GPU probe gives no field, not an error."""
    parts = [socket.gethostname()]
    for cmd in (["rocm-smi", "--showserial", "--json"], ["nvidia-smi", "--query-gpu=serial,uuid", "--format=csv,noheader"]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and r.stdout.strip():
            parts.append(" ".join(r.stdout.split()))
    return " | ".join(parts)


machine_identity = local_identity  # the old name


def command_identity(cmd: str, timeout: float = 120):
    """Return an identity function that runs cmd and uses its stdout. A failure raises IdentityError."""
    def identity() -> str:
        rc, text = run_command(cmd, timeout, merge=False)
        if rc != 0:
            raise IdentityError(f"--identity-cmd rc={rc}")
        value = " ".join(text.split())
        if not value:
            raise IdentityError("--identity-cmd printed nothing")
        return value
    return identity


def kill_group(p: subprocess.Popen) -> None:
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_command(cmd: str, timeout: float, merge: bool = True, kill_wait: float = KILL_WAIT):
    """Run a shell command in its own process group. Return (rc, output); rc is "timeout" on a timeout.

    After every run, also after rc=0, the whole process group gets SIGKILL, so a background child cannot hold the
    GPU during the next arm. On a timeout the group is killed first; then the script waits at most kill_wait
    seconds for the output pipe. A child outside the group (setsid) can keep the pipe open; then the pipe is closed
    and the rest of its output is lost. Over ssh a local kill stops only the local ssh client: see
    docs/EXPERIMENTS.md.
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merge else subprocess.PIPE,
                         text=True, errors="replace", start_new_session=True)
    try:
        try:
            out, _ = p.communicate(timeout=timeout)
            return p.returncode, out or ""
        except subprocess.TimeoutExpired:
            kill_group(p)
            note = f"\n[ab_same_box] timeout after {timeout} s; process group killed\n"
            try:
                out, _ = p.communicate(timeout=kill_wait)
            except subprocess.TimeoutExpired:
                out = ""
                note += f"[ab_same_box] the output pipe stayed open {kill_wait} s after the kill; output lost\n"
            return "timeout", (out or "") + note
    finally:
        kill_group(p)
        for pipe in (p.stdout, p.stderr):
            if pipe is not None:
                pipe.close()
        try:
            p.wait(timeout=kill_wait)
        except subprocess.TimeoutExpired:
            p.kill()


def order(round_index: int) -> tuple[str, ...]:
    """The arm order of a round: the 6 orders in ORDERS, cycled."""
    return ORDERS[round_index % len(ORDERS)]


def check_metric(metric: str) -> re.Pattern:
    pattern = re.compile(metric)
    if pattern.groups != 1:
        raise ValueError(f"--metric must have exactly 1 group, it has {pattern.groups}: {metric!r}")
    return pattern


def parse_value(pattern: re.Pattern, text: str):
    """Return (value, problem). The last match counts. The value must be a finite number above 0."""
    found = pattern.findall(text)
    if not found:
        return None, f"printed no value for {pattern.pattern!r}"
    raw = found[-1]
    try:
        value = float(raw)
    except ValueError:
        return None, f"printed a value that is not a number: {raw!r}"
    if not math.isfinite(value) or value <= 0:
        return None, f"printed a value that is not a positive finite number: {raw!r}"
    return value, None


def spread(values: list[float]) -> dict:
    return dict(n=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values),
                spread=max(values) - min(values))


def sign_test_p(gains: list[float], positive: bool) -> float:
    """One-sided exact sign test: P(at least k of n nonzero gains have this sign | no effect). Zeros are dropped."""
    nonzero = [g for g in gains if g != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = sum(1 for g in nonzero if (g > 0) == positive)
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def verdict(gains: list[float], controls: list[float], min_gain: float = 0.0) -> tuple[str, str, float]:
    """Return (verdict, rule, p). min_gain is a fraction (0.01 = 1 %)."""
    noise = max(abs(c) for c in controls)
    med = statistics.median(gains)
    bar = max(min_gain, noise)
    p = sign_test_p(gains, positive=med >= 0)
    tail = f"max |control| {noise:.6%}, min gain {min_gain:.6%}, sign test p={p:.6g}"
    if all(g > 0 for g in gains) and med > bar and p <= ALPHA:
        return "WIN", f"every gain > 0, median {med:+.6%} > {bar:.6%}, p <= {ALPHA}; {tail}", p
    if all(g < 0 for g in gains) and -med > bar and p <= ALPHA:
        return "LOSS", f"every gain < 0, |median| {-med:.6%} > {bar:.6%}, p <= {ALPHA}; {tail}", p
    return "NO-DIFF", f"median {med:+.6%}; {tail}", p


def check_args(args: argparse.Namespace) -> None:
    """Raise ValueError with a clear message when an argument is not valid. main() maps it to exit code 2."""
    for name in ("a", "b", "out"):
        if not isinstance(getattr(args, name, None), str) or not getattr(args, name).strip():
            raise ValueError(f"--{name} must be a non-empty string")
    rounds = getattr(args, "rounds", None)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 6 or rounds % 6:
        raise ValueError(f"--rounds must be a multiple of 6 (6, 12, ...), not {rounds!r}: "
                         "the rounds cycle through all 6 arm orders")
    timeout = getattr(args, "timeout", None)
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"--timeout must be a finite number of seconds above 0, not {timeout!r}")
    min_gain = getattr(args, "min_gain", 1.0)
    if not isinstance(min_gain, (int, float)) or not math.isfinite(min_gain) or min_gain < 0:
        raise ValueError(f"--min-gain must be a finite percent of 0 or more, not {min_gain!r}")
    label = getattr(args, "label", None)
    if not isinstance(label, str) or not label or re.search(r"\s", label):
        raise ValueError(f"--label must be non-empty and contain no white space, not {label!r}")
    try:
        check_metric(args.metric)
    except re.error as exc:
        raise ValueError(f"--metric is not a valid regex: {exc}") from None


def run(args: argparse.Namespace, identity=None) -> dict:
    check_args(args)
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    pattern = check_metric(args.metric)
    min_gain = getattr(args, "min_gain", 1.0) / 100
    identity_cmd = getattr(args, "identity_cmd", None)
    if identity is None:
        identity = command_identity(identity_cmd) if identity_cmd else local_identity
    cmds = {"A": args.a, "B": args.b, "C": args.control or args.a}
    lines: list[str] = []
    values = {arm: [] for arm in ARMS}
    warm: dict = {}
    problems: list[str] = []
    result = dict(label=args.label, rounds=args.rounds, commands=cmds, metric=args.metric, min_gain_pct=min_gain * 100,
                  identity_cmd=identity_cmd, values=values, warmup=warm, problems=problems)

    def save():
        (out / "ab.log").write_text("\n".join(lines) + "\n")
        (out / "ab.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    def check_identity(when: str) -> bool:
        try:
            now = identity()
        except IdentityError as exc:
            problems.append(f"{when}: {exc}")
            lines.append(f"ABMACHINE {when} ERROR {exc}")
            return False
        if "machine" not in result:
            result["machine"] = now
            lines.append(f"ABMACHINE start {now}")
            if not identity_cmd and " | " not in now:
                lines.append("ABWARN no GPU serial was read; the identity is the host name only")
            return True
        if now != result["machine"]:
            problems.append(f"the machine identity changed ({when})")
            lines.append(f"ABMACHINE {when} {now}")
            return False
        return True

    def one(tag: str, name: str, arm: str):
        log = out / "runs" / f"{name}.log"
        rc, text = run_command(cmds[arm], args.timeout)
        log.write_text(f"$ {cmds[arm]}\n{text}")
        value, issue = parse_value(pattern, text)
        lines.append(f"{tag} arm={arm} rc={rc} value={value} src=runs/{log.name}")
        where = tag.replace("ABRUN round=", "round ") if tag != "ABWARM" else "warm-up"
        if rc != 0:
            problems.append(f"{where} arm {arm} rc={rc}")
        elif issue:
            problems.append(f"{where} arm {arm} {issue}")
        save()
        return value

    finished = False
    try:
        if check_identity("start"):
            for arm in ARMS:
                warm[arm] = one("ABWARM", f"warm_{arm}", arm)
                if problems:
                    break
        for r in range(args.rounds):
            if problems or not check_identity(f"before round {r}"):
                break
            for arm in order(r):
                values[arm].append(one(f"ABRUN round={r}", f"r{r}_{arm}", arm))
                if problems:
                    break
        if not problems:
            check_identity("end")
        finished = True
    finally:
        if not finished:
            problems.append(f"the session stopped early: {sys.exc_info()[0].__name__}")
        if problems:
            result.update(verdict="INCOMPLETE", rule="; ".join(problems))
            lines.append(f"ABREPORT label={args.label} verdict=INCOMPLETE reason={result['rule']}")
        else:
            gains = [a / b - 1 for a, b in zip(values["A"], values["B"])]
            controls = [a / c - 1 for a, c in zip(values["A"], values["C"])]
            v, rule, p = verdict(gains, controls, min_gain)
            g, c = spread(gains), spread(controls)
            noise = max(abs(x) for x in controls)
            result.update(gains=gains, controls=controls, gain=g, control=c, max_control=noise, p=p, verdict=v,
                          rule=rule, medians={arm: statistics.median(values[arm]) for arm in ARMS})
            lines.append(f"ABREPORT label={args.label} B vs A: gain median {g['median'] * 100:+.6f}% spread "
                         f"{g['spread'] * 100:.6f}% control median {c['median'] * 100:+.6f}% spread "
                         f"{c['spread'] * 100:.6f}% max|control| {noise * 100:.6f}% rounds={args.rounds} "
                         f"p={p:.6g} min_gain={min_gain * 100:g}% verdict={v}")
        save()
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, help="baseline command (shell)")
    ap.add_argument("--b", required=True, help="candidate command (shell)")
    ap.add_argument("--control", help="control command; default: the baseline command again (A/A)")
    ap.add_argument("--metric", required=True, help="regex with exactly one group: the timing value, lower is better")
    ap.add_argument("--rounds", type=int, default=6, help="a multiple of 6 (default 6)")
    ap.add_argument("--min-gain", type=float, default=1.0,
                    help="smallest median gain in percent for WIN or LOSS (default 1)")
    ap.add_argument("--identity-cmd", help="shell command whose stdout is the machine identity (for ssh runs)")
    ap.add_argument("--timeout", type=float, default=1800, help="seconds per run; a timeout makes INCOMPLETE")
    ap.add_argument("--label", default="ab", help="a name with no white space")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        check_args(args)
    except ValueError as exc:
        ap.error(str(exc))  # exit code 2
    r = run(args)
    print((Path(args.out) / "ab.log").read_text(), end="")
    return 0 if r["verdict"] != "INCOMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
