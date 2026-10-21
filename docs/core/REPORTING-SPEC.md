# Reporting specification

Scope: how results are reported. Work conventions that are not about reporting
(delegation, GPU teardown, do-only-useful-work) are out of scope. Rule IDs map
to the source requirements in `requirements-20261018.md`. Every rule is written
as a check an editor can run against a draft. "From:" is the source sentence,
verbatim.

## 1. Measurement frame

### F1 — Improvement is stated as before -> after
Check: for every claimed improvement the draft gives a pair "from X to Y" with
units (ms, tokens/s, %). A gain with no start value fails, unless B4 applies.
From: 具体我们提升了多少？从多少提升到多少？

### F2 — Setting is stated and references agreed context
Check: the draft names the setting — GPU, model, batch, context length, dtype,
decode-token count — and points to context everyone already agrees on (a named
public model, a named baseline such as vLLM). A number without its setting
fails.
From: 然后是在什么样的那个简单的 set 下，这个 set 可以引去一些就是大家共识的一些上下文。

### F3 — Two-week period is stated
Check: the draft states the period's start date and end date.
From: Two weeks at a time; state period start and period end.

### F4 — Every claim is measured or marked [U]
Check: each number is traceable to a measurement or code-verified result, or is
marked [U]. An estimate carries an estimate label and is never written as a bare
measurement.
From: Every claim must be measured or code-verified; unmeasured cells are [U];
no estimates presented as measurements.

## 2. What a result line must contain

### L1 — Every ratio names both sides
Check: for each ratio or percentage the draft names numerator and denominator
(e.g. "ETX tokens/s / vLLM tokens/s = 1.22", "4.087 ms -> 4.985 ms"). A bare
"1.22x" or "18% faster" with no sides fails.
From: 吞吐是谁的

### L2 — Fewest words, full information
Check: every word whose removal drops no fact required by any rule in this spec
is removed; every such fact is present.
From: 能用多少字就多少，但是信息量还是要全。

## 3. Section structure

### S1 — Grouped by direction, each with its concrete gain
Check: the report is grouped by direction; each direction states a concrete
gain as before -> after (per F1).
From: 按照方向、具体的收益，然后总结一下这两周的进展。

### S2 — Fixed skeleton
Check: sections appear in this order — ETX framework, Qwen, DeepSeek — and
inside each model two subsections: multi-batch baseline, then batch-1
performance.
From: Structure: ETX framework, Qwen, DeepSeek; inside each model split into
multi-batch baseline and batch-1 performance work.

### S3 — Only goal-bearing results
Check: every section and item maps to a stated goal; a result that does not
bear on a goal is absent.
From: Only results that bear on the goals.

### S4 — Key learnings are summarized
Check: the draft contains a short "key learnings" summary that states process
lessons and repeats no number already reported in the body.
From: 包括我们在过程中的一些关键 learning，其实也是很重要的，也可以去做一个摘要。

## 4. What is banned

### B1 — Internal codes
Check: no shorthand identifier (C1, C5, G, F0, arm name, session id) appears;
each experiment is named by its stage and its data change instead.
From: 永远不要用代号，用阶段的数据变化呈现。

### B2 — Correctness as a section
Check: no standalone "correctness" section; correctness appears only as a
one-line precondition on the numbers it certifies.
From: 正确性为什么单独说？

### B3 — Summary-of-summary
Check: no section restates numbers already reported in the body.
From: 净结果有意义吗？我们这个汇报本身不就是净结果

### B4 — Invented baseline
Check: a before -> after appears only when the "before" was measured; otherwise
the end state is stated alone.
From: 没有期初的就不用说

### B5 — Redacted material
Check: no redacted, secret, or token material appears in the report.
From: no [REDACTED] material in reports.

## 5. Delivery

### D1 — Chinese first, English separately
Check: the Chinese message is sent first; the English version is a separate
message.
From: Chinese message first, English message separately.

### D2 — Language split
Check: the reply is in Chinese; implementation and commits are in English.
From: Implementation and commits in English; replies to kailun in Chinese.

### D3 — Push to both remotes
Check: the commit is pushed to both remotes (origin and amd) using the HTTP
proxy.
From: Push to both remotes (origin, amd) with the HTTP proxy.

## 6. Conflict resolutions

- **L2 "fewest words" vs F1 "full information":** concision may delete only
  connective prose, never a fact another rule requires. A cut is valid iff no
  mandatory fact is lost.
- **B3 "no summary-of-summary" vs S4 "learnings summary":** the learnings
  summary reports lessons, not numbers, and must not repeat a before -> after
  already stated. A section that restates numbers is banned (B3); a section
  that states lessons is required (S4).
- **F1 "before -> after" vs B4 "no invented baseline":** F1 applies only when
  both endpoints were measured; if the start was never measured, report the end
  state alone (B4).
- **B2 "correctness not a section" vs F4 "every claim verified":** correctness
  is required (F4) but only as a one-line precondition on the numbers, never a
  section (B2).
