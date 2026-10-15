# Required bounded review closure

Original two diagnostic implementation: a0c3e724dbd1a342d96812d232f04ced902d6496; evidence-only pin 1a23336e1bdf1c78c63ec35f3dab136f88acc2b0. Required final report remains /Users/ckl/.hermes/cache/scratch/cc-gpt-deepseek-final-review.md. The earlier external driver reached its 30-turn cap without producing that report; no verdict is inferred.

The full suite exposed one real shared-supervisor failure. Current fcb5f649463573c9dc728f8d4de46d5bb5a27c7a changes only etx/session/proc.py and tests/test_session_proc.py. Review the bounded diff `git diff 1a23336..fcb5f6 -- etx/session/proc.py tests/test_session_proc.py` in /Users/ckl/code/.wt/etx-deepseek-two-gaps. No DeepSeek implementation/build source changed.

Condition: a macOS job exits after starting `sleep 30 &`; without /proc, sweep sent SIGKILL and returned immediately, while the child could still be observed alive. New sweep waits until the group no longer exists, under the existing 30-second bound, and reports its group ID if the group persists. Existing Linux descendant reaping remains. New deterministic tests simulate delayed group exit with no /proc and an unexited group at deadline. Both fail before the fix. Relevant suite: 61 passed, 1 Linux-only skipped. Full final suite is in progress; see full-suite-final.log/json.

Minimum recheck: original two diagnostic paths per the existing final-review handoff, plus this process-group cleanup diff. Do not demand a replay-safe DeepSeek kernel or another experiment. No GPU, builds, source edits, broad new audit, or agents are needed for the bounded read-only check.
