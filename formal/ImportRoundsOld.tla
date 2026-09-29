-------------------------- MODULE ImportRoundsOld --------------------------
(* THE FIRST VERSION of the protocol (2026-09-28, replaced after a review): a wave leaves when it reads done = NW after
   a barrier, and wave 1 resets done at the next use's begin. TLC must find its race (this module is the negative test
   of the model: if it passed, the model would be too weak to mean anything).
   Original description:
   NW waves of one workgroup run a chained sequence of USES imported blocks. In use u wave w takes blocks[u][w]
   counted barriers (its block's __syncthreads; 0 for an idle wave or an early return), then etx_imp_rounds_end():
     r    := nbar[w]                                  (its own round number, wave-private counter)
     if fetch_add(done, 1) = NW - 1 then done := 0; exit := r + 1     (the last wave to finish decides the exit round)
     loop: counted_barrier(); r := r + 1; if exit = r then break
   where counted_barrier = s_barrier (all NW waves rendezvous) followed by nbar[w] := nbar[w] + 1.
   Shared memory operations are atomic and may interleave arbitrarily between barriers. The claims checked by TLC:
     NoDeadlock       every reachable state has a successor until all waves are done (TLC's deadlock check)
     SameRounds       when a use ends, every wave has taken the same number of barriers (Invariant at Done, and at
                      every use boundary: SameNbarAtBoundary)
     Termination      with weak fairness every wave finishes every use                                            *)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS NW, USES, MAXB
ASSUME NW \in Nat /\ NW >= 1 /\ USES >= 1

Waves == 1..NW

VARIABLES
  blocks,     \* [use][wave] -> number of block barriers (chosen at start, fixed)
  pc,         \* wave -> "blk" | "readr" | "fadd" | "reset" | "sexit" | "bar" | "post" | "check" | "done"
  use,        \* wave -> current use, 1..USES+1
  bleft,      \* wave -> block barriers still to take in this use
  r,          \* wave -> local round number
  last,       \* wave -> TRUE iff this wave's fetch_add completed the count
  nbar,       \* wave -> counted barriers taken (wave-private, but stored in LDS)
  done, exit, \* the shared words
  at,         \* wave -> TRUE iff waiting at the hardware barrier
  gen         \* barrier generation (a rendezvous releases all at once)

vars == <<blocks, pc, use, bleft, r, last, nbar, done, exit, at, gen>>

Init ==
  /\ blocks \in [1..USES -> [Waves -> 0..MAXB]]
  /\ pc = [w \in Waves |-> "blk"]
  /\ use = [w \in Waves |-> 1]
  /\ bleft = [w \in Waves |-> blocks[1][w]]
  /\ r = [w \in Waves |-> 0]
  /\ last = [w \in Waves |-> FALSE]
  /\ nbar = [w \in Waves |-> 0]
  /\ done = 0 /\ exit = 0
  /\ at = [w \in Waves |-> FALSE]
  /\ gen = 0

\* --- the hardware barrier: a wave arrives; when all NW have arrived, one step releases them all
Arrive(w) ==
  /\ ~at[w]
  /\ (pc[w] = "blk" /\ bleft[w] > 0) \/ pc[w] = "bar"
  /\ at' = [at EXCEPT ![w] = TRUE]
  /\ UNCHANGED <<blocks, pc, use, bleft, r, last, nbar, done, exit, gen>>

Release ==
  /\ \A w \in Waves : at[w]
  /\ at' = [w \in Waves |-> FALSE]
  /\ gen' = gen + 1
  \* after the barrier every wave counts it (nbar[w] := nbar[w] + 1, wave-private) and moves on
  /\ nbar' = [w \in Waves |-> nbar[w] + 1]
  /\ bleft' = [w \in Waves |-> IF pc[w] = "blk" THEN bleft[w] - 1 ELSE bleft[w]]
  /\ r' = [w \in Waves |-> IF pc[w] = "bar" THEN r[w] + 1 ELSE r[w]]
  /\ pc' = [w \in Waves |-> IF pc[w] = "bar" THEN "check" ELSE pc[w]]
  /\ UNCHANGED <<blocks, use, last, done, exit>>

\* --- the block is done (no barriers left): enter rounds_end
BlockDone(w) ==
  /\ pc[w] = "blk" /\ bleft[w] = 0 /\ ~at[w]
  /\ pc' = [pc EXCEPT ![w] = "readr"]
  /\ UNCHANGED <<blocks, use, bleft, r, last, nbar, done, exit, at, gen>>

ReadR(w) ==
  /\ pc[w] = "readr"
  /\ r' = r
  /\ pc' = [pc EXCEPT ![w] = "fadd"]
  /\ UNCHANGED <<blocks, use, bleft, last, nbar, done, exit, at, gen>>

FetchAdd(w) ==
  /\ pc[w] = "fadd"
  /\ done' = done + 1
  /\ last' = last
  /\ pc' = [pc EXCEPT ![w] = "bar"]
  /\ UNCHANGED <<blocks, use, bleft, r, nbar, exit, at, gen>>

ResetDone(w) ==
  /\ pc[w] = "reset"
  /\ done' = 0
  /\ pc' = [pc EXCEPT ![w] = "blk"]
  /\ UNCHANGED <<blocks, use, bleft, r, last, nbar, exit, at, gen>>

StoreExit(w) ==
  /\ pc[w] = "sexit"
  /\ exit' = r[w] + 1
  /\ pc' = [pc EXCEPT ![w] = "bar"]
  /\ UNCHANGED <<blocks, use, bleft, r, last, nbar, done, at, gen>>

\* after a counted barrier of the exit loop: leave when exit = r, else another barrier
Check(w) ==
  /\ pc[w] = "check"
  /\ IF done = NW
       THEN /\ use' = [use EXCEPT ![w] = use[w] + 1]
            /\ IF use[w] + 1 > USES
                 THEN /\ pc' = [pc EXCEPT ![w] = "done"] /\ bleft' = bleft
                 ELSE /\ pc' = [pc EXCEPT ![w] = IF w = 1 THEN "reset" ELSE "blk"]
                      /\ bleft' = [bleft EXCEPT ![w] = blocks[use[w] + 1][w]]
       ELSE /\ pc' = [pc EXCEPT ![w] = "bar"] /\ use' = use /\ bleft' = bleft
  /\ UNCHANGED <<blocks, r, last, nbar, done, exit, at, gen>>

Next ==
  \/ Release
  \/ \E w \in Waves : Arrive(w) \/ BlockDone(w) \/ ReadR(w) \/ FetchAdd(w) \/ ResetDone(w) \/ StoreExit(w) \/ Check(w)
  \/ (\A w \in Waves : pc[w] = "done") /\ UNCHANGED vars      \* the terminal state stutters: no false deadlock

Spec == Init /\ [][Next]_vars /\ \A w \in Waves : WF_vars(Arrive(w) \/ BlockDone(w) \/ ReadR(w) \/ FetchAdd(w) \/ ResetDone(w) \/ StoreExit(w) \/ Check(w)) /\ WF_vars(Release)

\* --- properties
TypeOK ==
  /\ done \in 0..NW /\ exit \in Nat
  /\ \A w \in Waves : pc[w] \in {"blk", "readr", "fadd", "reset", "sexit", "bar", "check", "done"}

\* a wave that leaves a use has taken exactly as many barriers as every wave still inside or past that use will have
\* taken when it leaves: at the moment a wave is in "blk" or "done" of use u+1 (it left use u), no other wave can still be
\* at an earlier barrier count of use u -> all waves that have left use u agree on nbar
SameNbarAtBoundary ==
  \A v, w \in Waves : (use[v] = use[w] /\ pc[v] \in {"blk", "done"} /\ pc[w] \in {"blk", "done"}) => nbar[v] = nbar[w]

\* no wave leaves a use while another wave of the same use still has block barriers to take (they would have no partner)
NoOrphanBarrier ==
  \A v, w \in Waves : (pc[v] \in {"blk", "done"} /\ use[v] > use[w]) => bleft[w] = 0 \/ pc[w] # "blk"

AllDone == \A w \in Waves : pc[w] = "done"
Termination == <>AllDone
=============================================================================
