---------------------------- MODULE EventFlush ----------------------------
(* ETX's DEVICE-scope event with the last-arriver flush (etx/runtime/include/etx/primitives.h,
   etx_arrive_flush_DEVICE_s) and its consumers (etx_wait_DEVICE): P producers spread over D domains (XCDs), SHARE
   producers per domain. A producer finishes its task, drains its stores into its domain's L2 (RELEASE_DOMAIN) and
   bumps the domain's sub-counter; the arrival that completes the domain's share writes the domain's L2 back
   (RELEASE_DEVICE) and subtracts the whole share from the global event word. A consumer polls the global word until
   it is <= 0, then invalidates its caches (ACQUIRE_DEVICE) and reads the producers' outputs.
   Modelled memory: every producer's output is "in L2 of its domain" after the release and "visible device-wide" after
   a write-back of that domain; a consumer's read after its acquire returns visible data only.
   Claims: Safety  a consumer that has passed its wait sees every producer's output (no stale read)
           Exact   the global word reaches 0 exactly when the last domain flushed (never negative)
           Termination  every consumer eventually wakes                                                          *)
EXTENDS Naturals, FiniteSets

CONSTANTS D, SHARE, C
ASSUME D >= 1 /\ SHARE >= 1 /\ C >= 1

Producers == 1..(D * SHARE)
Domain(p) == ((p - 1) \div SHARE) + 1
Consumers == 1..C

VARIABLES
  ppc,      \* producer -> "work" | "rel" | "sub" | "flush" | "done"
  cpc,      \* consumer -> "wait" | "acq" | "read" | "done"
  ev,       \* the global event word (starts at the producer count)
  sub,      \* domain -> sub-counter
  inL2,     \* producer -> TRUE once its output is in its domain's L2
  wb,       \* domain -> number of write-backs done (a write-back publishes everything then in that L2)
  visible,  \* producer -> TRUE once its output is visible device-wide
  seen,     \* consumer -> the set of producers whose output it read as visible
  last      \* producer -> TRUE iff its sub-counter increment completed the domain's share

vars == <<ppc, cpc, ev, sub, inL2, wb, visible, seen, last>>

Init ==
  /\ ppc = [p \in Producers |-> "work"]
  /\ cpc = [c \in Consumers |-> "wait"]
  /\ ev = D * SHARE
  /\ sub = [d \in 1..D |-> 0]
  /\ inL2 = [p \in Producers |-> FALSE]
  /\ wb = [d \in 1..D |-> 0]
  /\ visible = [p \in Producers |-> FALSE]
  /\ seen = [c \in Consumers |-> {}]
  /\ last = [p \in Producers |-> FALSE]

\* producer p finishes its task: its output is written (into its own L2 after the domain release)
Release(p) ==
  /\ ppc[p] = "work"
  /\ inL2' = [inL2 EXCEPT ![p] = TRUE]
  /\ ppc' = [ppc EXCEPT ![p] = "sub"]
  /\ UNCHANGED <<cpc, ev, sub, wb, visible, seen, last>>

\* atomicAdd on the domain's sub-counter; the last of the share goes on to flush
SubArrive(p) ==
  /\ ppc[p] = "sub"
  /\ LET d == Domain(p) IN
     /\ sub' = [sub EXCEPT ![d] = sub[d] + 1]
     /\ last' = [last EXCEPT ![p] = (sub[d] + 1 = SHARE)]
     /\ ppc' = [ppc EXCEPT ![p] = IF sub[d] + 1 = SHARE THEN "flush" ELSE "done"]
  /\ UNCHANGED <<cpc, ev, inL2, wb, visible, seen>>

\* RELEASE_DEVICE: the domain's L2 is written back (everything released into it so far becomes visible), then the
\* global word loses the domain's share (one atomic)
Flush(p) ==
  /\ ppc[p] = "flush"
  /\ LET d == Domain(p) IN
     /\ visible' = [q \in Producers |-> visible[q] \/ (Domain(q) = d /\ inL2[q])]
     /\ wb' = [wb EXCEPT ![d] = wb[d] + 1]
     /\ ev' = ev - SHARE
     /\ ppc' = [ppc EXCEPT ![p] = "done"]
  /\ UNCHANGED <<cpc, sub, inL2, seen, last>>

\* the consumer's poll: it leaves the wait only when the word is <= 0
Wake(c) ==
  /\ cpc[c] = "wait" /\ ev <= 0
  /\ cpc' = [cpc EXCEPT ![c] = "acq"]
  /\ UNCHANGED <<ppc, ev, sub, inL2, wb, visible, seen, last>>

Acquire(c) ==
  /\ cpc[c] = "acq"
  /\ cpc' = [cpc EXCEPT ![c] = "read"]
  /\ UNCHANGED <<ppc, ev, sub, inL2, wb, visible, seen, last>>

\* the read after the acquire returns what is visible device-wide at that moment
Read(c) ==
  /\ cpc[c] = "read"
  /\ seen' = [seen EXCEPT ![c] = {p \in Producers : visible[p]}]
  /\ cpc' = [cpc EXCEPT ![c] = "done"]
  /\ UNCHANGED <<ppc, ev, sub, inL2, wb, visible, last>>

AllDone == (\A p \in Producers : ppc[p] = "done") /\ (\A c \in Consumers : cpc[c] = "done")

Next ==
  \/ \E p \in Producers : Release(p) \/ SubArrive(p) \/ Flush(p)
  \/ \E c \in Consumers : Wake(c) \/ Acquire(c) \/ Read(c)
  \/ AllDone /\ UNCHANGED vars

Spec == Init /\ [][Next]_vars
        /\ \A p \in Producers : WF_vars(Release(p) \/ SubArrive(p) \/ Flush(p))
        /\ \A c \in Consumers : WF_vars(Wake(c) \/ Acquire(c) \/ Read(c))

\* --- properties
NeverNegative == ev >= 0
\* the word is 0 only when every producer's output is visible; a consumer past its wait sees all of them
ZeroMeansAllVisible == ev = 0 => \A p \in Producers : visible[p]
NoStaleRead == \A c \in Consumers : cpc[c] = "done" => seen[c] = Producers
\* the global word is decremented exactly once per domain
OneFlushPerDomain == \A d \in 1..D : wb[d] <= 1
Termination == <>AllDone
=============================================================================
