-------------------------------- MODULE DataVis --------------------------------
(* Data visibility across the megakernel's phase boundary on a CDNA3 device (per-XCD
   write-back L2, no cross-XCD probes), for the ETX_VL_COH redesign: which producer
   release is needed for each combination of
     producer class: "plain" (cached stores, dirty lines stay in its XCD's L2) or
                     "coherent" (sc0 sc1 stores: memory is updated past L2),
     consumer class: "plain" (cached loads behind a buffer_inv acquire that drops
                     CLEAN lines; dirty lines survive as the newest data) or
                     "coherent" (sc0 sc1 loads: straight from memory, both L2s bypassed),
     placement: same XCD (DOMAIN events; the scope pass guarantees this) or cross XCD.
   One cache line, values 0 (stale) / 1 (fresh); both L2s start with a stale CLEAN copy
   (the activation buffers are reused every layer). The emitted policy (Strict = TRUE):
     wbl2 on release  <=>  producer plain  /\  (consumer coherent  \/  cross XCD).
   Strict = FALSE drops the "consumer coherent" half on same-XCD events (the code before
   this change): TLC must find the stale read (the imported -> coherent DSL trap;
   reduce -> o_proj is that edge when they share an XCD). *)
EXTENDS Naturals

CONSTANT Strict

VARIABLES pc,          \* "store" -> "release" -> "read" -> "done"
          pclass, cclass, samexcd,
          mem,         \* the line in memory / MALL (coherent point)
          l2,          \* [xcd \in {0, 1} |-> [val: 0..1, valid: BOOLEAN, dirty: BOOLEAN]]
          readv

vars == <<pc, pclass, cclass, samexcd, mem, l2, readv>>

CXcd == IF samexcd THEN 0 ELSE 1     \* the producer runs on XCD 0

WbRequired ==
  /\ pclass = "plain"
  /\ IF Strict THEN (cclass = "coherent" \/ ~samexcd)
               ELSE ~samexcd

Init ==
  /\ pc = "store"
  /\ pclass \in {"plain", "coherent"}
  /\ cclass \in {"plain", "coherent"}
  /\ samexcd \in BOOLEAN
  /\ mem = 0
  /\ l2 = [x \in {0, 1} |-> [val |-> 0, valid |-> TRUE, dirty |-> FALSE]]
  /\ readv = 0

(* The producer's store. Plain: the new value sits dirty in its XCD's L2 only.
   Coherent: memory is updated; the worst legal fate of the producer's own L2 copy is
   an untouched stale CLEAN line (write-through never leaves it dirty). *)
Store ==
  /\ pc = "store"
  /\ IF pclass = "plain"
       THEN l2' = [l2 EXCEPT ![0] = [val |-> 1, valid |-> TRUE, dirty |-> TRUE]] /\ mem' = mem
       ELSE mem' = 1 /\ l2' = l2
  /\ pc' = "release"
  /\ UNCHANGED <<pclass, cclass, samexcd, readv>>

(* The release (then the arrive; the event flag itself is the already verified
   ImportRounds/EventFlush protocol, so it is not modelled here). *)
Release ==
  /\ pc = "release"
  /\ IF WbRequired /\ l2[0].dirty
       THEN mem' = l2[0].val /\ l2' = [l2 EXCEPT ![0].dirty = FALSE]
       ELSE mem' = mem /\ l2' = l2
  /\ pc' = "read"
  /\ UNCHANGED <<pclass, cclass, samexcd, readv>>

(* The consumer, after its wait. Plain: buffer_inv drops clean lines (dirty ones stay:
   they are the newest data), then a cached read (line fill from memory on a miss).
   Coherent: the read bypasses both L2s and takes memory as it is. *)
Read ==
  /\ pc = "read"
  /\ IF cclass = "plain"
       THEN LET inv == [l2 EXCEPT ![CXcd].valid = @ /\ l2[CXcd].dirty]
                v   == IF inv[CXcd].valid THEN inv[CXcd].val ELSE mem
            IN  readv' = v /\ l2' = [inv EXCEPT ![CXcd] = [val |-> v, valid |-> TRUE, dirty |-> inv[CXcd].valid /\ inv[CXcd].dirty]]
       ELSE readv' = mem /\ l2' = l2
  /\ mem' = mem
  /\ pc' = "done"
  /\ UNCHANGED <<pclass, cclass, samexcd>>

Next == Store \/ Release \/ Read \/ (pc = "done" /\ UNCHANGED vars)

Spec == Init /\ [][Next]_vars

(* The consumer always reads the produced value. *)
Fresh == pc = "done" => readv = 1

=================================================================================
