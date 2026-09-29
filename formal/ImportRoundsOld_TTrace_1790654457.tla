---- MODULE ImportRoundsOld_TTrace_1790654457 ----
EXTENDS Sequences, TLCExt, ImportRoundsOld, Toolbox, Naturals, TLC

_expression ==
    LET ImportRoundsOld_TEExpression == INSTANCE ImportRoundsOld_TEExpression
    IN ImportRoundsOld_TEExpression!expression
----

_trace ==
    LET ImportRoundsOld_TETrace == INSTANCE ImportRoundsOld_TETrace
    IN ImportRoundsOld_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        gen = (1)
        /\
        exit = (0)
        /\
        r = (<<1, 1, 1>>)
        /\
        at = (<<FALSE, FALSE, FALSE>>)
        /\
        pc = (<<"check", "bar", "check">>)
        /\
        last = (<<FALSE, FALSE, FALSE>>)
        /\
        blocks = (<<<<0, 0, 0>>, <<0, 0, 0>>>>)
        /\
        nbar = (<<1, 1, 1>>)
        /\
        use = (<<1, 2, 1>>)
        /\
        done = (4)
        /\
        bleft = (<<0, 0, 0>>)
    )
----

_init ==
    /\ gen = _TETrace[1].gen
    /\ blocks = _TETrace[1].blocks
    /\ done = _TETrace[1].done
    /\ at = _TETrace[1].at
    /\ last = _TETrace[1].last
    /\ nbar = _TETrace[1].nbar
    /\ r = _TETrace[1].r
    /\ pc = _TETrace[1].pc
    /\ exit = _TETrace[1].exit
    /\ use = _TETrace[1].use
    /\ bleft = _TETrace[1].bleft
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ gen  = _TETrace[i].gen
        /\ gen' = _TETrace[j].gen
        /\ blocks  = _TETrace[i].blocks
        /\ blocks' = _TETrace[j].blocks
        /\ done  = _TETrace[i].done
        /\ done' = _TETrace[j].done
        /\ at  = _TETrace[i].at
        /\ at' = _TETrace[j].at
        /\ last  = _TETrace[i].last
        /\ last' = _TETrace[j].last
        /\ nbar  = _TETrace[i].nbar
        /\ nbar' = _TETrace[j].nbar
        /\ r  = _TETrace[i].r
        /\ r' = _TETrace[j].r
        /\ pc  = _TETrace[i].pc
        /\ pc' = _TETrace[j].pc
        /\ exit  = _TETrace[i].exit
        /\ exit' = _TETrace[j].exit
        /\ use  = _TETrace[i].use
        /\ use' = _TETrace[j].use
        /\ bleft  = _TETrace[i].bleft
        /\ bleft' = _TETrace[j].bleft

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("ImportRoundsOld_TTrace_1790654457.json", _TETrace)

=============================================================================

 Note that you can extract this module `ImportRoundsOld_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `ImportRoundsOld_TEExpression.tla` file takes precedence 
  over the module `ImportRoundsOld_TEExpression` below).

---- MODULE ImportRoundsOld_TEExpression ----
EXTENDS Sequences, TLCExt, ImportRoundsOld, Toolbox, Naturals, TLC

expression == 
    [
        \* To hide variables of the `ImportRoundsOld` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        gen |-> gen
        ,blocks |-> blocks
        ,done |-> done
        ,at |-> at
        ,last |-> last
        ,nbar |-> nbar
        ,r |-> r
        ,pc |-> pc
        ,exit |-> exit
        ,use |-> use
        ,bleft |-> bleft
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_genUnchanged |-> gen = gen'
        
        \* Format the `gen` variable as Json value.
        \* ,_genJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(gen)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_genModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].gen # _TETrace[s-1].gen
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE ImportRoundsOld_TETrace ----
\*EXTENDS IOUtils, ImportRoundsOld, TLC
\*
\*trace == IODeserialize("ImportRoundsOld_TTrace_1790654457.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE ImportRoundsOld_TETrace ----
EXTENDS ImportRoundsOld, TLC

trace == 
    <<
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"blk", "blk", "blk">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 0,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"readr", "blk", "blk">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 0,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"fadd", "blk", "blk">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 0,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"bar", "blk", "blk">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 1,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, FALSE, FALSE>>,pc |-> <<"bar", "blk", "blk">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 1,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, FALSE, FALSE>>,pc |-> <<"bar", "blk", "readr">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 1,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, FALSE, FALSE>>,pc |-> <<"bar", "readr", "readr">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 1,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, FALSE, FALSE>>,pc |-> <<"bar", "fadd", "readr">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 1,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, FALSE, FALSE>>,pc |-> <<"bar", "bar", "readr">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 2,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, FALSE, FALSE>>,pc |-> <<"bar", "bar", "fadd">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 2,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, TRUE, FALSE>>,pc |-> <<"bar", "bar", "fadd">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 2,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, TRUE, FALSE>>,pc |-> <<"bar", "bar", "bar">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 3,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 0,exit |-> 0,r |-> <<0, 0, 0>>,at |-> <<TRUE, TRUE, TRUE>>,pc |-> <<"bar", "bar", "bar">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<0, 0, 0>>,use |-> <<1, 1, 1>>,done |-> 3,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 1,exit |-> 0,r |-> <<1, 1, 1>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"check", "check", "check">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<1, 1, 1>>,use |-> <<1, 1, 1>>,done |-> 3,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 1,exit |-> 0,r |-> <<1, 1, 1>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"check", "blk", "check">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<1, 1, 1>>,use |-> <<1, 2, 1>>,done |-> 3,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 1,exit |-> 0,r |-> <<1, 1, 1>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"check", "readr", "check">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<1, 1, 1>>,use |-> <<1, 2, 1>>,done |-> 3,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 1,exit |-> 0,r |-> <<1, 1, 1>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"check", "fadd", "check">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<1, 1, 1>>,use |-> <<1, 2, 1>>,done |-> 3,bleft |-> <<0, 0, 0>>]),
    ([gen |-> 1,exit |-> 0,r |-> <<1, 1, 1>>,at |-> <<FALSE, FALSE, FALSE>>,pc |-> <<"check", "bar", "check">>,last |-> <<FALSE, FALSE, FALSE>>,blocks |-> <<<<0, 0, 0>>, <<0, 0, 0>>>>,nbar |-> <<1, 1, 1>>,use |-> <<1, 2, 1>>,done |-> 4,bleft |-> <<0, 0, 0>>])
    >>
----


=============================================================================

---- CONFIG ImportRoundsOld_TTrace_1790654457 ----
CONSTANTS
    NW = 3
    USES = 2
    MAXB = 2

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Tue Sep 29 04:01:01 UTC 2026