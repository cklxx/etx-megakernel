/-
The stream loop of etx/ops/gemv.py (dsl_gemv_rows), as a function: segments 0 .. S-1 of a task's rows, segment s in row
s / spr; wave w takes the contiguous run [w * per, min S ((w+1) * per)) and folds its segments in order, keeping a running
sum `acc` for the current row `cur` and flushing `acc` into part[cur] when the row changes and once at the end. The
per-wave partials are then summed over waves per row (the epilogue).

Theorem `gemv_correct`: for every row r, the sum over waves of part_w[r] equals the sum of the segment values v s over
the segments s of row r. This holds for any spr >= 1, any S, any per >= 1, any wave count covering S, and any values v
(integers here; the fp32 order of the real loop is what the host's --check-dsl tolerance covers). In particular the
flush-on-row-change bookkeeping and the tail path (waves whose run is shorter than the unroll) cannot lose or duplicate a
segment. Core Lean 4, no Mathlib.
-/

namespace Gemv

/-- one wave's fold: state (cur, acc, part); `part` is a function row -> partial sum -/
structure St where
  cur  : Nat
  acc  : Int
  part : Nat → Int

def step (spr : Nat) (v : Nat → Int) (st : St) (s : Nat) : St :=
  let row := s / spr
  if row = st.cur then
    { st with acc := st.acc + v s }
  else
    { cur := row, acc := v s, part := fun r => if r = st.cur then st.part r + st.acc else st.part r }

def flush (st : St) : Nat → Int := fun r => if r = st.cur then st.part r + st.acc else st.part r

/-- the wave's partials after folding the segment list `L` from an initial current row `c0` (the code uses row of the
first segment; correctness does not depend on it, since an empty acc flushes 0) -/
def wave (spr : Nat) (v : Nat → Int) (c0 : Nat) (L : List Nat) : Nat → Int :=
  flush (L.foldl (step spr v) { cur := c0, acc := 0, part := fun _ => 0 })

/-- the sum of v over the segments of `L` that lie in row r -/
def rowSum (spr : Nat) (v : Nat → Int) (r : Nat) (L : List Nat) : Int :=
  (L.filter (fun s => s / spr = r)).foldl (fun a s => a + v s) 0

/-- one step's effect on the flushed view: only row s / spr gains v s -/
theorem flush_step (spr : Nat) (v : Nat → Int) (st : St) (s r : Nat) :
    flush (step spr v st s) r = flush st r + (if s / spr = r then v s else 0) := by
  unfold step flush
  by_cases hc : s / spr = st.cur
  · rw [if_pos hc]
    by_cases hr : r = st.cur
    · subst hr; rw [if_pos rfl, if_pos rfl, if_pos hc]; simp [Int.add_assoc]
    · have hsr : ¬ s / spr = r := fun e => hr (e.symm.trans hc)
      simp only [if_neg hr, if_neg hsr, Int.add_zero]
  · rw [if_neg hc]
    by_cases hr : r = s / spr
    · have hrc : ¬ r = st.cur := fun e => hc (hr.symm.trans e)
      simp only [if_pos hr, if_pos hr.symm, if_neg hrc]
    · have hsr : ¬ s / spr = r := fun e => hr e.symm
      simp only [if_neg hr, if_neg hsr, Int.add_zero]

theorem foldl_add_shift (v : Nat → Int) (M : List Nat) (a : Int) :
    M.foldl (fun a s => a + v s) a = a + M.foldl (fun a s => a + v s) 0 := by
  induction M generalizing a with
  | nil => simp
  | cons t M ihm => simp only [List.foldl_cons]; rw [ihm (a + v t), ihm (0 + v t)]; simp [Int.add_assoc]

theorem rowSum_cons (spr : Nat) (v : Nat → Int) (r s : Nat) (L : List Nat) :
    rowSum spr v r (s :: L) = (if s / spr = r then v s else 0) + rowSum spr v r L := by
  unfold rowSum
  by_cases hs : s / spr = r
  · simp only [List.filter_cons, hs, decide_true, List.foldl_cons, Int.zero_add, if_true]
    rw [foldl_add_shift]
  · simp [List.filter_cons, hs]

/-- invariant of the fold: flushing the state gives the row sums of the processed list, on top of the start state -/
theorem fold_inv (spr : Nat) (v : Nat → Int) (L : List Nat) (st : St) (r : Nat) :
    flush (L.foldl (step spr v) st) r = flush st r + rowSum spr v r L := by
  induction L generalizing st with
  | nil => simp [rowSum]
  | cons s L ih =>
    rw [List.foldl_cons, ih, flush_step, rowSum_cons, Int.add_assoc]

/-- a wave's partials are exactly the row sums of its segment list -/
theorem wave_rowSum (spr : Nat) (v : Nat → Int) (c0 : Nat) (L : List Nat) (r : Nat) :
    wave spr v c0 L r = rowSum spr v r L := by
  unfold wave
  rw [fold_inv]
  simp [flush]

/-- the wave runs: wave w owns segments [w * per, min S ((w + 1) * per)) -/
def run (S per w : Nat) : List Nat :=
  (List.range (min S ((w + 1) * per) - w * per)).map (fun i => w * per + i)

theorem flatMap_congr' {α β : Type} (L : List α) (f g : α → List β) (h : ∀ w ∈ L, f w = g w) :
    L.flatMap f = L.flatMap g := by
  induction L with
  | nil => rfl
  | cons a L ih =>
    simp only [List.flatMap_cons]
    rw [h a (List.mem_cons_self ..), ih (fun w hw => h w (List.mem_cons_of_mem a hw))]

/-- the concatenation of the runs of waves 0 .. nw-1 is the segment list 0 .. S-1 when the runs cover S -/
theorem runs_cover (S per nw : Nat) (hper : per ≥ 1) (hcov : nw * per ≥ S) :
    (List.range nw).flatMap (run S per) = List.range S := by
  -- by induction on nw: range (nw+1) = range nw ++ [nw]; run nw = [nw*per, min S ((nw+1)*per))
  induction nw generalizing S with
  | zero =>
    simp at hcov; subst hcov; rfl
  | succ n ih =>
    rw [List.range_succ, List.flatMap_append]
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
    by_cases h : S ≤ n * per
    · rw [ih S h]
      have : run S per n = [] := by
        unfold run
        have : min S ((n + 1) * per) - n * per = 0 := by omega
        rw [this]; rfl
      rw [this, List.append_nil]
    · -- segments below n*per come from the first n waves (on the prefix S' = n*per), the rest from wave n
      have hS' : n * per ≥ n * per := Nat.le_refl _
      have hpre := ih (n * per) hS'
      -- run S per w for w < n equals run (n*per) per w since both are capped below n*per
      have hsame : (List.range n).flatMap (run S per) = (List.range n).flatMap (run (n * per) per) := by
        apply flatMap_congr'
        intro w hw
        have hw' : w < n := List.mem_range.mp hw
        unfold run
        have : min S ((w + 1) * per) = min (n * per) ((w + 1) * per) := by
          have : (w + 1) * per ≤ n * per := Nat.mul_le_mul_right _ hw'
          omega
        rw [this]
      rw [hsame, hpre]
      unfold run
      have hmin : min S ((n + 1) * per) = S := by omega
      rw [hmin]
      -- range (n*per) ++ map (n*per + ·) (range (S - n*per)) = range S
      have e := @List.range_add (n * per) (S - n * per)
      have hs : n * per + (S - n * per) = S := by omega
      rw [hs] at e
      exact e.symm

theorem rowSum_append (spr : Nat) (v : Nat → Int) (r : Nat) (A B : List Nat) :
    rowSum spr v r (A ++ B) = rowSum spr v r A + rowSum spr v r B := by
  unfold rowSum
  rw [List.filter_append, List.foldl_append, foldl_add_shift]

theorem rowSum_flatMap (spr : Nat) (v : Nat → Int) (r : Nat) (f : Nat → List Nat) (W : List Nat) :
    rowSum spr v r (W.flatMap f) = (W.map (fun w => rowSum spr v r (f w))).foldl (· + ·) 0 := by
  induction W with
  | nil => simp [rowSum]
  | cons w W ih =>
    rw [List.flatMap_cons, rowSum_append, ih]
    simp only [List.map_cons, List.foldl_cons]
    have key : ∀ (M : List Int) (a : Int), M.foldl (· + ·) a = a + M.foldl (· + ·) 0 := by
      intro M; induction M with
      | nil => intro a; simp
      | cons t M ihm => intro a; simp only [List.foldl_cons]; rw [ihm (a + t), ihm (0 + t)]; simp [Int.add_assoc]
    rw [Int.zero_add, key _ (rowSum spr v r (f w))]

/-- THE THEOREM: summing every wave's partial for row r gives the row's dot product (the sum of its segments'
values), for any wave count that covers the S segments -/
theorem gemv_correct (spr S per nw : Nat) (v : Nat → Int) (c0 : Nat → Nat) (r : Nat)
    (hper : per ≥ 1) (hcov : nw * per ≥ S) :
    ((List.range nw).map (fun w => wave spr v (c0 w) (run S per w) r)).foldl (· + ·) 0
      = rowSum spr v r (List.range S) := by
  rw [← runs_cover S per nw hper hcov, rowSum_flatMap]
  congr 1
  apply List.map_congr_left
  intro w _
  exact wave_rowSum spr v (c0 w) (run S per w) r

end Gemv

#print axioms Gemv.gemv_correct
