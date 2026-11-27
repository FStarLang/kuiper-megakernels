module KPlay.Spec.Bridge

(* Bridge from the flat program postcondition KPlay.SharedSpec, which the
   program proofs conclude, to the typed specification KPlay.Spec.

   inputs_of / outputs_of are layout views only (row-major flat buffers).
   program_postcondition_iff shows that, on inputs of the right lengths, the
   flat postcondition holds exactly when the outputs have their lengths and
   the typed postcondition holds of the views: the typed spec is neither
   weaker nor stronger than what the program proofs establish. *)

open Kuiper
open Kuiper.Approximates
open KPlay.Spec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64
module SS = KPlay.SharedSpec
module M = FStar.Math.Lemmas

#set-options "--fuel 1 --ifuel 1 --z3rlimit 20"

(* ------------------------------------------------------------------ *)
(* Views (layout only)                                                  *)

let entry (#a:Type) (default:a) (s:S.seq a) (k:nat) : a =
  if k < S.length s then S.index s k else default

let vector_view (#a:Type0) (default:a) (s:S.seq a) (n:nat) : vector a n =
  fun i -> entry default s i

let matrix_view (#a:Type0) (default:a) (s:S.seq a) (rows cols:nat) : matrix a rows cols =
  fun i j -> entry default s (i * cols + j)

(* tokens [256, 512] and router_weights [512, 8], row-major. *)
let inputs_of (x w:S.seq BF.t) : inputs = {
  tokens = matrix_view zero x token_count hidden_size;
  router_weights = matrix_view zero w hidden_size expert_count;
}

(* demand [8], token indices [8, 32], probabilities [8, 32], survival
   [256, 8], row-major. *)
let outputs_of (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t) (surv:S.seq I32.t)
  : outputs = {
  expert_demand = vector_view 0l demand expert_count;
  token_indices = matrix_view 0L idx expert_count capacity;
  probabilities = matrix_view zero probs expert_count capacity;
  survival = matrix_view 0l surv token_count expert_count;
}

let input_lengths (x w:S.seq BF.t) : prop =
  S.length x == token_count * hidden_size /\ S.length w == hidden_size * expert_count

let output_lengths (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t) (surv:S.seq I32.t)
  : prop =
  S.length demand == expert_count /\ S.length idx == expert_count * capacity /\
  S.length probs == expert_count * capacity /\ S.length surv == token_count * expert_count

(* ------------------------------------------------------------------ *)
(* Index arithmetic                                                     *)

let split_index (cols:pos) (i:nat) (j:nat{j < cols})
  : Lemma ((i * cols + j) / cols == i /\ (i * cols + j) % cols == j)
  = M.lemma_div_plus j i cols;
    M.small_div j cols;
    M.lemma_mod_plus j i cols;
    M.small_mod j cols

let join_index (cols:pos) (k:nat)
  : Lemma ((k / cols) * cols + k % cols == k /\ k % cols < cols)
  = M.euclidean_division_definition k cols

(* ------------------------------------------------------------------ *)
(* Real sums: flat (fold, dot_prefix) versus typed (sum)                *)

let rec sum_ext (#n:nat) (f g:vector real n) (k:nat{k <= n})
  : Lemma (requires forall (i:index n). i < k ==> f i == g i)
          (ensures sum f k == sum g k) (decreases k)
  = if k > 0 then sum_ext f g (k - 1)

let rec rsum_prefix (#n:nat) (s:S.seq real) (v:vector real n) (k:nat{k <= n})
  : Lemma (requires S.length s == n /\ (forall (i:index n). S.index s i == v i))
          (ensures rsum (S.slice s 0 k) == sum v k) (decreases k)
  = if k > 0 then begin
      rsum_prefix s v (k - 1);
      Kuiper.Seq.Common.lemma_seq_fold_left_slice 0.0R ( +. ) s 0 (k - 1)
    end

let rsum_sum (#n:nat) (s:S.seq real) (v:vector real n)
  : Lemma (requires S.length s == n /\ (forall (i:index n). S.index s i == v i))
          (ensures rsum s == sum v n)
  = rsum_prefix s v n;
    S.lemma_eq_elim (S.slice s 0 n) s

let rec dot_sum (rx rw:S.seq real) (r:router_inputs real)
  (t:index token_count) (e:index expert_count) (n:nat{n <= hidden_size})
  : Lemma (requires forall (h:index hidden_size).
             SS.ridx rx (t * hidden_size + h) == r.tokens t h /\
             SS.ridx rw (h * expert_count + e) == r.router_weights h e)
          (ensures SS.dot_prefix SS.hidden SS.experts rx rw t e n ==
                   sum (products r.tokens r.router_weights t e) n)
          (decreases n)
  = if n > 0 then dot_sum rx rw r t e (n - 1)

(* One flat reference softmax row equals the typed router probability row. *)
let reference_row (rx rw:S.seq real) (r:router_inputs real) (t:index token_count)
  : Lemma (requires (forall (t:index token_count) (h:index hidden_size).
                       SS.ridx rx (t * hidden_size + h) == r.tokens t h) /\
                    (forall (h:index hidden_size) (e:index expert_count).
                       SS.ridx rw (h * expert_count + e) == r.router_weights h e))
          (ensures (let row = SS.ref_softmax (SS.row_of (SS.ref_scores rx rw) t) in
                    S.length row == expert_count /\
                    (forall (e:index expert_count). S.index row e == router_probabilities r t e)))
  = let rs = SS.row_of (SS.ref_scores rx rw) t in
    let v = scores r.tokens r.router_weights t in
    let score (e:index expert_count) : Lemma (S.index rs e == v e) =
      split_index expert_count t e;
      dot_sum rx rw r t e hidden_size
    in
    Classical.forall_intro score;
    rsum_sum (SS.exp_row rs) (exps v)

(* ------------------------------------------------------------------ *)
(* Capacity routing: flat closed forms versus typed equations           *)

let same_selection (sel:S.seq int) (selected:vector (index expert_count) token_count) : prop =
  S.length sel == token_count /\ (forall (t:index token_count). S.index sel t == selected t)

let rec count_assigned (sel:S.seq int) (selected:vector (index expert_count) token_count)
  (e:index expert_count) (n:nat{n <= token_count})
  : Lemma (requires same_selection sel selected)
          (ensures SS.count_upto sel e n == assigned_before selected e n) (decreases n)
  = if n > 0 then count_assigned sel selected e (n - 1)

let rec survivors_capped (selected:vector (index expert_count) token_count)
  (e:index expert_count) (n:nat{n <= token_count})
  : Lemma (ensures survivors_before selected e n ==
                   (if assigned_before selected e n <= capacity
                    then assigned_before selected e n else capacity))
          (decreases n)
  = if n > 0 then survivors_capped selected e (n - 1)

let kept_survives (sel:S.seq int) (selected:vector (index expert_count) token_count)
  (t:index token_count) (e:index expert_count)
  : Lemma (requires same_selection sel selected)
          (ensures (S.index sel t = e && SS.kept sel t) == (survival_of selected t e = 1) /\
                   survival_of selected t e <= 1)
  = count_assigned sel selected e t

let rec slot_kept (sel:S.seq int) (selected:vector (index expert_count) token_count)
  (e:index expert_count) (c:index capacity) (n:nat{n <= token_count})
  : Lemma (requires same_selection sel selected)
          (ensures SS.slot_token sel e c n == kept_token selected e c n) (decreases n)
  = if n > 0 then begin
      count_assigned sel selected e (n - 1);
      survivors_capped selected e (n - 1);
      slot_kept sel selected e c (n - 1)
    end

(* ------------------------------------------------------------------ *)
(* Flat witnesses versus the typed routing record                      *)

let index_bound (rows cols:pos) (i:index rows) (j:index cols)
  : Lemma (i * cols + j < rows * cols)
  = M.lemma_mult_le_right cols i (rows - 1)

let div_index (rows cols:pos) (k:nat{k < rows * cols})
  : Lemma (k / cols < rows /\ k % cols < cols /\ (k / cols) * cols + k % cols == k)
  = join_index cols k;
    if k / cols >= rows then M.lemma_mult_le_right cols rows (k / cols)

(* The flat buffers rows / sel / comb hold exactly the values of run. *)
let corresponds (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) (run:routing) : prop =
  S.length rows == token_count * expert_count /\ S.length comb == token_count /\
  same_selection sel run.selected /\
  (forall (t:index token_count) (e:index expert_count).
     S.index rows (t * expert_count + e) == run.probability_rows t e) /\
  (forall (t:index token_count). S.index comb t == run.combine_probability t)

(* Forward: read the routing record off the flat witnesses. *)
let routing_of (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) : routing = {
  probability_rows = matrix_view zero rows token_count expert_count;
  selected = (fun t -> let j = entry 0 sel t in
                       (if 0 <= j && j < expert_count then j else 0) <: index expert_count);
  combine_probability = vector_view zero comb token_count;
}

(* Converse: lay the routing record out as flat witnesses. *)
let rows_of (run:routing) : GTot (S.seq F32.t) =
  S.init_ghost (token_count * expert_count) (fun k ->
    div_index token_count expert_count k;
    run.probability_rows (k / expert_count) (k % expert_count))

let sel_of (run:routing) : GTot (S.seq int) =
  S.init_ghost token_count (fun t -> (run.selected t <: int))

let comb_of (run:routing) : GTot (S.seq F32.t) =
  S.init_ghost token_count (fun t -> run.combine_probability t)

let corresponds_routing_of (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t)
  : Lemma (requires S.length rows == token_count * expert_count /\
                    S.length sel == token_count /\ S.length comb == token_count /\
                    (forall (t:index token_count). 0 <= S.index sel t /\ S.index sel t < expert_count))
          (ensures corresponds rows sel comb (routing_of rows sel comb))
  = Classical.forall_intro_2 (index_bound token_count expert_count)

let corresponds_rows_of (run:routing)
  : Lemma (corresponds (rows_of run) (sel_of run) (comb_of run) run)
  = let at (t:index token_count) (e:index expert_count)
      : Lemma (S.index (rows_of run) (t * expert_count + e) == run.probability_rows t e)
      = index_bound token_count expert_count t e;
        split_index expert_count t e
    in
    Classical.forall_intro_2 at

let row_entries (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) (run:routing)
  (t:index token_count)
  : Lemma (requires corresponds rows sel comb run)
          (ensures S.length (SS.row_of rows t) == expert_count /\
                   (forall (i:index expert_count). S.index (SS.row_of rows t) i == run.probability_rows t i))
  = ()

(* ------------------------------------------------------------------ *)
(* Component equivalences, under corresponds                            *)

let selection_iff (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) (run:routing)
  : Lemma (requires corresponds rows sel comb run)
          (ensures (forall (t:nat). t < SS.tokens ==>
                      SS.allowed_choice (SS.row_of rows t) (S.index sel t)) <==>
                   selection run)
  = let at (t:index token_count)
      : Lemma (SS.allowed_choice (SS.row_of rows t) (S.index sel t) <==>
               top1_relation (run.probability_rows t) (run.selected t))
      = row_entries rows sel comb run t
    in
    Classical.forall_intro at

let combine_iff (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) (run:routing)
  : Lemma (requires corresponds rows sel comb run)
          (ensures (forall (t:nat). t < SS.tokens /\ SS.kept sel t ==>
                      SS.selected_combine_ok (SS.row_of rows t) (S.index sel t) (S.index comb t)) <==>
                   (forall (t:index token_count).
                      survival_of run.selected t (run.selected t) = 1 ==>
                      combine_relation (run.probability_rows t (run.selected t))
                                       (run.combine_probability t)))
  = let at (t:index token_count)
      : Lemma ((SS.kept sel t ==>
                SS.selected_combine_ok (SS.row_of rows t) (S.index sel t) (S.index comb t)) <==>
               (survival_of run.selected t (run.selected t) = 1 ==>
                combine_relation (run.probability_rows t (run.selected t))
                                 (run.combine_probability t)))
      = row_entries rows sel comb run t;
        kept_survives sel run.selected t (run.selected t)
    in
    Classical.forall_intro at

let demand_iff (demand:S.seq I32.t) (sel:S.seq int) (s:vector (index expert_count) token_count)
  : Lemma (requires S.length demand == expert_count /\ same_selection sel s)
          (ensures (forall (e:nat). e < SS.experts ==>
                      SS.count_v (S.index demand e) == SS.count_upto sel e SS.tokens) <==>
                   (forall (e:index expert_count).
                      I32.v (vector_view 0l demand expert_count e) == KPlay.Spec.demand s e))
  = let at (e:index expert_count)
      : Lemma (SS.count_upto sel e SS.tokens == KPlay.Spec.demand s e)
      = count_assigned sel s e token_count
    in
    Classical.forall_intro at

let token_indices_iff (idx:S.seq I64.t) (sel:S.seq int) (s:vector (index expert_count) token_count)
  : Lemma (requires S.length idx == expert_count * capacity /\ same_selection sel s)
          (ensures (forall (k:nat). k < SS.slots ==>
                      I64.v (S.index idx k) ==
                        SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens) <==>
                   (forall (e:index expert_count) (c:index capacity).
                      I64.v (matrix_view 0L idx expert_count capacity e c) ==
                        kept_token s e c token_count))
  = introduce (forall (k:nat). k < SS.slots ==>
                 I64.v (S.index idx k) ==
                   SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens) ==>
              (forall (e:index expert_count) (c:index capacity).
                 I64.v (matrix_view 0L idx expert_count capacity e c) ==
                   kept_token s e c token_count)
    with begin
      let at (e:index expert_count) (c:index capacity)
        : Lemma (I64.v (matrix_view 0L idx expert_count capacity e c) ==
                   kept_token s e c token_count)
        = index_bound expert_count capacity e c;
          split_index capacity e c;
          slot_kept sel s e c token_count
      in
      Classical.forall_intro_2 at
    end;
    introduce (forall (e:index expert_count) (c:index capacity).
                 I64.v (matrix_view 0L idx expert_count capacity e c) ==
                   kept_token s e c token_count) ==>
              (forall (k:nat). k < SS.slots ==>
                 I64.v (S.index idx k) ==
                   SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens)
    with begin
      let at (k:nat)
        : Lemma (k < SS.slots ==>
                 I64.v (S.index idx k) ==
                   SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens)
        = if k < SS.slots then begin
            div_index expert_count capacity k;
            slot_kept sel s (k / capacity) (k % capacity) token_count
          end
      in
      Classical.forall_intro at
    end

let kept_probabilities_iff (probs:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t)
  (s:vector (index expert_count) token_count) (combine:vector F32.t token_count)
  : Lemma (requires S.length probs == expert_count * capacity /\ same_selection sel s /\
                    S.length comb == token_count /\
                    (forall (t:index token_count). S.index comb t == combine t))
          (ensures (forall (k:nat). k < SS.slots ==>
                      S.index probs k ==
                        (let t = SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens in
                         if t >= 0 then S.index comb t else zero)) <==>
                   (forall (e:index expert_count) (c:index capacity).
                      matrix_view zero probs expert_count capacity e c ==
                        kept_probability s combine e c))
  = introduce (forall (k:nat). k < SS.slots ==>
                 S.index probs k ==
                   (let t = SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens in
                    if t >= 0 then S.index comb t else zero)) ==>
              (forall (e:index expert_count) (c:index capacity).
                 matrix_view zero probs expert_count capacity e c ==
                   kept_probability s combine e c)
    with begin
      let at (e:index expert_count) (c:index capacity)
        : Lemma (matrix_view zero probs expert_count capacity e c ==
                   kept_probability s combine e c)
        = index_bound expert_count capacity e c;
          split_index capacity e c;
          slot_kept sel s e c token_count
      in
      Classical.forall_intro_2 at
    end;
    introduce (forall (e:index expert_count) (c:index capacity).
                 matrix_view zero probs expert_count capacity e c ==
                   kept_probability s combine e c) ==>
              (forall (k:nat). k < SS.slots ==>
                 S.index probs k ==
                   (let t = SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens in
                    if t >= 0 then S.index comb t else zero))
    with begin
      let at (k:nat)
        : Lemma (k < SS.slots ==>
                 S.index probs k ==
                   (let t = SS.slot_token sel (k / SS.capacity) (k % SS.capacity) SS.tokens in
                    if t >= 0 then S.index comb t else zero))
        = if k < SS.slots then begin
            div_index expert_count capacity k;
            slot_kept sel s (k / capacity) (k % capacity) token_count;
            assert (matrix_view zero probs expert_count capacity (k / capacity) (k % capacity) ==
                    kept_probability s combine (k / capacity) (k % capacity))
          end
      in
      Classical.forall_intro at
    end

let survival_iff (surv:S.seq I32.t) (sel:S.seq int) (s:vector (index expert_count) token_count)
  : Lemma (requires S.length surv == token_count * expert_count /\ same_selection sel s)
          (ensures SS.survival_rows sel SS.tokens surv <==>
                   (forall (t:index token_count) (e:index expert_count).
                      I32.v (matrix_view 0l surv token_count expert_count t e) ==
                        survival_of s t e))
  = let at (t:index token_count) (e:index expert_count)
      : Lemma (t * expert_count + e < S.length surv /\
               (if S.index sel t = e && SS.kept sel t then 1 else 0) == survival_of s t e)
      = index_bound token_count expert_count t e;
        kept_survives sel s t e
    in
    Classical.forall_intro_2 at

let routing_iff (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t) (surv:S.seq I32.t)
  (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) (run:routing)
  : Lemma (requires output_lengths demand idx probs surv /\ corresponds rows sel comb run)
          (ensures (SS.routing_state sel comb SS.tokens demand idx probs /\
                    SS.survival_rows sel SS.tokens surv) <==>
                   structural run (outputs_of demand idx probs surv))
  = demand_iff demand sel run.selected;
    token_indices_iff idx sel run.selected;
    kept_probabilities_iff probs sel comb run.selected run.combine_probability;
    survival_iff surv sel run.selected

(* Router probabilities. The flat relation quantifies over flat real
   sequences, the typed one over real matrices; each direction converts one
   into the other (reals_of_matrix, matrix_view) and uses reference_row. *)

let reals_of_matrix (#rows #cols:pos) (m:matrix real rows cols) : GTot (S.seq real) =
  S.init_ghost (rows * cols) (fun k -> div_index rows cols k; m (k / cols) (k % cols))

let reals_of_matrix_at (#rows #cols:pos) (m:matrix real rows cols) (i:index rows) (j:index cols)
  : Lemma (SS.ridx (reals_of_matrix m) (i * cols + j) == m i j)
  = index_bound rows cols i j;
    split_index cols i j

let reals_of_matrix_approximated (#rows #cols:pos) (x:S.seq BF.t) (m:matrix real rows cols)
  : Lemma (requires S.length x == rows * cols /\
                    matrix_approximates (matrix_view zero x rows cols) m)
          (ensures x %~ reals_of_matrix m)
  = let at (k:nat{k < S.length x}) : Lemma (S.index x k %~ S.index (reals_of_matrix m) k) =
      div_index rows cols k;
      assert (matrix_view zero x rows cols (k / cols) (k % cols) %~ m (k / cols) (k % cols))
    in
    Classical.forall_intro at;
    assert (seq_approximates x (reals_of_matrix m))

let matrix_view_approximated (#rows #cols:pos) (x:S.seq BF.t) (rx:S.seq real)
  : Lemma (requires S.length x == rows * cols /\ x %~ rx)
          (ensures matrix_approximates (matrix_view zero x rows cols) (matrix_view 0.0R rx rows cols))
  = assert (seq_approximates x rx);
    Classical.forall_intro_2 (index_bound rows cols)

let f32_approximates_at (s:S.seq F32.t) (r:S.seq real) (i:nat)
  : Lemma (requires s %~ r /\ i < S.length s) (ensures S.index s i %~ S.index r i)
  = assert (seq_approximates s r)

let reference_row_of_matrices (r:router_inputs real) (t:index token_count)
  : Lemma (let row = SS.ref_softmax (SS.row_of (SS.ref_scores (reals_of_matrix r.tokens)
                                                              (reals_of_matrix r.router_weights)) t) in
           S.length row == expert_count /\
           (forall (e:index expert_count). S.index row e == router_probabilities r t e))
  = Classical.forall_intro_2 (reals_of_matrix_at r.tokens);
    Classical.forall_intro_2 (reals_of_matrix_at r.router_weights);
    reference_row (reals_of_matrix r.tokens) (reals_of_matrix r.router_weights) r t

let row_forward (x w:S.seq BF.t) (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t)
  (run:routing) (r:router_inputs real) (t:index token_count)
  : Lemma (requires corresponds rows sel comb run /\ SS.softmax_row_ok x w rows t /\
                    x %~ reals_of_matrix r.tokens /\ w %~ reals_of_matrix r.router_weights)
          (ensures forall (e:index expert_count).
                     run.probability_rows t e %~ router_probabilities r t e)
  = let rx = reals_of_matrix r.tokens in
    let rw = reals_of_matrix r.router_weights in
    let sm = SS.ref_softmax (SS.row_of (SS.ref_scores rx rw) t) in
    reference_row_of_matrices r t;
    row_entries rows sel comb run t;
    assert (SS.row_of rows t %~ sm);
    let at (e:index expert_count)
      : Lemma (run.probability_rows t e %~ router_probabilities r t e)
      = f32_approximates_at (SS.row_of rows t) sm e
    in
    Classical.forall_intro at

let probabilities_forward (x w:S.seq BF.t) (rows:S.seq F32.t) (sel:S.seq int)
  (comb:S.seq F32.t) (run:routing)
  : Lemma (requires input_lengths x w /\ corresponds rows sel comb run /\
                    SS.softmax_rows_ok x w rows)
          (ensures router_probabilities_relation (inputs_of x w) run.probability_rows)
  = let each (r:router_inputs real)
      : Lemma (inputs_approximates (inputs_of x w) r ==>
               matrix_approximates run.probability_rows (router_probabilities r))
      = introduce inputs_approximates (inputs_of x w) r ==>
                  matrix_approximates run.probability_rows (router_probabilities r)
        with begin
          reals_of_matrix_approximated x r.tokens;
          reals_of_matrix_approximated w r.router_weights;
          Classical.forall_intro (Classical.move_requires (row_forward x w rows sel comb run r))
        end
    in
    Classical.forall_intro each

let probabilities_backward (x w:S.seq BF.t) (rows:S.seq F32.t) (sel:S.seq int)
  (comb:S.seq F32.t) (run:routing)
  : Lemma (requires input_lengths x w /\ corresponds rows sel comb run /\
                    router_probabilities_relation (inputs_of x w) run.probability_rows)
          (ensures SS.softmax_rows_ok x w rows)
  = let each (t:nat) (rx rw:S.seq real)
      : Lemma (t < SS.tokens ==> x %~ rx /\ w %~ rw ==>
               SS.row_of rows t %~ SS.ref_softmax (SS.row_of (SS.ref_scores rx rw) t))
      = if t < SS.tokens then
        introduce x %~ rx /\ w %~ rw ==>
                  SS.row_of rows t %~ SS.ref_softmax (SS.row_of (SS.ref_scores rx rw) t)
        with begin
          let r : router_inputs real = {
            tokens = matrix_view 0.0R rx token_count hidden_size;
            router_weights = matrix_view 0.0R rw hidden_size expert_count;
          } in
          matrix_view_approximated #token_count #hidden_size x rx;
          matrix_view_approximated #hidden_size #expert_count w rw;
          assert (inputs_approximates (inputs_of x w) r);
          reference_row rx rw r t;
          row_entries rows sel comb run t;
          assert (seq_approximates (SS.row_of rows t)
                    (SS.ref_softmax (SS.row_of (SS.ref_scores rx rw) t)))
        end
    in
    Classical.forall_intro_3 each

(* ------------------------------------------------------------------ *)
(* The flat existential body, and its equivalence with the typed body   *)

let flat_witness (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) : prop =
  S.length rows == SS.tokens * SS.experts /\
  S.length sel == SS.tokens /\ S.length comb == SS.tokens /\
  SS.softmax_rows_ok x w rows /\
  (forall (t:nat). t < SS.tokens ==> SS.allowed_choice (SS.row_of rows t) (S.index sel t)) /\
  (forall (t:nat). t < SS.tokens /\ SS.kept sel t ==>
     SS.selected_combine_ok (SS.row_of rows t) (S.index sel t) (S.index comb t)) /\
  SS.routing_state sel comb SS.tokens demand idx probs /\
  SS.survival_rows sel SS.tokens surv

let typed_witness (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) (run:routing) : prop =
  structural run (outputs_of demand idx probs surv) /\ numerical (inputs_of x w) run /\
  selection run

let witness_iff (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t) (run:routing)
  : Lemma (requires input_lengths x w /\ output_lengths demand idx probs surv /\
                    corresponds rows sel comb run)
          (ensures flat_witness x w demand idx probs surv rows sel comb <==>
                   typed_witness x w demand idx probs surv run)
  = selection_iff rows sel comb run;
    combine_iff rows sel comb run;
    routing_iff demand idx probs surv rows sel comb run;
    Classical.move_requires (probabilities_forward x w rows sel comb) run;
    Classical.move_requires (probabilities_backward x w rows sel comb) run

let witness_forward (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t)
  : Lemma (requires input_lengths x w /\ output_lengths demand idx probs surv /\
                    flat_witness x w demand idx probs surv rows sel comb)
          (ensures postcondition (inputs_of x w) (outputs_of demand idx probs surv))
  = let run = routing_of rows sel comb in
    assert (forall (t:index token_count). SS.allowed_choice (SS.row_of rows t) (S.index sel t));
    corresponds_routing_of rows sel comb;
    witness_iff x w demand idx probs surv rows sel comb run;
    assert (typed_witness x w demand idx probs surv run)

let witness_backward (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) (run:routing)
  : Lemma (requires input_lengths x w /\ output_lengths demand idx probs surv /\
                    typed_witness x w demand idx probs surv run)
          (ensures flat_witness x w demand idx probs surv (rows_of run) (sel_of run) (comb_of run))
  = corresponds_rows_of run;
    witness_iff x w demand idx probs surv (rows_of run) (sel_of run) (comb_of run) run

(* ------------------------------------------------------------------ *)
(* Main theorems                                                        *)

(* Flat implies typed. *)
let program_postcondition_implies (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t)
  (probs:S.seq F32.t) (surv:S.seq I32.t)
  : Lemma (requires input_lengths x w /\ SS.program_postcondition x w demand idx probs surv)
          (ensures output_lengths demand idx probs surv /\
                   postcondition (inputs_of x w) (outputs_of demand idx probs surv))
  = eliminate exists (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t).
      S.length rows == SS.tokens * SS.experts /\
      S.length sel == SS.tokens /\ S.length comb == SS.tokens /\
      SS.softmax_rows_ok x w rows /\
      (forall (t:nat). t < SS.tokens ==> SS.allowed_choice (SS.row_of rows t) (S.index sel t)) /\
      (forall (t:nat). t < SS.tokens /\ SS.kept sel t ==>
         SS.selected_combine_ok (SS.row_of rows t) (S.index sel t) (S.index comb t)) /\
      SS.routing_state sel comb SS.tokens demand idx probs /\
      SS.survival_rows sel SS.tokens surv
    with witness_forward x w demand idx probs surv rows sel comb

(* Typed implies flat. *)
let postcondition_implies_program (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t)
  (probs:S.seq F32.t) (surv:S.seq I32.t)
  : Lemma (requires input_lengths x w /\ output_lengths demand idx probs surv /\
                    postcondition (inputs_of x w) (outputs_of demand idx probs surv))
          (ensures SS.program_postcondition x w demand idx probs surv)
  = let each (run:routing)
      : Lemma (typed_witness x w demand idx probs surv run ==>
               flat_witness x w demand idx probs surv (rows_of run) (sel_of run) (comb_of run))
      = Classical.move_requires (witness_backward x w demand idx probs surv) run
    in
    Classical.forall_intro each;
    assert (exists (run:routing). typed_witness x w demand idx probs surv run);
    assert (exists (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t).
              flat_witness x w demand idx probs surv rows sel comb)

let program_postcondition_iff (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t)
  (probs:S.seq F32.t) (surv:S.seq I32.t)
  : Lemma (requires input_lengths x w)
          (ensures SS.program_postcondition x w demand idx probs surv <==>
                   (output_lengths demand idx probs surv /\
                    postcondition (inputs_of x w) (outputs_of demand idx probs surv)))
  = Classical.move_requires (program_postcondition_implies x w demand idx probs) surv;
    Classical.move_requires (postcondition_implies_program x w demand idx probs) surv
