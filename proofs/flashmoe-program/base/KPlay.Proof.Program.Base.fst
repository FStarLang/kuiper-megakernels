module KPlay.Proof.Program.Base

(* Composition theorem for the base program: the run's boundary values form a
   per-token trace (token 0 = the fixed argmax/capacity calls, token t >= 1 =
   loop iteration t - 1), and the one generic chain lemma
   KPlay.Proof.Base.Chain.trace_refines is instantiated once. *)

open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base
open KPlay.Proof.Base.Chain
module S = FStar.Seq
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

#set-options "--z3rlimit 20 --fuel 0 --ifuel 1"

let selv_of (r:run) (t:nat) : S.seq I32.t =
  if t = 0 then r.sample_0000_argmax_000_0 else r.loop_sample_0000_argmax_i_0 (t - 1)

let spv_of (r:run) (t:nat) : S.seq F32.t =
  if t = 0 then r.sample_0000_argmax_000_1 else r.loop_sample_0000_argmax_i_1 (t - 1)

let cnt_of (r:run) (t:nat) : S.seq I64.t =
  if t = 0 then r.sample_0000_projection_softmax_1
  else if t = 1 then r.sample_0000_capacity_000_0
  else r.loop_sample_0000_capacity_i_0 (t - 2)

let idx_of (r:run) (t:nat) : S.seq I64.t =
  if t = 0 then r.sample_0000_projection_softmax_2
  else if t = 1 then r.sample_0000_capacity_000_1
  else r.loop_sample_0000_capacity_i_1 (t - 2)

let prb_of (r:run) (t:nat) : S.seq F32.t =
  if t = 0 then r.sample_0000_projection_softmax_3
  else if t = 1 then r.sample_0000_capacity_000_2
  else r.loop_sample_0000_capacity_i_2 (t - 2)

let srv_of (r:run) (t:nat) : S.seq I32.t =
  if t = 0 then r.sample_0000_projection_softmax_4
  else if t = 1 then r.sample_0000_capacity_000_3
  else r.loop_sample_0000_capacity_i_3 (t - 2)

let tr_of (r:run) : trace = {
  rows = r.sample_0000_projection_softmax_0;
  selv = selv_of r;
  spv  = spv_of r;
  cnt  = cnt_of r;
  idx  = idx_of r;
  prb  = prb_of r;
  srv  = srv_of r;
}

let tr_fields (r:run) (t:nat)
  : Lemma ((tr_of r).rows == r.sample_0000_projection_softmax_0 /\
           (tr_of r).selv t == selv_of r t /\ (tr_of r).spv t == spv_of r t /\
           (tr_of r).cnt t == cnt_of r t /\ (tr_of r).idx t == idx_of r t /\
           (tr_of r).prb t == prb_of r t /\ (tr_of r).srv t == srv_of r t)
  = ()

let count_v_i32 (x:I32.t) : Lemma (count_v x == I32.v x) = ()

let view_row (s:S.seq F32.t) (t:nat)
  : Lemma (view s (8 * t) 8 == row_of s t)
  = ()

let input_consts () : Lemma (tokens * hidden == 131072 /\ hidden * experts == 4096) = ()

let projection_facts (r:run)
  : Lemma (requires input_lengths r /\ family_ok r)
          (ensures S.length r.sample_0000_projection_softmax_0 == tokens * experts /\
                   softmax_rows_ok r.input0 r.input1 r.sample_0000_projection_softmax_0 /\
                   initial_routing_state r.sample_0000_projection_softmax_1
                     r.sample_0000_projection_softmax_2 r.sample_0000_projection_softmax_3 /\
                   S.length r.sample_0000_projection_softmax_4 == 0)
  = input_consts ()

let fixed_ok (r:run) : prop =
  base_argmax_relation (view r.sample_0000_projection_softmax_0 0 8)
    r.sample_0000_argmax_000_0 r.sample_0000_argmax_000_1 /\
  base_capacity_routing_relation r.sample_0000_argmax_000_0 r.sample_0000_argmax_000_1
    r.sample_0000_projection_softmax_1 r.sample_0000_projection_softmax_2
    r.sample_0000_projection_softmax_3 r.sample_0000_projection_softmax_4 0
    r.sample_0000_capacity_000_0 r.sample_0000_capacity_000_1
    r.sample_0000_capacity_000_2 r.sample_0000_capacity_000_3

let fixed_calls (r:run)
  : Lemma (requires family_ok r) (ensures fixed_ok r)
  = ()

let first_fields (r:run)
  : Lemma ((tr_of r).selv 0 == r.sample_0000_argmax_000_0 /\
           (tr_of r).spv 0 == r.sample_0000_argmax_000_1 /\
           (tr_of r).cnt 0 == r.sample_0000_projection_softmax_1 /\
           (tr_of r).idx 0 == r.sample_0000_projection_softmax_2 /\
           (tr_of r).prb 0 == r.sample_0000_projection_softmax_3 /\
           (tr_of r).srv 0 == r.sample_0000_projection_softmax_4 /\
           (tr_of r).cnt 1 == r.sample_0000_capacity_000_0 /\
           (tr_of r).idx 1 == r.sample_0000_capacity_000_1 /\
           (tr_of r).prb 1 == r.sample_0000_capacity_000_2 /\
           (tr_of r).srv 1 == r.sample_0000_capacity_000_3 /\
           row_of (tr_of r).rows 0 == view r.sample_0000_projection_softmax_0 0 8)
  = tr_fields r 0;
    tr_fields r 1;
    view_row r.sample_0000_projection_softmax_0 0

let step_first' (r:run)
  : Lemma (requires fixed_ok r) (ensures step_ok (tr_of r) 0)
  = first_fields r

let step_first (r:run)
  : Lemma (requires family_ok r) (ensures step_ok (tr_of r) 0)
  = fixed_calls r;
    step_first' r

let loop_at (r:run) (i:nat{i < loop_count})
  : Lemma (requires family_ok r) (ensures loop_ok r i)
  = ()

let lin (t:nat{1 <= t}) : Lemma (8 + 8 * (t - 1) == 8 * t) = ()

let step_loop' (r:run) (t:nat{1 <= t /\ t < tokens})
  : Lemma (requires loop_ok r (t - 1)) (ensures step_ok (tr_of r) t)
  = tr_fields r t;
    tr_fields r (t + 1);
    lin t;
    view_row r.sample_0000_projection_softmax_0 t

let step_loop (r:run) (t:nat{1 <= t /\ t < tokens})
  : Lemma (requires family_ok r) (ensures step_ok (tr_of r) t)
  = loop_at r (t - 1);
    step_loop' r t

let step_any (r:run) (t:nat)
  : Lemma (requires family_ok r /\ t < tokens) (ensures step_ok (tr_of r) t)
  = if t = 0 then step_first r else step_loop r t

let trace_of_run (r:run)
  : Lemma (requires input_lengths r /\ family_ok r) (ensures trace_ok (tr_of r))
  = projection_facts r;
    tr_fields r 0;
    Classical.forall_intro (Classical.move_requires (step_any r))

let outputs (r:run)
  : Lemma (requires family_ok r)
          (ensures base_expert_demand_relation ((tr_of r).cnt tokens) (output0 r) /\
           output1 r == (tr_of r).idx tokens /\
           output2 r == (tr_of r).prb tokens /\ output3 r == (tr_of r).srv tokens)
  = tr_fields r tokens

(* The int64 prefix counts lie in [0, tokens], so the int32 demand conversion
   is exact and the demand carries the same routing state. *)
let demand_state (sel:S.seq int) (comb:S.seq F32.t)
  (prefix:S.seq I64.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  : Lemma (requires S.length sel == tokens /\ S.length comb == tokens /\
                    routing_state sel comb tokens prefix idx probs /\
                    base_expert_demand_relation prefix demand)
          (ensures routing_state sel comb tokens demand idx probs)
  = introduce forall (e:nat). e < experts ==>
        0 <= I64.v (S.index prefix e) /\ I64.v (S.index prefix e) <= tokens
    with introduce _ ==> _ with begin
      count_v_i64 (S.index prefix e);
      count_upto_le sel e tokens
    end;
    introduce forall (e:nat). e < experts ==>
        count_v (S.index demand e) == count_upto sel e tokens
    with introduce _ ==> _ with begin
      count_v_i64 (S.index prefix e);
      count_v_i32 (S.index demand e)
    end

let refines (r:run) =
  input_consts ();
  trace_of_run r;
  projection_facts r;
  let tr = tr_of r in
  trace_refines tr;
  outputs r;
  demand_state (sel_of tr) (comb_of tr) (tr.cnt tokens) (output0 r) (tr.idx tokens) (tr.prb tokens);
  tr_fields r 0;
  assert (trace_post tr);
  introduce exists (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t).
      S.length rows == tokens * experts /\
      S.length sel == tokens /\ S.length comb == tokens /\
      softmax_rows_ok r.input0 r.input1 rows /\
      (forall (t:nat). t < tokens ==> allowed_choice (row_of rows t) (S.index sel t)) /\
      (forall (t:nat). t < tokens /\ kept sel t ==>
         selected_combine_ok (row_of rows t) (S.index sel t) (S.index comb t)) /\
      routing_state sel comb tokens (output0 r) (output1 r) (output2 r) /\
      survival_rows sel tokens (output3 r)
  with tr.rows (sel_of tr) (comb_of tr)
  and ()
