module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I32 = FStar.Int32
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

(* Base advances one already-represented routing prefix; optimized builds the
   complete route. Counts are I64 versus I32. Base emits eight I32 demand cells
   only at token 255 (even in Full); its final Full demand is a view of computed
   I64 counts, not a returned 16-cell I32 buffer. Both use the same native
   selection boundary and complete shared prefix/route specification. *)
noextract type implementation = | Base | Optimized
inline_for_extraction noextract unfold
let count_type (i:implementation) : Type0 = match i with | Base -> i64 | Optimized -> i32
inline_for_extraction noextract
let count_value (i:implementation) : (count_type i -> int) =
  match i with | Base -> I64.v | Optimized -> I32.v
inline_for_extraction noextract unfold
let call_index (i:implementation) (p:Shared.model_profile) : Type0 =
  match i with
  | Base -> t:i64{0 <= I64.v t /\ I64.v t < Shared.token_count p}
  | Optimized -> unit
inline_for_extraction noextract
let completed (i:implementation) (p:Shared.model_profile)
  : (call_index i p -> Shared.prefix_length p) =
  match i with | Base -> fun t -> I64.v t + 1 | Optimized -> fun _ -> Shared.token_count p
inline_for_extraction noextract
let input_rows (i:implementation) (p:Shared.model_profile) : pos =
  match i with | Base -> 1 | Optimized -> Shared.token_count p
inline_for_extraction noextract
let slots (p:Shared.model_profile) : nat = Shared.expert_count p * Shared.capacity p
inline_for_extraction noextract
let demand_count (token:nat) : nat = if token = 255 then 8 else 0

noextract noeq type physical_state (i:implementation) (p:Shared.model_profile)
  (rows:Shared.prefix_length p) = {
  counts: lseq (count_type i) (Shared.expert_count p);
  indices: lseq i64 (slots p);
  weights: lseq f32 (slots p);
  flags: lseq i32 (rows * Shared.expert_count p)
}
inline_for_extraction noextract
let represented (i:implementation) (p:Shared.model_profile) (rows:Shared.prefix_length p)
  (v:physical_state i p rows) (s:Shared.routing_state p) : prop =
  (forall (e:Shared.expert_index p). count_value i (Seq.index v.counts e) == s.counts e) /\
  (forall (e:Shared.expert_index p) (j:Shared.slot_index p).
    I64.v (Seq.index v.indices (e * Shared.capacity p + j)) == s.owners e j /\
    v_approximates (Seq.index v.weights (e * Shared.capacity p + j)) (s.weights e j) /\
    (s.owners e j == -1 ==> Seq.index v.weights (e * Shared.capacity p + j) == zero #f32)) /\
  (forall (t:Shared.token_index p) (e:Shared.expert_index p).
    if t < rows then I32.v (Seq.index v.flags (t * Shared.expert_count p + e)) == s.survival_cells t e
    else s.survival_cells t e == 0)
inline_for_extraction noextract
let output_represents (i:implementation) (p:Shared.model_profile) (rows:Shared.prefix_length p)
  (v:physical_state i p rows) (out:Shared.public_outputs p) : prop =
  (forall (e:Shared.expert_index p).
    I32.v (out.expert_demand e) == count_value i (Seq.index v.counts e)) /\
  (forall (e:Shared.expert_index p) (j:Shared.slot_index p).
    out.token_indices e j == Seq.index v.indices (e * Shared.capacity p + j) /\
    out.probabilities e j == Seq.index v.weights (e * Shared.capacity p + j) /\
    (I64.v (out.token_indices e j) == -1 ==> out.probabilities e j == zero #f32)) /\
  (forall (t:Shared.token_index p) (e:Shared.expert_index p).
    if t < rows then out.survival t e == Seq.index v.flags (t * Shared.expert_count p + e)
    else I32.v (out.survival t e) == 0)
inline_for_extraction noextract
let post (i:implementation) (p:Shared.model_profile) (b:Shared.selected_boundary p)
  (step:call_index i p) (v:physical_state i p (completed i p step)) : prop =
  represented i p (completed i p step) v (Shared.prefix_state b (completed i p step)) /\
  (completed i p step == Shared.token_count p ==>
    (exists (out:Shared.public_outputs p).
      output_represents i p (completed i p step) v out /\ Shared.route_post b out))

inline_for_extraction noextract unfold
let device_array (a:Type0) (n:nat) = ar:larray a n{is_global_array ar}
noextract noeq type buffers (i:implementation) (p:Shared.model_profile)
  (rows:Shared.prefix_length p) = {
  counts_buffer: device_array (count_type i) (Shared.expert_count p);
  indices_buffer: device_array i64 (slots p);
  weights_buffer: device_array f32 (slots p);
  flags_buffer: device_array i32 (rows * Shared.expert_count p)
}
inline_for_extraction noextract
let owns (i:implementation) (p:Shared.model_profile) (rows:Shared.prefix_length p)
  (ptr:buffers i p rows) (v:physical_state i p rows) : slprop =
  on gpu_loc (
    A.pts_to ptr.counts_buffer v.counts ** A.pts_to ptr.indices_buffer v.indices **
    A.pts_to ptr.weights_buffer v.weights ** A.pts_to ptr.flags_buffer v.flags)
noextract noeq type base_buffers (p:Shared.model_profile) (step:call_index Base p) = {
  previous_buffers: buffers Base p (completed Base p step);
  previous: erased (physical_state Base p (completed Base p step));
  emitted_demand: device_array i32 (demand_count (I64.v step))
}
noextract noeq type optimized_buffers (p:Shared.model_profile) = {
  best_buffer: device_array f32 (Shared.token_count p);
  best: erased (lseq f32 (Shared.token_count p))
}
inline_for_extraction noextract unfold
let auxiliary (i:implementation) (p:Shared.model_profile) : (call_index i p -> Type0) =
  match i with | Base -> fun t -> base_buffers p t | Optimized -> fun _ -> optimized_buffers p
inline_for_extraction noextract
let before (i:implementation) (p:Shared.model_profile)
  : (step:call_index i p -> auxiliary i p step -> slprop) =
  match i with
  | Base -> fun (step:call_index Base p) (aux:base_buffers p step) ->
    owns Base p (completed Base p step) aux.previous_buffers aux.previous **
    (exists* (d:lseq i32 (demand_count (I64.v step))). on gpu_loc (A.pts_to aux.emitted_demand d))
  | Optimized -> fun (_:unit) (aux:optimized_buffers p) -> on gpu_loc (A.pts_to aux.best_buffer aux.best)
inline_for_extraction noextract
let after (i:implementation) (p:Shared.model_profile)
  : (step:call_index i p -> auxiliary i p step -> physical_state i p (completed i p step) -> slprop) =
  match i with
  | Base -> fun (step:call_index Base p) (aux:base_buffers p step)
    (v:physical_state Base p (completed Base p step)) ->
    owns Base p (completed Base p step) aux.previous_buffers aux.previous **
    (exists* (d:lseq i32 (demand_count (I64.v step))).
      on gpu_loc (A.pts_to aux.emitted_demand d) **
      pure (I64.v step == 255 ==> (forall (e:nat{e < 8}).
        I32.v (Seq.index d e) == I64.v (Seq.index v.counts e))))
  | Optimized -> fun (_:unit) (aux:optimized_buffers p)
    (_:physical_state Optimized p (Shared.token_count p)) -> on gpu_loc (A.pts_to aux.best_buffer aux.best)
inline_for_extraction noextract
let input_represents (i:implementation) (p:Shared.model_profile) (b:Shared.selected_boundary p)
  : (step:call_index i p -> auxiliary i p step ->
     lseq i32 (input_rows i p) -> lseq f32 (input_rows i p * Shared.expert_count p) -> prop) =
  match i with
  | Base -> fun (step:call_index Base p) (aux:base_buffers p step) selected probabilities ->
    I32.v (Seq.index selected 0) == b.selected_expert (I64.v step) /\
    (forall (e:Shared.expert_index p). Seq.index probabilities e == b.probability_rows (I64.v step) e) /\
    represented Base p (completed Base p step) aux.previous (Shared.prefix_state b (I64.v step))
  | Optimized -> fun (_:unit) (aux:optimized_buffers p) selected probabilities ->
    (forall (t:Shared.token_index p). I32.v (Seq.index selected t) == b.selected_expert t) /\
    (forall (t:Shared.token_index p) (e:Shared.expert_index p).
      Seq.index probabilities (t * Shared.expert_count p + e) == b.probability_rows t e) /\
    (forall (t:Shared.token_index p). Seq.index aux.best t == b.selected_probability t)

val completed_refines_shared :
  i:implementation -> p:Shared.model_profile -> b:Shared.selected_boundary p ->
  v:physical_state i p (Shared.token_count p) -> out:Shared.public_outputs p ->
  Lemma
    (requires represented i p (Shared.token_count p) v (Shared.route_result b) /\
      output_represents i p (Shared.token_count p) v out)
    (ensures Shared.route_post b out)

fn base
  (p:Shared.model_profile) (#b:erased (Shared.selected_boundary p))
  (step:call_index Base p)
  (selected:device_array i32 (input_rows Base p))
  (probabilities:device_array f32 (input_rows Base p * Shared.expert_count p))
  (output:buffers Base p (completed Base p step)) (aux:auxiliary Base p step)
  (#s:erased (lseq i32 (input_rows Base p)))
  (#r:erased (lseq f32 (input_rows Base p * Shared.expert_count p)))
  preserves cpu ** on gpu_loc (A.pts_to selected s ** A.pts_to probabilities r)
  requires pure (input_represents Base p b step aux s r) ** before Base p step aux **
    (exists* (old:physical_state Base p (completed Base p step)). owns Base p (completed Base p step) output old)
  ensures exists* (v:physical_state Base p (completed Base p step)).
    owns Base p (completed Base p step) output v ** after Base p step aux v ** pure (post Base p b step v)

fn optimized
  (p:Shared.model_profile) (#b:erased (Shared.selected_boundary p))
  (step:call_index Optimized p)
  (selected:device_array i32 (input_rows Optimized p))
  (probabilities:device_array f32 (input_rows Optimized p * Shared.expert_count p))
  (output:buffers Optimized p (completed Optimized p step)) (aux:auxiliary Optimized p step)
  (#s:erased (lseq i32 (input_rows Optimized p)))
  (#r:erased (lseq f32 (input_rows Optimized p * Shared.expert_count p)))
  preserves cpu ** on gpu_loc (A.pts_to selected s ** A.pts_to probabilities r)
  requires pure (input_represents Optimized p b step aux s r) ** before Optimized p step aux **
    (exists* (old:physical_state Optimized p (completed Optimized p step)). owns Optimized p (completed Optimized p step) output old)
  ensures exists* (v:physical_state Optimized p (completed Optimized p step)).
    owns Optimized p (completed Optimized p step) output v ** after Optimized p step aux v ** pure (post Optimized p b step v)

fn switcher
  (i:implementation) (p:Shared.model_profile) (#b:erased (Shared.selected_boundary p))
  (step:call_index i p)
  (selected:device_array i32 (input_rows i p))
  (probabilities:device_array f32 (input_rows i p * Shared.expert_count p))
  (output:buffers i p (completed i p step)) (aux:auxiliary i p step)
  (#s:erased (lseq i32 (input_rows i p)))
  (#r:erased (lseq f32 (input_rows i p * Shared.expert_count p)))
  preserves cpu ** on gpu_loc (A.pts_to selected s ** A.pts_to probabilities r)
  requires pure (input_represents i p b step aux s r) ** before i p step aux **
    (exists* (old:physical_state i p (completed i p step)). owns i p (completed i p step) output old)
  ensures exists* (v:physical_state i p (completed i p step)).
    owns i p (completed i p step) output v ** after i p step aux v ** pure (post i p b step v)

#pop-options
