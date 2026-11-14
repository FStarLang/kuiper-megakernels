module KPlay.Review
#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional

(* The common task is exact BF16 row selection, not merely approximation.
   Base additionally produces rotary data and initializes decoder state.
   Optimized has no such outputs. Half-permissions allow the previous-token
   pointer to alias a prompt cell, including the initial-position alias. *)
noextract type implementation = | Base | Optimized
noextract type position_t = p:i64{0 <= I64.v p /\ I64.v p < 2048}
noextract type length_t = n:i64{0 < I64.v n /\ I64.v n <= 2048}
inline_for_extraction noextract
let weight_length = Shared.vocab_size * Shared.hidden_size
inline_for_extraction noextract
let state_length (p:position_t) : nat = if I64.v p = 0 then 4718592 else 0
inline_for_extraction noextract
let convolution_length (p:position_t) : nat = if I64.v p = 0 then 442368 else 0

noextract noeq type base_auxiliary (p:position_t) = {
  rotary: (r:larray bf16 64{is_global_array r});
  state: (s:larray f32 (state_length p){is_global_array s});
  convolution: (c:larray bf16 (convolution_length p){is_global_array c})
}
inline_for_extraction noextract unfold
let auxiliary (i:implementation) (p:position_t) : Type0 =
  match i with | Base -> base_auxiliary p | Optimized -> unit

inline_for_extraction noextract
let auxiliary_before (i:implementation) (p:position_t) : (auxiliary i p -> slprop) =
  match i with
  | Base -> fun (a:base_auxiliary p) ->
      exists* (r:lseq bf16 64) (s:lseq f32 (state_length p))
        (c:lseq bf16 (convolution_length p)).
      on gpu_loc (A.pts_to a.rotary r ** A.pts_to a.state s ** A.pts_to a.convolution c)
  | Optimized -> fun _ -> emp

inline_for_extraction noextract
let auxiliary_after (i:implementation) (p:position_t) : (auxiliary i p -> slprop) =
  match i with
  | Base -> fun (a:base_auxiliary p) ->
      exists* (r:lseq bf16 64) (s:lseq f32 (state_length p))
        (c:lseq bf16 (convolution_length p)).
      on gpu_loc (A.pts_to a.rotary r ** A.pts_to a.state s ** A.pts_to a.convolution c) **
      pure (
        (forall (j:nat{j < 64}). S.index r j == B.rotary_component p j) /\
        (forall (j:nat{j < state_length p}). S.index s j == zero #f32) /\
        (forall (j:nat{j < convolution_length p}). S.index c j == zero #bf16))
  | Optimized -> fun _ -> emp

inline_for_extraction noextract
let logical_row (w:lseq bf16 weight_length) (t:Shared.token_id)
  : Shared.hidden_vector bf16 =
  S.slice w (t * Shared.hidden_size) ((t + 1) * Shared.hidden_size)
inline_for_extraction noextract
let logical_weights (w:lseq bf16 weight_length) : Shared.embedding_weights bf16 =
  S.init Shared.vocab_size (logical_row w)

inline_for_extraction noextract
let inputs (n:length_t) (position:position_t)
  (q:lseq i64 (I64.v n)) (previous:lseq i64 1) (t:Shared.token_id) : prop =
  (forall (j:nat{j < I64.v n}).
    0 <= I64.v (S.index q j) /\ I64.v (S.index q j) < Shared.vocab_size) /\
  (if I64.v position < I64.v n then I64.v (S.index q (I64.v position))
   else I64.v (S.index previous 0)) == t

inline_for_extraction noextract
let post (w:lseq bf16 weight_length) (t:Shared.token_id)
  (h:Shared.hidden_vector bf16) : prop =
  Shared.embedding_post (logical_weights w) t h

fn base
  (n:length_t) (position:position_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (prompt:larray i64 (I64.v n){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 Shared.hidden_size{is_global_array hidden})
  (extra:auxiliary Base position)
  (#w:erased (lseq bf16 weight_length)) (#q:erased (lseq i64 (I64.v n)))
  (#p:erased (lseq i64 1)) (#t:erased Shared.token_id)
  preserves cpu ** on gpu_loc (
    A.pts_to weights #0.5R w ** A.pts_to prompt #0.5R q ** A.pts_to previous #0.5R p)
  requires pure (inputs n position q p t) ** auxiliary_before Base position extra
  requires exists* (h:Shared.hidden_vector bf16). on gpu_loc (A.pts_to hidden h)
  ensures exists* (h:Shared.hidden_vector bf16).
    on gpu_loc (A.pts_to hidden h) ** auxiliary_after Base position extra ** pure (post w t h)

fn optimized
  (n:length_t) (position:position_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (prompt:larray i64 (I64.v n){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 Shared.hidden_size{is_global_array hidden})
  (extra:auxiliary Optimized position)
  (#w:erased (lseq bf16 weight_length)) (#q:erased (lseq i64 (I64.v n)))
  (#p:erased (lseq i64 1)) (#t:erased Shared.token_id)
  preserves cpu ** on gpu_loc (
    A.pts_to weights #0.5R w ** A.pts_to prompt #0.5R q ** A.pts_to previous #0.5R p)
  requires pure (inputs n position q p t) ** auxiliary_before Optimized position extra
  requires exists* (h:Shared.hidden_vector bf16). on gpu_loc (A.pts_to hidden h)
  ensures exists* (h:Shared.hidden_vector bf16).
    on gpu_loc (A.pts_to hidden h) ** auxiliary_after Optimized position extra ** pure (post w t h)

fn switcher
  (i:implementation) (n:length_t) (position:position_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (prompt:larray i64 (I64.v n){is_global_array prompt})
  (previous:larray i64 1{is_global_array previous})
  (hidden:larray bf16 Shared.hidden_size{is_global_array hidden})
  (extra:auxiliary i position)
  (#w:erased (lseq bf16 weight_length)) (#q:erased (lseq i64 (I64.v n)))
  (#p:erased (lseq i64 1)) (#t:erased Shared.token_id)
  preserves cpu ** on gpu_loc (
    A.pts_to weights #0.5R w ** A.pts_to prompt #0.5R q ** A.pts_to previous #0.5R p)
  requires pure (inputs n position q p t) ** auxiliary_before i position extra
  requires exists* (h:Shared.hidden_vector bf16). on gpu_loc (A.pts_to hidden h)
  ensures exists* (h:Shared.hidden_vector bf16).
    on gpu_loc (A.pts_to hidden h) ** auxiliary_after i position extra ** pure (post w t h)
