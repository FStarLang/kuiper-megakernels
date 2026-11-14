module KPlay.Review
#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module F = Kuiper.Floating
module Cast = Kuiper.Float.Casts.Base
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec

(* BF16 versus FP32 is a representation difference, not a claim that upstream
   producers compute equal scores. Base selects the first maximum; optimized
   may select any maximum, including NaNs under the shared priority policy.
   The separated full permissions retain the original disjoint-byte-range
   caller obligation. Empty histories require no live allocation. *)
noextract type implementation = | Base | Optimized

inline_for_extraction noextract unfold
let score_type (i:implementation) : Type0 =
  match i with | Base -> bf16 | Optimized -> f32

inline_for_extraction noextract
let vocabulary : Shared.size = 248320
noextract type position_t = x:i64{0 <= I64.v x /\ I64.v x < 2048}
noextract type prompt_length_t = x:i64{1 <= I64.v x /\ I64.v x <= 2048}

inline_for_extraction noextract
let previous_count (position:position_t) (prompt:prompt_length_t) =
  Shared.retained_count (I64.v prompt) (I64.v position)
inline_for_extraction noextract
let output_count (position:position_t) (prompt:prompt_length_t) =
  Shared.retained_count (I64.v prompt) (I64.v position + 1)
inline_for_extraction noextract
let active_array (n:nat) (ptr:array i64) : prop =
  n > 0 ==> (A.length ptr == n /\ is_global_array ptr)
inline_for_extraction noextract
let history (ptr:array i64) (values:seq i64) : slprop =
  if S.length values = 0 then emp
  else on gpu_loc (A.pts_to ptr values) ** pure (is_global_array ptr)

inline_for_extraction noextract unfold
let widened (i:implementation) (xs:lseq (score_type i) vocabulary)
  : lseq f32 vocabulary =
  match i with
  | Base -> S.init vocabulary (fun j -> Cast.cast_bf16_to_f32 (S.index xs j))
  | Optimized -> xs

inline_for_extraction noextract
let priority_le (x y:f32) : GTot bool =
  if F.is_nan x then F.is_nan y else F.is_nan y || not (F.lt y x)
inline_for_extraction noextract
let positive_infinity : f32 = F.infinity
inline_for_extraction noextract
let score_class (x:f32) : GTot nat =
  if F.is_nan x then 3
  else if F.ieee_eq x (F.neg positive_infinity) then 0
  else if F.ieee_eq x positive_infinity then 2
  else 1

inline_for_extraction noextract
let represents_scores
  (i:implementation) (xs:lseq (score_type i) vocabulary)
  (logical:Shared.vec Shared.score vocabulary) : GTot prop =
  let scores = widened i xs in
  (forall (j:Shared.token vocabulary).
    Shared.score_rank (S.index logical j) == score_class (S.index scores j) /\
    (F.is_zero (S.index scores j) ==> S.index logical j == Shared.FiniteScore 0.0R)) /\
  (forall (j k:Shared.token vocabulary).
    Shared.score_le (S.index logical j) (S.index logical k) <==>
    priority_le (S.index scores j) (S.index scores k))

inline_for_extraction noextract unfold
let selection (i:implementation) (logical:Shared.vec Shared.score vocabulary)
  (winner:Shared.token vocabulary) : prop =
  match i with
  | Base -> Shared.first_maximum logical winner
  | Optimized -> Shared.maximum logical winner

inline_for_extraction noextract
let post (i:implementation) (position:position_t) (prompt:prompt_length_t)
  (logical:Shared.vec Shared.score vocabulary)
  (previous:lseq i64 (previous_count position prompt))
  (selected:lseq i64 1) (emitted:lseq i64 (output_count position prompt)) : prop =
  (exists (winner:Shared.token vocabulary).
    I64.v (S.index selected 0) == winner /\ selection i logical winner) /\
  emitted == (if I64.v position < I64.v prompt - 1 then previous
              else S.append previous selected)

val base_post_refines_optimized
  (position:position_t) (prompt:prompt_length_t)
  (logical:Shared.vec Shared.score vocabulary)
  (previous:lseq i64 (previous_count position prompt))
  (selected:lseq i64 1) (emitted:lseq i64 (output_count position prompt))
  : Lemma
      (requires post Base position prompt logical previous selected emitted)
      (ensures post Optimized position prompt logical previous selected emitted)

fn base
  (position:position_t) (prompt:prompt_length_t)
  (scores:larray bf16 vocabulary{is_global_array scores})
  (previous:array i64{active_array (previous_count position prompt) previous})
  (selected:larray i64 1{is_global_array selected})
  (emitted:array i64{active_array (output_count position prompt) emitted})
  (#xs:erased (lseq bf16 vocabulary))
  (#ps:erased (lseq i64 (previous_count position prompt)))
  (#logical:erased (Shared.vec Shared.score vocabulary))
  preserves cpu ** on gpu_loc (A.pts_to scores xs) ** history previous ps
  requires pure (represents_scores Base xs logical)
  requires exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e
  ensures exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e **
    pure (post Base position prompt logical ps s e)

fn optimized
  (position:position_t) (prompt:prompt_length_t)
  (scores:larray f32 vocabulary{is_global_array scores})
  (previous:array i64{active_array (previous_count position prompt) previous})
  (selected:larray i64 1{is_global_array selected})
  (emitted:array i64{active_array (output_count position prompt) emitted})
  (#xs:erased (lseq f32 vocabulary))
  (#ps:erased (lseq i64 (previous_count position prompt)))
  (#logical:erased (Shared.vec Shared.score vocabulary))
  preserves cpu ** on gpu_loc (A.pts_to scores xs) ** history previous ps
  requires pure (represents_scores Optimized xs logical)
  requires exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e
  ensures exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e **
    pure (post Optimized position prompt logical ps s e)

fn switcher
  (i:implementation) (position:position_t) (prompt:prompt_length_t)
  (scores:larray (score_type i) vocabulary{is_global_array scores})
  (previous:array i64{active_array (previous_count position prompt) previous})
  (selected:larray i64 1{is_global_array selected})
  (emitted:array i64{active_array (output_count position prompt) emitted})
  (#xs:erased (lseq (score_type i) vocabulary))
  (#ps:erased (lseq i64 (previous_count position prompt)))
  (#logical:erased (Shared.vec Shared.score vocabulary))
  preserves cpu ** on gpu_loc (A.pts_to scores xs) ** history previous ps
  requires pure (represents_scores i xs logical)
  requires exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e
  ensures exists* (s:lseq i64 1) (e:lseq i64 (output_count position prompt)).
    on gpu_loc (A.pts_to selected s) ** history emitted e **
    pure (post i position prompt logical ps s e /\
          post Optimized position prompt logical ps s e)
