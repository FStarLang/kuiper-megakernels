module KPlay.Review
#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Approx = Kuiper.Approximates.Base
module Shared = KPlay.SharedSpec

(* Identical input cells and mathematical targets; the returned score cells
   are BF16 in base and FP32 in optimized. Approximation is not injective.
   Both preserve every input cell and append the exact, unnormalized hidden
   row only on emitting calls. Empty histories have no live allocation. *)
noextract type implementation = | Base | Optimized
inline_for_extraction noextract unfold
let score_type (i:implementation) : Type0 =
  match i with | Base -> bf16 | Optimized -> f32
inline_for_extraction noextract unfold
let represents_score (i:implementation) : (score_type i -> real -> GTot prop) =
  match i with
  | Base -> fun (value:bf16) target -> Approx.v_approximates value target
  | Optimized -> fun (value:f32) target -> Approx.v_approximates value target

inline_for_extraction noextract
let width = Shared.reference_hidden_size
inline_for_extraction noextract
let vocabulary = Shared.reference_vocab_size
inline_for_extraction noextract
let weight_length = vocabulary * width + width
noextract type position_t = p:i64{0 <= I64.v p /\ I64.v p < 2048}
noextract type prompt_length_t = p:i64{1 <= I64.v p /\ I64.v p <= 2048}

inline_for_extraction noextract
let previous_rows (position:position_t) (prompt:prompt_length_t) : nat =
  if I64.v position < I64.v prompt then 0 else I64.v position - I64.v prompt + 1
inline_for_extraction noextract
let output_rows (position:position_t) (prompt:prompt_length_t) : nat =
  if I64.v position < I64.v prompt - 1 then 0 else I64.v position - I64.v prompt + 2
inline_for_extraction noextract
let history_buffer (n:nat) =
  ptr:array bf16{n > 0 ==> A.length ptr == n /\ is_global_array ptr}
inline_for_extraction noextract
let history_owns (ptr:array bf16) (fraction:perm) (values:seq bf16) : slprop =
  if S.length values = 0 then emp else on gpu_loc (A.pts_to ptr #fraction values)

inline_for_extraction noextract
let targets (weights:lseq bf16 weight_length) (hidden:lseq bf16 width)
  : GTot (Shared.vector real vocabulary) =
  Shared.logical_logits #width #vocabulary Shared.reference_epsilon
    (fun j -> S.index hidden j)
    (fun j -> S.index weights (vocabulary * width + j))
    (fun token j -> S.index weights (token * width + j))

inline_for_extraction noextract
let post (i:implementation) (position:position_t) (prompt:prompt_length_t)
  (weights:lseq bf16 weight_length) (hidden:lseq bf16 width)
  (previous:lseq bf16 (previous_rows position prompt * width))
  (scores:lseq (score_type i) vocabulary)
  (history:lseq bf16 (output_rows position prompt * width)) : prop =
  (forall (token:Shared.index vocabulary).
    represents_score i (S.index scores token) (targets weights hidden token)) /\
  history == (if I64.v position < I64.v prompt - 1 then previous
              else S.append previous hidden)

fn base
  (position:position_t) (prompt:prompt_length_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (hidden:larray bf16 width{is_global_array hidden})
  (previous:history_buffer (previous_rows position prompt * width))
  (scores:larray bf16 vocabulary{is_global_array scores})
  (history:history_buffer (output_rows position prompt * width))
  (#w:erased (lseq bf16 weight_length)) (#h:erased (lseq bf16 width))
  (#p:erased (lseq bf16 (previous_rows position prompt * width)))
  (#pw #ph #pp:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    history_owns previous pp p
  requires exists* (s:lseq bf16 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e
  ensures exists* (s:lseq bf16 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e **
    pure (post Base position prompt w h p s e)

fn optimized
  (position:position_t) (prompt:prompt_length_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (hidden:larray bf16 width{is_global_array hidden})
  (previous:history_buffer (previous_rows position prompt * width))
  (scores:larray f32 vocabulary{is_global_array scores})
  (history:history_buffer (output_rows position prompt * width))
  (#w:erased (lseq bf16 weight_length)) (#h:erased (lseq bf16 width))
  (#p:erased (lseq bf16 (previous_rows position prompt * width)))
  (#pw #ph #pp:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    history_owns previous pp p
  requires exists* (s:lseq f32 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e
  ensures exists* (s:lseq f32 vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e **
    pure (post Optimized position prompt w h p s e)

fn switcher
  (i:implementation) (position:position_t) (prompt:prompt_length_t)
  (weights:larray bf16 weight_length{is_global_array weights})
  (hidden:larray bf16 width{is_global_array hidden})
  (previous:history_buffer (previous_rows position prompt * width))
  (scores:larray (score_type i) vocabulary{is_global_array scores})
  (history:history_buffer (output_rows position prompt * width))
  (#w:erased (lseq bf16 weight_length)) (#h:erased (lseq bf16 width))
  (#p:erased (lseq bf16 (previous_rows position prompt * width)))
  (#pw #ph #pp:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    history_owns previous pp p
  requires exists* (s:lseq (score_type i) vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e
  ensures exists* (s:lseq (score_type i) vocabulary)
    (e:lseq bf16 (output_rows position prompt * width)).
    on gpu_loc (A.pts_to scores s) ** history_owns history 1.0R e **
    pure (post i position prompt w h p s e)
