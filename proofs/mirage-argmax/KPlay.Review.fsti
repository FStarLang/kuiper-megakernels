module KPlay.Review
#lang-pulse

open Kuiper
open FStar.Real
module S = FStar.Seq.Base
module I64 = FStar.Int64
module F = Kuiper.Floating
module Cast = Kuiper.Float.Casts.Base
module Shared = KPlay.SharedSpec

(* Only the physical row stride differs. Both return the same allowed-maximum
   relation on the same logical prompt/logits. Finite comparison-faithful
   inputs are explicit; first-index versus permitted native tie choices are
   not promoted to an equality of results. Ownership is returned synchronously. *)
noextract type implementation = | Base | Optimized

inline_for_extraction noextract unfold
let stride (i:implementation) : pos =
  match i with | Base -> Shared.vocab_size | Optimized -> 153600
inline_for_extraction noextract
let logits_size (i:implementation) = Shared.prompt_length * stride i
inline_for_extraction noextract
let physical_logits (i:implementation) = lseq bf16 (logits_size i)
inline_for_extraction noextract
let physical_prompt = lseq I64.t Shared.prompt_length
inline_for_extraction noextract
let physical_output = lseq I64.t Shared.sequence_length
noextract type row_index = r:nat{r < Shared.prompt_length}
noextract type token_index = j:nat{j < Shared.vocab_size}

inline_for_extraction noextract
let row_scores (i:implementation) (logits:physical_logits i) (row:row_index)
  : Tot (lseq f32 Shared.vocab_size) =
  S.init Shared.vocab_size
    (fun j -> Cast.cast_bf16_to_f32 (S.index logits (row * stride i + j)))

inline_for_extraction noextract
let represents_inputs
  (i:implementation) (logits:physical_logits i) (prefix:physical_prompt) (row:row_index)
  (prompt:Shared.prompt_tokens) (logical:Shared.prompt_logits) : GTot prop =
  let scores = row_scores i logits row in
  let values = Shared.final_logits logical in
  (forall (j:row_index). I64.v (S.index prefix j) == S.index prompt j) /\
  (forall (j:token_index). F.is_finite (S.index scores j)) /\
  ((forall (j:token_index). not (F.is_nan (S.index scores j))) /\
   (forall (j k:token_index).
     (lt (S.index scores j) (S.index scores k) <==> S.index values j <. S.index values k) /\
     (F.ieee_eq (S.index scores j) (S.index scores k) <==> S.index values j == S.index values k)))

inline_for_extraction noextract
let represents_output (output:physical_output) (logical:Shared.result) : prop =
  I64.v (S.index output Shared.prompt_length) == logical._1 /\
  (forall (j:nat{j < Shared.sequence_length}).
    I64.v (S.index output j) == S.index logical._2 j)

noextract
let post (prompt:Shared.prompt_tokens) (logical:Shared.prompt_logits)
  (output:physical_output) : GTot prop =
  exists (result:Shared.result).
    represents_output output result /\ Shared.postcondition prompt logical 1 result

fn base
  (logits:larray bf16 (logits_size Base){is_global_array logits})
  (prefix:larray I64.t Shared.prompt_length{is_global_array prefix})
  (row:I64.t{0 <= I64.v row /\ I64.v row < Shared.prompt_length})
  (output:larray I64.t Shared.sequence_length{is_global_array output})
  (#xs:erased (physical_logits Base)) (#ps:erased physical_prompt)
  (#prompt:erased Shared.prompt_tokens) (#logical:erased Shared.prompt_logits)
  preserves cpu ** on gpu_loc (logits |-> xs) ** on gpu_loc (prefix |-> ps)
  requires pure (represents_inputs Base xs ps (I64.v row) prompt logical)
  requires exists* old. on gpu_loc (output |-> old)
  ensures exists* (result:physical_output).
    on gpu_loc (output |-> result) ** pure (post prompt logical result)

fn optimized
  (logits:larray bf16 (logits_size Optimized){is_global_array logits})
  (prefix:larray I64.t Shared.prompt_length{is_global_array prefix})
  (row:I64.t{0 <= I64.v row /\ I64.v row < Shared.prompt_length})
  (output:larray I64.t Shared.sequence_length{is_global_array output})
  (#xs:erased (physical_logits Optimized)) (#ps:erased physical_prompt)
  (#prompt:erased Shared.prompt_tokens) (#logical:erased Shared.prompt_logits)
  preserves cpu ** on gpu_loc (logits |-> xs) ** on gpu_loc (prefix |-> ps)
  requires pure (represents_inputs Optimized xs ps (I64.v row) prompt logical)
  requires exists* old. on gpu_loc (output |-> old)
  ensures exists* (result:physical_output).
    on gpu_loc (output |-> result) ** pure (post prompt logical result)

fn switcher
  (i:implementation)
  (logits:larray bf16 (logits_size i){is_global_array logits})
  (prefix:larray I64.t Shared.prompt_length{is_global_array prefix})
  (row:I64.t{0 <= I64.v row /\ I64.v row < Shared.prompt_length})
  (output:larray I64.t Shared.sequence_length{is_global_array output})
  (#xs:erased (physical_logits i)) (#ps:erased physical_prompt)
  (#prompt:erased Shared.prompt_tokens) (#logical:erased Shared.prompt_logits)
  preserves cpu ** on gpu_loc (logits |-> xs) ** on gpu_loc (prefix |-> ps)
  requires pure (represents_inputs i xs ps (I64.v row) prompt logical)
  requires exists* old. on gpu_loc (output |-> old)
  ensures exists* (result:physical_output).
    on gpu_loc (output |-> result) ** pure (post prompt logical result)
