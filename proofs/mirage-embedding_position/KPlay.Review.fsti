module KPlay.Review
#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module Shared = KPlay.SharedSpec

(* Same seven-token prefill task. Optimized retains an eighth hidden row and
   a 4096-position rotary table; the common observations use only prompt rows.
   Both return ownership synchronously, with no pending completion epoch. *)
noextract type implementation = | Base | Optimized

inline_for_extraction noextract unfold
let hidden_rows (i:implementation) : pos =
  match i with | Base -> Shared.prompt_length | Optimized -> 8

inline_for_extraction noextract unfold
let table_positions (i:implementation) : pos =
  match i with | Base -> Shared.prompt_length | Optimized -> 4096

inline_for_extraction noextract
let embedding_count = Shared.vocab_size * Shared.hidden_size
inline_for_extraction noextract
let hidden_count (i:implementation) = hidden_rows i * Shared.hidden_size
inline_for_extraction noextract
let table_count (i:implementation) = 2 * table_positions i * Shared.head_dim
inline_for_extraction noextract
let embedding_values = lseq bf16 embedding_count
inline_for_extraction noextract
let hidden_values (i:implementation) = lseq bf16 (hidden_count i)
inline_for_extraction noextract
let table_values (i:implementation) = lseq bf16 (table_count i)
inline_for_extraction noextract
let valid_tokens (ts:lseq I64.t Shared.prompt_length) : prop =
  forall (p:Shared.prompt_index).
    0 <= I64.v (S.index ts p) /\ I64.v (S.index ts p) < Shared.vocab_size
inline_for_extraction noextract
let token_values = ts:lseq I64.t Shared.prompt_length{valid_tokens ts}
inline_for_extraction noextract
let rotary_view = Shared.prompt_index -> Shared.rotary_index -> real

inline_for_extraction noextract
let logical_input (w:embedding_values) (t:token_values) : Shared.inputs bf16 = {
  embedding_weight = (fun v h -> S.index w (v * Shared.hidden_size + h));
  tokens = (fun p -> I64.v (S.index t p));
  decode_steps = 1
}

inline_for_extraction noextract
let logical_output (i:implementation) (h:hidden_values i)
  (cosines sines:rotary_view) : Shared.outputs bf16 = {
  hidden_rows = (fun p d -> S.index h (p * Shared.hidden_size + d));
  position_ids = (fun p -> p);
  cache_positions = (fun p -> p);
  causal_bias = Shared.full_attention_bias;
  rope_cos = cosines;
  rope_sin = sines;
  decoder_state = Shared.initial_decoder_state ()
}

inline_for_extraction noextract
let rotary_representation (i:implementation) (r:table_values i)
  (cosines sines:rotary_view) : prop =
  forall (p:Shared.prompt_index) (d:Shared.rotary_index).
    v_approximates (S.index r (p * Shared.head_dim + d)) (cosines p d) /\
    v_approximates
      (S.index r ((table_positions i + p) * Shared.head_dim + d)) (sines p d)

inline_for_extraction noextract
let post (i:implementation) (w:embedding_values) (t:token_values)
  (h:hidden_values i) (r:table_values i) : prop =
  exists (cosines sines:rotary_view).
    rotary_representation i r cosines sines /\
    Shared.postcondition (logical_input w t) (logical_output i h cosines sines)

fn base
  (weights:larray bf16 embedding_count{is_global_array weights})
  (tokens:larray I64.t Shared.prompt_length{is_global_array tokens})
  (hidden:larray bf16 (hidden_count Base){is_global_array hidden})
  (table:larray bf16 (table_count Base){is_global_array table})
  (#w:erased embedding_values)
  (#t:erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  preserves cpu ** on gpu_loc (weights |-> w ** Pulse.Lib.Array.pts_to tokens t)
  requires exists* (h:hidden_values Base) (r:table_values Base).
    on gpu_loc (hidden |-> h ** table |-> r)
  ensures exists* (h:hidden_values Base) (r:table_values Base).
    on gpu_loc (hidden |-> h ** table |-> r) ** pure (post Base w t h r)

fn optimized
  (weights:larray bf16 embedding_count{is_global_array weights})
  (tokens:larray I64.t Shared.prompt_length{is_global_array tokens})
  (hidden:larray bf16 (hidden_count Optimized){is_global_array hidden})
  (table:larray bf16 (table_count Optimized){is_global_array table})
  (#w:erased embedding_values)
  (#t:erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  preserves cpu ** on gpu_loc (weights |-> w ** Pulse.Lib.Array.pts_to tokens t)
  requires exists* (h:hidden_values Optimized) (r:table_values Optimized).
    on gpu_loc (hidden |-> h ** table |-> r)
  ensures exists* (h:hidden_values Optimized) (r:table_values Optimized).
    on gpu_loc (hidden |-> h ** table |-> r) ** pure (post Optimized w t h r)

(*
Signature relation: both take the same packed embedding weights and seven
valid prompt tokens. hidden_count is 7*1024 for Base versus 8*1024 for
Optimized; table_count is 2*7*128 versus 2*4096*128. The sine offset follows
table_positions, so the physical rotary layouts must not be interchanged.
Both preserve the input cells and return output ownership without a pending
epoch. post maps exactly the seven prompt rows to the same
Shared.postcondition, including logical position/cache IDs, mask, rotary
targets and initial decoder-state observations.

The seven embedding rows are exact BF16 lookups. Rotary cells are related to
their real targets by rotary_representation. The eighth optimized hidden row
and unused rotary positions are outside this common observation; no equality
or additional guarantee about them is obtained from post. Logical metadata
in logical_output is not a claim that separate physical position/mask/cache
buffers are returned.

Assessment: low logical significance, moderate allocation/indexing impact.
Tightness.unique fixes every stated shared observation for fixed logical
input. The native review is exact for embedding rows and relative for rotary
values, and deliberately not tight over the entire padded buffers. The
dispatcher allocates or trims nothing and does not reindex data: its two
branches simply call the wrappers with the role-indexed arrays.
*)
fn switcher
  (i:implementation)
  (weights:larray bf16 embedding_count{is_global_array weights})
  (tokens:larray I64.t Shared.prompt_length{is_global_array tokens})
  (hidden:larray bf16 (hidden_count i){is_global_array hidden})
  (table:larray bf16 (table_count i){is_global_array table})
  (#w:erased embedding_values)
  (#t:erased (lseq I64.t Shared.prompt_length){valid_tokens t})
  preserves cpu ** on gpu_loc (weights |-> w ** Pulse.Lib.Array.pts_to tokens t)
  requires exists* (h:hidden_values i) (r:table_values i).
    on gpu_loc (hidden |-> h ** table |-> r)
  ensures exists* (h:hidden_values i) (r:table_values i).
    on gpu_loc (hidden |-> h ** table |-> r) ** pure (post i w t h r)
{
  match i {
    Base -> { base weights tokens hidden table #w #t }
    Optimized -> { optimized weights tokens hidden table #w #t }
  }
}

