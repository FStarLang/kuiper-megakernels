module KPlay.Review
#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let establish_post (i:implementation) (w:embedding_values) (t:token_values)
  (h:hidden_values i) (r:table_values i)
  : Lemma
      (requires
        (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
          S.index h (p * Shared.hidden_size + d) ==
          S.index w (I64.v (S.index t p) * Shared.hidden_size + d)) /\
        rotary_representation i r Shared.cosine Shared.sine)
      (ensures post i w t h r)
  =
  assert (Shared.state_postcondition (Shared.initial_decoder_state #bf16 ()));
  assert (Shared.postcondition (logical_input w t)
    (logical_output i h Shared.cosine Shared.sine))

inline_for_extraction noextract
let base_post_refines (w:embedding_values) (t:token_values)
  (h:hidden_values Base) (r:table_values Base)
  : Lemma
      (requires B.semantic_postcondition w t 0L h r)
      (ensures post Base w t h r)
  =
  B.semantic_postcondition_implies_pointwise w t 0L h r;
  let hidden_at (p:Shared.prompt_index) (d:Shared.hidden_index)
    : Lemma (S.index h (p * Shared.hidden_size + d) ==
        S.index w (I64.v (S.index t p) * Shared.hidden_size + d))
    =
    assert (S.index h (B.hidden_offset p d) ==
      S.index w (B.embedding_offset (B.token_at t p) d));
    assert_norm (B.hidden_offset p d == p * Shared.hidden_size + d);
    assert_norm (B.embedding_offset (B.token_at t p) d ==
      I64.v (S.index t p) * Shared.hidden_size + d) in
  FStar.Classical.forall_intro_2 hidden_at;
  establish_post Base w t h r

inline_for_extraction noextract
let optimized_post_refines (w:embedding_values) (t:token_values)
  (h:hidden_values Optimized) (r:table_values Optimized)
  : Lemma
      (requires O.semantic_postcondition w t h r)
      (ensures post Optimized w t h r)
  =
  assert (O.physical_postcondition w t h r);
  let hidden_at (p:Shared.prompt_index) (d:Shared.hidden_index)
    : Lemma (S.index h (p * Shared.hidden_size + d) ==
        S.index w (I64.v (S.index t p) * Shared.hidden_size + d))
    =
    assert (S.index h (O.hidden_offset p d) ==
      S.index w (O.embedding_offset (O.physical_token t p) d));
    assert_norm (O.hidden_offset p d == p * Shared.hidden_size + d);
    assert (O.physical_token t p == I64.v (S.index t p)) in
  FStar.Classical.forall_intro_2 hidden_at;
  let rotary_at (p:Shared.prompt_index) (d:Shared.rotary_index)
    : Lemma (
        v_approximates (S.index r (p * Shared.head_dim + d)) (Shared.cosine p d) /\
        v_approximates
          (S.index r ((table_positions Optimized + p) * Shared.head_dim + d))
          (Shared.sine p d))
    = O.table_restriction p d in
  FStar.Classical.forall_intro_2 rotary_at;
  establish_post Optimized w t h r

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
{
  B.kplay_run weights tokens 0L hidden table #w #t;
  with next_hidden next_table.
    assert on gpu_loc (hidden |-> next_hidden ** table |-> next_table);
  let _ = base_post_refines w t next_hidden next_table;
  assert pure (post Base w t next_hidden next_table);
}

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
{
  O.kplay_run weights tokens hidden table #1.0R #1.0R #w #t;
  with next_hidden next_table.
    assert on gpu_loc (hidden |-> next_hidden ** table |-> next_table);
  let _ = optimized_post_refines w t next_hidden next_table;
  assert pure (post Optimized w t next_hidden next_table);
}

#pop-options
