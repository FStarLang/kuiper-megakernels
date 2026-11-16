module KPlay.Review
#lang-pulse

open Kuiper
module I64 = FStar.Int64
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let base_post_refines
  (xs:physical_logits Base) (ps:physical_prompt) (row:row_index)
  (prompt:Shared.prompt_tokens) (logical:Shared.prompt_logits) (result:physical_output)
  : Lemma
      (requires represents_inputs Base xs ps row prompt logical /\
        B.postcondition xs ps row result)
      (ensures post prompt logical result)
  =
  assert_norm (represents_inputs Base xs ps row prompt logical ==
    B.represents_inputs xs ps row prompt logical);
  assert_norm ((fun r -> represents_output result r) ==
    (fun r -> B.represents_output result r));
  assert (B.shared_refinement xs ps row result)

inline_for_extraction noextract
let optimized_post_refines
  (xs:physical_logits Optimized) (ps:physical_prompt) (row:row_index)
  (prompt:Shared.prompt_tokens) (logical:Shared.prompt_logits) (result:physical_output)
  : Lemma
      (requires represents_inputs Optimized xs ps row prompt logical /\
        O.postcondition xs ps row result)
      (ensures post prompt logical result)
  =
  assert_norm (represents_inputs Optimized xs ps row prompt logical ==
    O.represents_inputs xs ps row prompt logical);
  assert_norm ((fun r -> represents_output result r) ==
    (fun r -> O.represents_output result r));
  assert (O.shared_refinement xs ps row result)

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
{
  B.kplay_run logits prefix row output #xs #ps;
  with observed. assert on gpu_loc (output |-> observed);
  let _ = base_post_refines xs ps (I64.v row) prompt logical observed;
  assert pure (post prompt logical observed);
}

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
{
  O.kplay_run logits prefix row output #xs #ps;
  with observed. assert on gpu_loc (output |-> observed);
  let _ = optimized_post_refines xs ps (I64.v row) prompt logical observed;
  assert pure (post prompt logical observed);
}

#pop-options
