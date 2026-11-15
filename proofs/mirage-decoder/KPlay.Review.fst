module KPlay.Review
#lang-pulse

open Kuiper
open FStar.Real
module Seq = FStar.Seq
module A = Pulse.Lib.Array
module I64 = FStar.Int64
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let canonical_offsets : o:B.weight_offsets{B.valid_weight_offsets weight_count o} = {
  B.pre_norm = 0; B.qkv = 2048; B.qk_norm = 6293504;
  B.attention_output = 4196352; B.post_norm = 1024; B.gate_up = 6293760; B.down = 12585216
}
inline_for_extraction noextract
let base_logical (input:Shared.prefill_input) (b:base_inputs)
  : d:B.input_denotation{B.valid_denotation d} = {
  B.weights = input.weight; B.incoming_residual = b.residual_model;
  B.incoming_delta = b.delta_model; B.attention = input.data
}
inline_for_extraction noextract
let optimized_logical (input:Shared.prefill_input) (b:optimized_inputs)
  : d:O.input_denotation{O.valid_denotation d} = {
  O.weights = input.weight; O.hidden = FStar.Ghost.reveal b.hidden_model; O.attention = input.data
}
inline_for_extraction noextract
let base_input_agrees (input:Shared.prefill_input) (b:base_inputs) (w:lseq bf16 weight_count)
  : Lemma
      (requires weights_represent w input.weight /\ represented_inputs Base input b)
      (ensures
        B.input_representation w b.residual_values b.delta_values b.rotary_values
          canonical_offsets b.fused (base_logical input b) /\
        B.input_mapping (base_logical input b) b.fused input.layer input.cache_before == input /\
        canonical_offsets == B.offsets_of_scalars 0L 2048L 6293504L 4196352L 1024L 6293760L 12585216L) =
  ()
inline_for_extraction noextract
let optimized_input_agrees (input:Shared.prefill_input) (b:optimized_inputs) (w:lseq bf16 weight_count)
  : Lemma
      (requires weights_represent w input.weight /\ represented_inputs Optimized input b)
      (ensures
        O.input_representation b.hidden_values w b.rotary_values (I64.v b.position) (optimized_logical input b) /\
        O.input_mapping (optimized_logical input b) input.layer input.cache_before == input /\
        run_inputs input w b == O.native_inputs b.hidden_values w b.rotary_values
          (I64.v b.position) input.layer input.cache_before) =
  assert (O.input_mapping (optimized_logical input b) input.layer input.cache_before == input) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Review.optimized_logical"; "KPlay.Optimized.Functional.input_mapping";
      "KPlay.Optimized.Functional.stored_hidden"]];
    FStar.Tactics.V2.smt ());
  ()

inline_for_extraction noextract
let semantic_witness (i:implementation) (input:Shared.prefill_input) (b:inputs i)
  (cells:output_cells i) (o:observation i) (h:Shared.hidden_states) (k v:Shared.heads 8)
  : Lemma
      (requires modeled_post i input b cells o h k v)
      (ensures semantic_post i input b cells o) =
  FStar.Classical.exists_intro (fun v -> modeled_post i input b cells o h k v) v;
  FStar.Classical.exists_intro (fun k -> exists v. modeled_post i input b cells o h k v) k;
  FStar.Classical.exists_intro (fun h -> exists k v. modeled_post i input b cells o h k v) h

inline_for_extraction noextract
let base_hidden (r d:lseq bf16 7168) (rh dh:Shared.hidden_states)
  : Lemma
      (requires B.hidden_representation r rh /\ B.hidden_representation d dh)
      (ensures hidden_represents Base (r,d) (Shared.residual_add rh dh)) =
  let h = Shared.residual_add rh dh in
  let represents rmodel dmodel =
    h == Shared.residual_add rmodel dmodel /\
    (forall (t:Shared.index 7) (c:Shared.index 1024).
      Seq.index r (t * 1024 + c) %~ rmodel t c /\ Seq.index d (t * 1024 + c) %~ dmodel t c)
  in
  FStar.Classical.exists_intro (fun dmodel -> represents rh dmodel) dh;
  FStar.Classical.exists_intro (fun rmodel -> exists dmodel. represents rmodel dmodel) rh

inline_for_extraction noextract
let base_post_refines (input:Shared.prefill_input) (b:base_inputs)
  (r d:lseq bf16 7168) (o:B.cache_observation)
  : Lemma
      (requires B.semantic_postcondition input r d o)
      (ensures post Base input b (r,d) o) =
  B.semantic_postcondition_requires_observed_cache input r d o;
  assert ((B.decoder_parts input).B.keys == expected_keys input /\
    (B.decoder_parts input).B.values == expected_values input) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Base.Functional.decoder_parts"; "KPlay.Review.expected_keys"; "KPlay.Review.expected_values"]];
    FStar.Tactics.V2.smt ());
  assert (cache_represents Base o (expected_keys input) (expected_values input));
  let point (logical:B.pending_output)
    : Lemma
        (requires B.output_representation r d logical /\
          B.computed_cache_representation o.B.qkv_values o.B.qk_values logical /\
          B.pending_postcondition input logical)
        (ensures semantic_post Base input b (r,d) o) =
    base_hidden r d logical.B.residual logical.B.delta;
    assert_norm (result input (Shared.residual_add logical.B.residual logical.B.delta)
      logical.B.keys logical.B.values == B.output_mapping input logical);
    semantic_witness Base input b (r,d) o
      (Shared.residual_add logical.B.residual logical.B.delta) logical.B.keys logical.B.values
  in
  FStar.Classical.ghost_lemma point

inline_for_extraction noextract
let padding_agrees (weight:Shared.parameters) (incoming:Shared.vector 1024)
  : Lemma (padding_after weight incoming == O.padding_after weight incoming) = ()
inline_for_extraction noextract
let optimized_cache_cell (input:Shared.prefill_input) (o:O.cache_observation)
  (t:Shared.index 7) (h:Shared.index 8) (c:Shared.index 128) : prop =
  Seq.index o.O.rotated_values (t * 4096 + h * 512 + 256 + c) %~ expected_keys input t h c /\
  Seq.index o.O.rotated_values (t * 4096 + h * 512 + 384 + c) %~ expected_values input t h c
inline_for_extraction noextract
let optimized_cache (input:Shared.prefill_input) (padding:Shared.vector 1024) (o:O.cache_observation)
  : Lemma
      (requires O.computed_cache_representation o (O.decoder_parts input padding))
      (ensures cache_represents Optimized o (expected_keys input) (expected_values input)) =
  let cell (t:Shared.index 7) (h:Shared.index 8) (c:Shared.index 128)
    : Lemma (optimized_cache_cell input o t h c) =
    O.decoder_parts_cache_at input padding t h c;
    ()
  in
  FStar.Classical.forall_intro_3 #(Shared.index 7) #(fun _ -> Shared.index 8)
    #(fun _ _ -> Shared.index 128) #(optimized_cache_cell input o) cell;
  assert (cache_represents Optimized o (expected_keys input) (expected_values input) ==
    (forall (t:Shared.index 7) (h:Shared.index 8) (c:Shared.index 128).
      optimized_cache_cell input o t h c)) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Review.cache_represents"; "KPlay.Review.optimized_cache_cell"]];
    FStar.Tactics.V2.trefl ());
  ()
inline_for_extraction noextract
let optimized_post_refines (input:Shared.prefill_input) (b:optimized_inputs)
  (cells:lseq bf16 8192) (o:O.cache_observation)
  : Lemma
      (requires
        O.computed_cache_representation o (O.decoder_parts input ((FStar.Ghost.reveal b.hidden_model) 7)) /\
        (O.observed_numerical_premises o ==>
          O.required_postcondition input ((FStar.Ghost.reveal b.hidden_model) 7) cells o))
      (ensures post Optimized input b cells o) =
  optimized_cache input ((FStar.Ghost.reveal b.hidden_model) 7) o;
  padding_agrees input.weight ((FStar.Ghost.reveal b.hidden_model) 7);
  assert (numerical_premise Optimized o == O.observed_numerical_premises o);
  let point (logical:O.output_denotation)
    : Lemma
        (requires
          O.output_representation cells logical /\ O.computed_cache_representation o logical /\
          Shared.postcondition input (O.output_mapping input logical) /\
          (forall (c:Shared.index 1024).
            logical.O.hidden 7 c == O.padding_after input.weight ((FStar.Ghost.reveal b.hidden_model) 7) c))
        (ensures semantic_post Optimized input b cells o) =
    assert_norm (result input (fun t c -> logical.O.hidden t c) logical.O.keys logical.O.values ==
      O.output_mapping input logical);
    semantic_witness Optimized input b cells o (fun t c -> logical.O.hidden t c) logical.O.keys logical.O.values
  in
  FStar.Classical.ghost_lemma point

fn base
  (#input:erased Shared.prefill_input) (weights:device_array bf16 weight_count)
  (b:inputs Base) (out:device_array bf16 (hidden_count Base)) (aux:auxiliary Base)
  (#w:erased (lseq bf16 weight_count)) (#pw #ph #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w) ** inputs_own Base ph pr b
  requires pure (weights_represent w input.Shared.weight /\ represented_inputs Base input b) ** output_before Base out aux
  returns o:erased (observation Base)
  ensures exists* (cells:output_cells Base).
    output_owns Base out aux cells ** receipt Base input w b cells o ** pure (post Base input b cells o)
{
  let _ = base_input_agrees input b w;
  unfold inputs_own Base ph pr b;
  unfold output_before Base out aux;
  with old_r old_d. assert on gpu_loc (A.pts_to out old_r ** A.pts_to aux old_d);
  let o = B.kplay_run 15730944UL weights b.residual b.delta b.rotary
    0L 2048L 6293504L 4196352L 1024L 6293760L 12585216L b.fused out aux
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.residual_values))
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.delta_values))
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.rotary_values))
    #old_r #old_d #pw #ph #ph #pr
    #(FStar.Ghost.hide (base_logical input b))
    #(FStar.Ghost.hide input.Shared.layer) #(FStar.Ghost.hide input.Shared.cache_before);
  with r d. assert on gpu_loc (A.pts_to out r ** A.pts_to aux d);
  let _ = base_post_refines input b r d o;
  assert pure (post Base input b (r,d) o);
  fold inputs_own Base ph pr b;
  fold output_owns Base out aux (r,d);
  fold receipt Base input w b (r,d) o;
  o
}

fn optimized
  (#input:erased Shared.prefill_input) (weights:device_array bf16 weight_count)
  (b:inputs Optimized) (out:device_array bf16 (hidden_count Optimized)) (aux:auxiliary Optimized)
  (#w:erased (lseq bf16 weight_count)) (#pw #ph #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w) ** inputs_own Optimized ph pr b
  requires pure (weights_represent w input.Shared.weight /\ represented_inputs Optimized input b) ** output_before Optimized out aux
  returns o:erased (observation Optimized)
  ensures exists* (cells:output_cells Optimized).
    output_owns Optimized out aux cells ** receipt Optimized input w b cells o ** pure (post Optimized input b cells o)
{
  let _ = optimized_input_agrees input b w;
  unfold inputs_own Optimized ph pr b;
  unfold output_before Optimized out aux;
  let run = O.begin_invocation #(FStar.Ghost.hide (run_inputs input w b));
  let o = O.kplay_run 8UL 1024UL 128UL 4096UL 15730944UL 1048576UL
    b.hidden weights b.rotary b.position out
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.hidden_values))
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.rotary_values)) #ph #pw #pr
    #(FStar.Ghost.hide (optimized_logical input b))
    #(FStar.Ghost.hide input.Shared.layer) #(FStar.Ghost.hide input.Shared.cache_before)
    #(FStar.Ghost.hide (run_inputs input w b)) #run;
  with cells. assert on gpu_loc (A.pts_to out cells);
  unfold O.completed_execution #(run_inputs input w b) run
    (O.input_mapping (optimized_logical input b) input.Shared.layer input.Shared.cache_before)
    ((FStar.Ghost.reveal b.hidden_model) 7) cells o;
  let _ = optimized_post_refines input b cells o;
  assert pure (post Optimized input b cells o);
  introduce exists* (r:O.invocation (run_inputs input w b)).
    O.invocation_consumed #(run_inputs input w b) r o **
    O.invocation_result #(run_inputs input w b) r o cells with run;
  fold receipt Optimized input w b cells o;
  fold inputs_own Optimized ph pr b;
  fold output_owns Optimized out aux cells;
  o
}

fn switcher
  (i:implementation) (#input:erased Shared.prefill_input) (weights:device_array bf16 weight_count)
  (b:inputs i) (out:device_array bf16 (hidden_count i)) (aux:auxiliary i)
  (#w:erased (lseq bf16 weight_count)) (#pw #ph #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w) ** inputs_own i ph pr b
  requires pure (weights_represent w input.Shared.weight /\ represented_inputs i input b) ** output_before i out aux
  returns o:erased (observation i)
  ensures exists* (cells:output_cells i).
    output_owns i out aux cells ** receipt i input w b cells o ** pure (post i input b cells o)
{
  match i {
    Base -> { base #input weights b out aux #w #pw #ph #pr }
    Optimized -> { optimized #input weights b out aux #w #pw #ph #pr }
  }
}

#pop-options
