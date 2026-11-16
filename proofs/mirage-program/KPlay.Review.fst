module KPlay.Review
#lang-pulse

open Kuiper
open Pulse.Lib.Send
module S = FStar.Seq
module C = FStar.Classical
module A = Pulse.Lib.Array
module I64 = FStar.Int64
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16
module RT = Kuiper.Tensor.Runtime
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--fuel 2 --ifuel 1 --z3rlimit 10"

let input_models_agree input =
  let left = input_model Base input in
  let right = input_model Optimized input in
  let prompt_at (i:Shared.index 7)
    : Lemma (S.index left.Shared.prompt i == S.index right.Shared.prompt i) = () in
  C.forall_intro #(Shared.index 7)
    #(fun i -> S.index left.Shared.prompt i == S.index right.Shared.prompt i) prompt_at;
  S.lemma_eq_intro left.Shared.prompt right.Shared.prompt;
  S.lemma_eq_elim left.Shared.prompt right.Shared.prompt;
  let layer_at (layer:Shared.index 28)
    : Lemma (Shared.same_parameters (left.Shared.weight.Shared.layers layer)
      (right.Shared.weight.Shared.layers layer)) =
    assert (Shared.same_parameters (left.Shared.weight.Shared.layers layer)
      (right.Shared.weight.Shared.layers layer)) by (
      FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
        "KPlay.Review.input_model"; "KPlay.Base.Functional.input_mapping";
        "KPlay.Base.Functional.source_parameters"; "KPlay.Base.Functional.layer_start";
        "KPlay.Base.Functional.suffix_weight_elements"; "KPlay.Base.Functional.embedding_elements";
        "KPlay.Base.Functional.layer_weight_elements";
        "KPlay.Optimized.Functional.mg_inputs"; "KPlay.Optimized.Functional.mg_model_weights";
        "KPlay.Optimized.Functional.pg_parameters"; "KPlay.Optimized.Functional.pg_layer_weights";
        "KPlay.Optimized.Functional.pg_layer_base";
        "KPlay.SharedSpec.same_parameters"; "KPlay.SharedSpec.same_vector"; "KPlay.SharedSpec.same_matrix"]];
      FStar.Tactics.V2.smt ()) in
  C.forall_intro #(Shared.index 28)
    #(fun layer -> Shared.same_parameters (left.Shared.weight.Shared.layers layer)
      (right.Shared.weight.Shared.layers layer)) layer_at;
  ()

let normalize_base_contract () =
    FStar.Tactics.V2.norm [zeta_full; iota; delta_only [
      "KPlay.Review.base_result"; "KPlay.Review.input_model";
      "KPlay.Review.output_cells"; "KPlay.Base.Functional.source_program_return_frame";
      "KPlay.Base.Functional.source_program_execution";
      "KPlay.Base.Functional.source_program_result";
      "KPlay.SharedSpec.mixed_legacy_execution"; "KPlay.SharedSpec.mixed_legacy_state";
      "KPlay.SharedSpec.postcondition"; "KPlay.SharedSpec.mixed_postcondition";
      "KPlay.SharedSpec.structural_postcondition"; "KPlay.SharedSpec.mixed_structural_postcondition";
      "KPlay.SharedSpec.embedding_copy"; "KPlay.SharedSpec.mixed_embedding_copy";
      "KPlay.SharedSpec.cache_execution"; "KPlay.SharedSpec.mixed_cache_execution";
      "KPlay.SharedSpec.numerical_execution"; "KPlay.SharedSpec.mixed_numerical_execution";
      "KPlay.SharedSpec.rotary_generation"; "KPlay.SharedSpec.mixed_rotary_generation";
      "KPlay.SharedSpec.numerical_denotation"; "KPlay.SharedSpec.mixed_numerical_denotation"]]

let base_result_equivalence #frame_t input v tokens frame =
  assert (B.source_program_result input v tokens frame <==> base_result input v tokens frame) by (
    normalize_base_contract ();
    FStar.Tactics.V2.smt ())

let normalize_optimized_contract () =
    FStar.Tactics.V2.norm [zeta_full; iota; delta_only [
      "KPlay.Review.optimized_result"; "KPlay.Review.optimized_typed_result";
      "KPlay.Review.output_cells"; "KPlay.Optimized.Functional.program_return_frame";
      "KPlay.Review.input_model"; "KPlay.Optimized.Functional.mg_program_result";
      "KPlay.Optimized.Functional.mg_typed_result"; "KPlay.Optimized.Functional.mg_inputs"]]

let optimized_result_equivalence #frame_t input v tokens frame =
  assert (O.mg_program_result input.B.weights input.B.prompt v tokens frame <==>
    optimized_result input v tokens frame) by (
    normalize_optimized_contract ();
    FStar.Tactics.V2.smt ())

let base_frame_agrees (#frame_t:Type0) (a:addresses) (input:inputs)
  (output:B.source_returned_result) (frame:frame_t)
  : Lemma (B.source_program_return_frame a input output frame ==
    output_cells Base a input output.B.returned_tokens output.B.returned_hidden frame) =
  assert (B.source_program_return_frame a input output frame ==
    output_cells Base a input output.B.returned_tokens output.B.returned_hidden frame) by (
    normalize_base_contract ();
    FStar.Tactics.V2.trefl ())

let optimized_frame_agrees (#frame_t:Type0) (a:addresses) (input:inputs)
  (output:O.pg_returned_result) (frame:frame_t)
  : Lemma (O.program_return_frame a.B.weights_address a.B.prompt_address input.B.weights input.B.prompt output frame ==
    output_cells Optimized a input output.O.returned_tokens output.O.returned_hidden frame) =
  assert (O.program_return_frame a.B.weights_address a.B.prompt_address input.B.weights input.B.prompt output frame ==
    output_cells Optimized a input output.O.returned_tokens output.O.returned_hidden frame) by (
    normalize_optimized_contract ();
    FStar.Tactics.V2.trefl ())

inline_for_extraction noextract
ghost
fn base_cells (#frame_t:Type0) (a:addresses) (input:inputs)
  (output:B.source_returned_result) (frame:frame_t) ()
  requires B.source_program_return_frame a input output frame
  ensures output_cells Base a input output.B.returned_tokens output.B.returned_hidden frame
{
  base_frame_agrees a input output frame;
  rewrite B.source_program_return_frame a input output frame
    as output_cells Base a input output.B.returned_tokens output.B.returned_hidden frame;
}

inline_for_extraction noextract
ghost
fn optimized_cells (#frame_t:Type0) (a:addresses) (input:inputs)
  (output:O.pg_returned_result) (frame:frame_t) ()
  requires O.program_return_frame a.B.weights_address a.B.prompt_address input.B.weights input.B.prompt output frame
  ensures output_cells Optimized a input output.O.returned_tokens output.O.returned_hidden frame
{
  optimized_frame_agrees a input output frame;
  rewrite O.program_return_frame a.B.weights_address a.B.prompt_address input.B.weights input.B.prompt output frame
    as output_cells Optimized a input output.O.returned_tokens output.O.returned_hidden frame;
}

noextract
fn base (#frame_t:Type0) (#frame:erased frame_t)
  (a:addresses) (#input:erased inputs) (stream:Stream.stream_t)
  (#start:Epoch.epoch_t) (ambient:slprop)
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** input_cells a input
  returns result:outcome
  ensures post Base a input stream start frame result
{
  rewrite input_cells a input as B.source_borrowed_inputs a input;
  let result = B.program_refines #frame_t #frame a #input stream #start ambient;
  match result {
    B.SourceProgramFailed error -> {
      fold post Base a input stream start frame (Failed error);
      Failed error
    }
    B.SourceProgramReturned output -> {
      Pledge.rewrite_pledge
        (B.source_program_return_frame a input output frame)
        (output_cells Base a input output.B.returned_tokens output.B.returned_hidden frame)
        (base_cells a input output frame);
      fold post Base a input stream start frame (Returned output.B.returned_tokens output.B.returned_hidden);
      Returned output.B.returned_tokens output.B.returned_hidden
    }
  }
}

noextract
let input_weights (input:inputs) : GTot (lseq BF16.t 596049920) = input.B.weights
noextract
let input_prompt (input:inputs) : GTot O.opt_prompt_values = input.B.prompt

noextract
fn optimized (#frame_t:Type0) (#frame:erased frame_t)
  (a:addresses) (#input:erased inputs) (stream:Stream.stream_t)
  (#start:Epoch.epoch_t) (ambient:slprop)
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** input_cells a input
  returns result:outcome
  ensures post Optimized a input stream start frame result
{
  rewrite input_cells a input
    as O.u_input_storage a.B.weights_address a.B.prompt_address (input_weights input) (input_prompt input);
  let result = O.program_refines #frame_t #frame a.B.weights_address a.B.prompt_address
    #(FStar.Ghost.hide (input_weights input)) #(FStar.Ghost.hide (input_prompt input)) stream #start ambient;
  match result {
    O.PGSourceFailed error -> {
      fold post Optimized a input stream start frame (Failed error);
      Failed error
    }
    O.PGSourceReturned output -> {
      Pledge.rewrite_pledge
        (O.program_return_frame a.B.weights_address a.B.prompt_address (input_weights input) (input_prompt input) output frame)
        (output_cells Optimized a input output.O.returned_tokens output.O.returned_hidden frame)
        (optimized_cells a input output frame);
      fold post Optimized a input stream start frame (Returned output.O.returned_tokens output.O.returned_hidden);
      Returned output.O.returned_tokens output.O.returned_hidden
    }
  }
}

#pop-options
