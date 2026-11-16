module KPlay.Review
#lang-pulse

open Kuiper
module S = FStar.Seq
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

type implementation = | Base | Optimized
// 596049920 packed BF16 weights, seven valid prompt tokens, one decode step.
inline_for_extraction noextract
let inputs = B.source_inputs
inline_for_extraction noextract
let addresses = B.source_input_addresses

inline_for_extraction noextract
let input_model (i:implementation) (input:inputs) : GTot Shared.inputs =
  match i with
  | Base -> B.input_mapping input
  | Optimized -> O.mg_inputs input.B.weights input.B.prompt

// Equality of prompt/decode count and every embedding, normalization and layer weight.
val input_models_agree : input:inputs ->
  Lemma (Shared.same_inputs (input_model Base input) (input_model Optimized input))

inline_for_extraction noextract
let input_cells (a:addresses) (input:inputs) : slprop =
  on gpu_loc (A.pts_to a.B.weights_address input.B.weights) **
  on gpu_loc (A.pts_to a.B.prompt_address input.B.prompt)

// Observations are from the returned execution: base FP32, optimized BF16
// rotary/layer/cache values; both use FP32 scores and rotary metadata.
inline_for_extraction noextract
let base_result (#frame_t:Type0) (input:inputs) (v:B.source_program_values)
  (tokens:lseq I64.t 8{B.source_selection_result (B.source_program_slot v 283) input.B.prompt tokens})
  (frame:frame_t) : GTot prop =
  let native_run = B.source_program_execution input v (I64.v (S.index tokens 7)) in
  let run = Shared.mixed_legacy_execution native_run in
  let output = B.source_program_outputs input v tokens in
  let after = Shared.mixed_legacy_state (B.source_program_after_state input v tokens frame) in
  Shared.mixed_structural_postcondition (input_model Base input) frame run output after /\
  (B.source_rotary_domain () ==> Shared.mixed_numerical_execution (input_model Base input) run) /\
  (Shared.selector_domain run.Shared.scores ==>
    Shared.maximal_actual_index run.Shared.scores run.Shared.selected) /\
  (B.source_rotary_domain () /\ Shared.selector_domain run.Shared.scores ==>
    Shared.mixed_postcondition (input_model Base input) frame run output after /\
    Shared.mixed_numerical_denotation (input_model Base input) run)

inline_for_extraction noextract
let optimized_typed_result (#frame_t:Type0) (input:inputs) (v:O.pg_values)
  (tokens:O.mg_valid_tokens) (frame:frame_t) : GTot prop =
  let run = O.mg_execution v (I64.v (S.index tokens 7)) in
  let output = O.mg_outputs v tokens in
  let after = O.mg_after_state input.B.weights input.B.prompt v tokens frame in
  Shared.mixed_structural_postcondition (input_model Optimized input) frame run output after /\
  (O.source_rotary_domain () /\ O.pg_attention_domain v ==>
    Shared.mixed_numerical_execution (input_model Optimized input) run /\
    Shared.mixed_numerical_denotation (input_model Optimized input) run) /\
  (Shared.selector_domain run.Shared.scores ==>
    Shared.maximal_actual_index run.Shared.scores run.Shared.selected) /\
  (O.source_rotary_domain () /\ O.pg_attention_domain v /\ Shared.selector_domain run.Shared.scores ==>
    Shared.mixed_postcondition (input_model Optimized input) frame run output after /\
    Shared.matrix_approximates output.Shared.normalized_hidden
      (Shared.mathematical_hidden (Shared.real_weights (input_model Optimized input).Shared.weight)
        (input_model Optimized input).Shared.prompt) /\
    Shared.maximal_actual_index run.Shared.scores (S.index output.Shared.tokens 7))

// Outside the actual score-domain premise the optimized selector may return -1.
// Do not confuse that raw return with a valid Shared.outputs witness.
inline_for_extraction noextract
let optimized_result (#frame_t:Type0) (input:inputs) (v:O.pg_values)
  (tokens:lseq I64.t 8) (frame:frame_t) : GTot prop =
  O.mg_raw_structural input.B.weights input.B.prompt v tokens /\
  O.pg_same_execution_edges input.B.weights v input.B.prompt tokens /\
  (Shared.selector_domain (O.pg_selector_scores (O.pg_slot v 255)) ==> O.mg_tokens_valid tokens) /\
  (O.mg_tokens_valid tokens ==> optimized_typed_result input v tokens frame)

// Checked equivalences, not assumptions that native implementations meet a new spec.
val base_result_equivalence : #frame_t:Type0 -> input:inputs -> v:B.source_program_values ->
  tokens:lseq I64.t 8{B.source_selection_result (B.source_program_slot v 283) input.B.prompt tokens} ->
  frame:frame_t ->
  Lemma (B.source_program_result input v tokens frame <==> base_result input v tokens frame)
val optimized_result_equivalence : #frame_t:Type0 -> input:inputs -> v:O.pg_values ->
  tokens:lseq I64.t 8 -> frame:frame_t ->
  Lemma (O.mg_program_result input.B.weights input.B.prompt v tokens frame <==>
    optimized_result input v tokens frame)

noextract
noeq type outcome =
  | Failed : error:RT.allocation_error -> outcome
  | Returned : tokens:RT.tensor I64.t -> hidden:RT.tensor BF16.t -> outcome

inline_for_extraction noextract
let completion_epoch (i:implementation) (start:Epoch.epoch_t) =
  start + (match i with | Base -> 229 | Optimized -> 256)

// Both return exactly seven hidden rows at row*1024+column.
// The optimized decoder's eight-row padding is internal, not returned.
inline_for_extraction noextract
let output_cells (#frame_t:Type0) (i:implementation) (a:addresses) (input:inputs)
  (tokens:RT.tensor I64.t) (hidden:RT.tensor BF16.t) (frame:frame_t) : slprop =
  on gpu_loc (
    A.pts_to a.B.weights_address input.B.weights **
    A.pts_to a.B.prompt_address input.B.prompt **
    (match i with
    | Base ->
      exists* (v:B.source_program_values)
        (values:lseq I64.t 8{B.source_selection_result (B.source_program_slot v 283) input.B.prompt values}).
        A.pts_to (RT.data tokens) values **
        A.pts_to (RT.data hidden) (B.source_program_slot v 282) **
        pure (B.source_program_stage input v 228 /\ base_result input v values frame)
    | Optimized ->
      exists* (v:O.pg_values) (values:lseq I64.t 8).
        A.pts_to (RT.data hidden) (O.pg_slot v 254) **
        A.pts_to (RT.data tokens) values **
        pure (O.pg_suffix_arithmetic input.B.weights input.B.prompt v values 3 /\
          optimized_result input v values frame)))

inline_for_extraction noextract
let post (#frame_t:Type0) (i:implementation) (a:addresses) (input:inputs)
  (stream:Stream.stream_t) (start:Epoch.epoch_t) (frame:frame_t) (result:outcome) : slprop =
  match result with
  | Failed _ ->
    Epoch.epoch_live stream start **
    Pledge.pledge0 (Epoch.epoch_done stream start) (input_cells a input)
  | Returned tokens hidden ->
    Epoch.epoch_live stream (completion_epoch i start) **
    RT.owner stream tokens ** RT.owner stream hidden **
    Pledge.pledge0 (Epoch.epoch_done stream (completion_epoch i start))
      (output_cells i a input tokens hidden frame) **
    pure (RT.metadata tokens [1; 8] /\ RT.metadata hidden [7; 1024])

noextract
fn base (#frame_t:Type0) (#frame:erased frame_t)
  (a:addresses) (#input:erased inputs) (stream:Stream.stream_t)
  (#start:Epoch.epoch_t) (ambient:slprop)
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** input_cells a input
  returns result:outcome
  ensures post Base a input stream start frame result

noextract
fn optimized (#frame_t:Type0) (#frame:erased frame_t)
  (a:addresses) (#input:erased inputs) (stream:Stream.stream_t)
  (#start:Epoch.epoch_t) (ambient:slprop)
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** input_cells a input
  returns result:outcome
  ensures post Optimized a input stream start frame result

(*
Signature relation: both wrappers use the same packed inputs (596049920 BF16
weights, seven prompt tokens, one decode step), addresses, stream, ambient
frame and outcome. input_models_agree proves prompt/decode equality and
pointwise equality of every weight. Both return token metadata [1;8] and
hidden metadata [7;1024]; the optimized decoder's eight-row padding is
INTERNAL. Success completes at start+229 for Base and start+256 for
Optimized. Allocation failure preserves the starting epoch and input pledge
in both.

base_result converts native FP32 layer/rotary/cache observations through
mixed_legacy_execution/state. optimized_typed_result uses BF16
layer/rotary/cache observations, with FP32 metadata and scores in both. The
exposed result equivalence lemmas relate these predicates exactly to the
native contracts. Both numerical guarantees retain their rotary domains;
optimized additionally requires pg_attention_domain of its actual returned
snapshot. Its raw result guarantees valid tokens under the actual selector
domain and gives the typed result when tokens are valid; outside that domain
-1 remains possible.

Assessment: high semantic significance for validity/numerical premises; the
epoch difference is additionally relevant to scheduling, not decode count or
a runtime ratio. Complete relative structural/execution correctness is
proved, not a whole-program uniqueness/tightness certificate or equal tokens
across runs. The body below only dispatches; frame adaptation, allocation
and enqueuing belong to the checked wrappers and their native
implementations.
*)
noextract
fn switcher (i:implementation) (#frame_t:Type0) (#frame:erased frame_t)
  (a:addresses) (#input:erased inputs) (stream:Stream.stream_t)
  (#start:Epoch.epoch_t) (ambient:slprop)
  preserves cpu ** Stream.stream_live stream ** RT.runtime stream ** ambient
  requires Epoch.epoch_live stream start ** input_cells a input
  returns result:outcome
  ensures post i a input stream start frame result
{
  match i {
    Base -> { base #frame_t #frame a #input stream #start ambient }
    Optimized -> { optimized #frame_t #frame a #input stream #start ambient }
  }
}


#pop-options
