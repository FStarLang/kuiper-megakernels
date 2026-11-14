module KPlay.Review
#lang-pulse

open Kuiper
open Pulse.Lib.Send

module Shared = KPlay.SharedSpec
module Native = KPlay.Proof.Shared.Native
module Input = Kuiper.Tensor.Input
module RT = Kuiper.Tensor.Runtime
module Results = Kuiper.Tensor.Results
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge

(* Both functions refine the same task relation on actual returned cells.
   Only completion differs: one launch for base, three for optimized.
   The .fst proves the adapters and calls both verified implementations.
   This is not a claim that two executions return equal native outputs. *)

noextract type token_values = lseq bf16 (256 * 512)
noextract type weight_values = lseq bf16 (8 * 512)
noextract type vector = Results.vector4 i32 i64 f32 i32
noextract type implementation = | BaseRole | OptimizedRole

noextract noeq
type outcome =
  | AllocationFailed : error:RT.allocation_error -> outcome
  | Returned : result:Results.construction_result i32 i64 f32 i32 -> outcome

inline_for_extraction noextract unfold
let completion (implementation:implementation) (epoch:Epoch.epoch_t) : Epoch.epoch_t =
  match implementation with
  | BaseRole -> Epoch.epoch_next epoch
  | OptimizedRole -> Epoch.epoch_next (Epoch.epoch_next (Epoch.epoch_next epoch))

inline_for_extraction noextract
let handles (value:vector) : GTot Native.output_handles = {
  demand = Results.first value;
  indices = Results.second value;
  probabilities = Results.third value;
  survival = Results.fourth value
}

inline_for_extraction noextract
let output_payload
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (value:vector)
  (cells:Native.output_values) (out:Shared.public_outputs) : slprop =
  Native.output_cells (handles value) cells **
  pure (Native.output_binding (handles value) cells out /\
    Shared.task_relation x w x w out)

inline_for_extraction noextract
let output (x:Shared.token_matrix) (w:Shared.weight_matrix) (value:vector) : slprop =
  exists* (cells:Native.output_values) (out:Shared.public_outputs).
    output_payload x w value cells out

inline_for_extraction noextract
let before_call
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument bf16)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  {| is_send frame |} : slprop =
  Native.call_resources stream epoch tokens weights fp fw xs ws frame **
  pure (Native.input_binding tokens weights xs ws x w /\ Shared.input_domain x w)

inline_for_extraction noextract
let after_call
  (implementation:implementation) (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument bf16)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  (result:outcome) : slprop =
  cpu ** Input.reference tokens ** Input.reference weights **
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
   | AllocationFailed _ ->
     Epoch.epoch_live stream epoch **
     Pledge.pledge0 (Epoch.epoch_done stream epoch)
       (Native.input_cells tokens weights fp fw xs ws ** frame)
   | Returned result ->
     Epoch.epoch_live stream (completion implementation epoch) **
     (match result with
      | Results.Failed _ ->
        Pledge.pledge0 (Epoch.epoch_done stream (completion implementation epoch))
          (Native.input_cells tokens weights fp fw xs ws ** frame)
      | Results.Constructed value ->
        Native.output_owners stream (handles value) **
        Pledge.pledge0 (Epoch.epoch_done stream (completion implementation epoch))
          ((Native.input_cells tokens weights fp fw xs ws **
            output x w value) ** frame)))

inline_for_extraction noextract
fn base
  (tokens weights:Input.argument bf16)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (#fp #fw:perm) (#xs:erased token_values) (#ws:erased weight_values)
  (#x:erased Shared.token_matrix) (#w:erased Shared.weight_matrix)
  (frame:slprop) {| is_send frame |} ()
  requires before_call stream epoch tokens weights xs ws fp fw x w frame
  returns result:outcome
  ensures after_call BaseRole stream epoch tokens weights xs ws fp fw x w frame result

inline_for_extraction noextract
fn optimized
  (tokens weights:Input.argument bf16)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (#fp #fw:perm) (#xs:erased token_values) (#ws:erased weight_values)
  (#x:erased Shared.token_matrix) (#w:erased Shared.weight_matrix)
  (frame:slprop) {| is_send frame |} ()
  requires before_call stream epoch tokens weights xs ws fp fw x w frame
  returns result:outcome
  ensures after_call OptimizedRole stream epoch tokens weights xs ws fp fw x w frame result

inline_for_extraction noextract
fn switcher
  (i:implementation)
  (tokens weights:Input.argument bf16)
  (stream:Stream.stream_t) (#epoch:Epoch.epoch_t)
  (#fp #fw:perm) (#xs:erased token_values) (#ws:erased weight_values)
  (#x:erased Shared.token_matrix) (#w:erased Shared.weight_matrix)
  (frame:slprop) {| is_send frame |} ()
  requires before_call stream epoch tokens weights xs ws fp fw x w frame
  returns result:outcome
  ensures after_call i stream epoch tokens weights xs ws fp fw x w frame result
