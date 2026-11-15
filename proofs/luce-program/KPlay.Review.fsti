module KPlay.Review
#lang-pulse

open Kuiper
module A = Pulse.Lib.Array
module I64 = FStar.Int64
module RT = Kuiper.Tensor.Runtime
module Stream = Kuiper.Kernel.Stream
module Epoch = Kuiper.Epoch
module Pledge = Pulse.Lib.Pledge
module NativeInput = KPlay.Proof.NativeInput
module Model = KPlay.Proof.Shared.Model
module D = KPlay.Proof.Shared.Selection
module Shared = KPlay.SharedSpec
module Storage = KPlay.Proof.Shared.Storage
module BI = KPlay.Proof.Base.Inputs
module BK = KPlay.Proof.Base.KernelState
module BR = KPlay.Proof.Base.KernelResult
module Layout = KPlay.Proof.Optimized.Layout
module Execution = KPlay.Proof.Optimized.Execution
module Math = KPlay.Proof.Optimized.ProgramMath

#push-options "--fuel 2 --ifuel 1 --z3rlimit 10"

type implementation = | Base | Optimized
inline_for_extraction noextract
let packed = lseq bf16 752393024

// Both physical packing maps denote these same parameters and input cells.
val input_views_agree :
  call:Model.invocation -> inputs:NativeInput.arguments call.Model.prompt_length ->
  w:packed -> physical:lseq I64.t call.Model.prompt_length ->
  Lemma (
    BI.parameter_values w == Layout.parameter_values w /\
    BI.parameter_cells inputs == Layout.parameter_cells inputs /\
    BI.prompt_storage inputs physical == Layout.prompt_storage inputs physical)

inline_for_extraction noextract
let input_cells (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (w:packed) (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) : slprop =
  Shared.caller_precondition call I64.v permission
    (Layout.parameter_values w) (Layout.parameter_cells inputs)
    prompt (Layout.prompt_storage inputs physical)

noextract
noeq type buffers (call:Model.invocation) = {
  tokens: A.larray I64.t call.Model.decode_steps;
  hidden: A.larray bf16 (call.Model.decode_steps * 1024);
}
inline_for_extraction noextract
let output_storage (call:Model.invocation) (b:buffers call)
  (values:lseq I64.t call.Model.decode_steps)
  : Storage.history_storage I64.t bf16 call.Model.decode_steps = {
  Storage.tokens = {
    Storage.contents = values;
    Storage.addresses = (fun i -> {Storage.buffer = b.tokens; Storage.offset = i});
  };
  Storage.hidden = (fun t c -> {Storage.buffer = b.hidden; Storage.offset = t * 1024 + c});
}

inline_for_extraction noextract
let base_observation (call:Model.invocation) =
  m:BK.metadata call{m.BK.completed == Model.calls call}
inline_for_extraction noextract
let optimized_observation (call:Model.invocation) =
  p:Execution.prefix call{p.Execution.count == Model.calls call}

// The premise concerns the actual returned execution, not an assumed replacement run.
// Base uses BF16 convolution/normalization observations; optimized uses FP32.
inline_for_extraction noextract
let optimized_numerical_premise (call:Model.invocation) (w:packed)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (p:optimized_observation call) : prop =
  Math.family_dependencies w prompt p /\ Math.numerical_scope p

inline_for_extraction noextract
let observed_payload (i:implementation) (call:Model.invocation)
  (inputs:NativeInput.arguments call.Model.prompt_length) (b:buffers call)
  (w:packed) (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) : slprop =
  match i with
  | Base ->
    exists* (m:base_observation call) (values:lseq I64.t call.Model.decode_steps).
      Shared.safety_caller_postcondition call I64.v (fun x -> x) permission
        (Layout.parameter_values w) (Layout.parameter_cells inputs)
        prompt (Layout.prompt_storage inputs physical)
        (BR.run call w prompt m) (BR.output call w prompt m)
        (output_storage call b values) (BK.caller call w prompt m) **
      pure (Shared.numerical_prefix call (Layout.parameter_values w) prompt
        (BR.run call w prompt m) (Model.calls call))
  | Optimized ->
    exists* (p:optimized_observation call) (values:lseq I64.t call.Model.decode_steps).
      Shared.safety_caller_postcondition call I64.v (fun x -> x) permission
        (Layout.parameter_values w) (Layout.parameter_cells inputs)
        prompt (Layout.prompt_storage inputs physical)
        (Execution.run w prompt p) (Execution.caller w prompt p p.Execution.count).D.returned
        (output_storage call b values) (Execution.caller w prompt p p.Execution.count) **
      pure (optimized_numerical_premise call w prompt p ==>
        Shared.numerical_prefix call (Layout.parameter_values w) prompt
          (Execution.run w prompt p) (Model.calls call))

inline_for_extraction noextract
let output_cells (i:implementation) (call:Model.invocation)
  (inputs:NativeInput.arguments call.Model.prompt_length) (tokens:RT.tensor I64.t) (hidden:RT.tensor bf16)
  (w:packed) (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) : slprop =
  exists* (b:buffers call).
    pure (b.tokens == RT.data tokens /\ b.hidden == RT.data hidden) **
    observed_payload i call inputs b w physical prompt permission

noextract
noeq type outcome =
  | Failed : error:RT.allocation_error -> outcome
  | Returned : tokens:RT.tensor I64.t -> hidden:RT.tensor bf16 -> outcome

inline_for_extraction noextract
let completion_epoch (_:implementation) (epoch:Epoch.epoch_t) = Epoch.epoch_next epoch

inline_for_extraction noextract
let pre (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t) (w:packed)
  (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) : slprop =
  Stream.stream_live stream ** RT.runtime stream ** Epoch.epoch_live stream epoch **
  Pledge.pledge0 (Epoch.epoch_done stream epoch)
    (on gpu_loc (input_cells call inputs w physical prompt permission))

// Success returns owning tensors while their complete task postcondition is still queued.
// Failure preserves the original queue tail and input-cell pledge.
inline_for_extraction noextract
let post (i:implementation) (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t) (w:packed)
  (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) (result:outcome) : slprop =
  Stream.stream_live stream ** RT.runtime stream **
  (match result with
  | Failed _ ->
      Epoch.epoch_live stream epoch **
      Pledge.pledge0 (Epoch.epoch_done stream epoch)
        (on gpu_loc (input_cells call inputs w physical prompt permission))
  | Returned tokens hidden ->
      Epoch.epoch_live stream (completion_epoch i epoch) **
      RT.owner stream tokens ** RT.owner stream hidden **
      Pledge.pledge0 (Epoch.epoch_done stream (completion_epoch i epoch))
        (on gpu_loc (output_cells i call inputs tokens hidden w physical prompt permission)) **
      pure (RT.metadata tokens [call.Model.decode_steps] /\
        RT.metadata hidden [call.Model.decode_steps; 1024]))

noextract
fn base
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (steps:I64.t{I64.v steps == call.Model.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#w:erased packed)
  (#physical:erased (lseq I64.t call.Model.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#permission:perm)
  preserves cpu
  requires pre call inputs stream epoch w physical prompt permission
  returns result:outcome
  ensures post Base call inputs stream epoch w physical prompt permission result

noextract
fn optimized
  (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (steps:I64.t{I64.v steps == call.Model.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#w:erased packed)
  (#physical:erased (lseq I64.t call.Model.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#permission:perm)
  preserves cpu
  requires pre call inputs stream epoch w physical prompt permission
  returns result:outcome
  ensures post Optimized call inputs stream epoch w physical prompt permission result

noextract
fn switcher
  (i:implementation) (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (steps:I64.t{I64.v steps == call.Model.decode_steps}) (stream:Stream.stream_t)
  (#epoch:Epoch.epoch_t) (#w:erased packed)
  (#physical:erased (lseq I64.t call.Model.prompt_length))
  (#prompt:erased (D.vec (D.token 248320) call.Model.prompt_length)) (#permission:perm)
  preserves cpu
  requires pre call inputs stream epoch w physical prompt permission
  returns result:outcome
  ensures post i call inputs stream epoch w physical prompt permission result

#pop-options
