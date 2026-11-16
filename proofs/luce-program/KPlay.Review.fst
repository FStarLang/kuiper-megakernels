module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
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
module BI = KPlay.Proof.Base.Inputs
module BK = KPlay.Proof.Base.KernelState
module BR = KPlay.Proof.Base.KernelResult
module BS = KPlay.Proof.Base.KernelStorage
module BQ = KPlay.Proof.Base.KernelQueue
module BE = KPlay.Proof.Base.Entry
module BP = KPlay.Proof.Base.Program
module BT = KPlay.Proof.Base.ProgramTypes
module OP = KPlay.Proof.Optimized.Program
module OQ = KPlay.Proof.Optimized.Queue
module OR = KPlay.Proof.Optimized.Return
module OE = KPlay.Proof.Optimized.Entry
module Layout = KPlay.Proof.Optimized.Layout
module Execution = KPlay.Proof.Optimized.Execution
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--fuel 4 --ifuel 2 --z3rlimit 10"

let input_views_agree call inputs w physical =
  assert (
    BI.parameter_values w == Layout.parameter_values w /\
    BI.parameter_cells inputs == Layout.parameter_cells inputs /\
    BI.prompt_storage inputs physical == Layout.prompt_storage inputs physical) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Proof.Base.Inputs.parameter_values"; "KPlay.Proof.Base.Inputs.parameter_cells";
      "KPlay.Proof.Base.Inputs.prompt_storage";
      "KPlay.Proof.Base.Inputs.parameters"; "KPlay.Proof.Base.Inputs.layer_base";
      "KPlay.Proof.Base.Inputs.layer_extent"; "KPlay.Proof.Optimized.Layout.parameters";
      "KPlay.Proof.Optimized.Layout.linear_weights"; "KPlay.Proof.Optimized.Layout.full_weights";
      "KPlay.Proof.Optimized.Layout.layer_base"; "KPlay.Proof.Optimized.Layout.layer_partition";
      "KPlay.Proof.Optimized.Layout.parameter_values"; "KPlay.Proof.Optimized.Layout.parameter_cells";
      "KPlay.Proof.Optimized.Layout.input_addresses"; "KPlay.Proof.Optimized.Layout.prompt_storage"]];
    FStar.Tactics.Canon.canon ();
    FStar.Tactics.V2.smt ())

let base_output_storage_agrees (call:Model.invocation) (args:BE.arguments call)
  (values:lseq I64.t call.Model.decode_steps)
  : Lemma (BS.output_at call args values ==
    output_storage call {tokens = args.BE.tokens; hidden = args.BE.hidden} values) =
  assert_norm (BS.output_at call args values ==
    output_storage call {tokens = args.BE.tokens; hidden = args.BE.hidden} values)

let optimized_output_storage_agrees (call:Model.invocation) (args:OE.arguments call)
  (values:lseq I64.t call.Model.decode_steps)
  : Lemma (OR.output_addresses call args values ==
    output_storage call {tokens = args.OE.tokens; hidden = args.OE.hidden} values) =
  assert_norm (OR.output_addresses call args values ==
    output_storage call {tokens = args.OE.tokens; hidden = args.OE.hidden} values)

inline_for_extraction noextract
ghost
fn base_cells_refine (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (tokens:RT.tensor I64.t) (hidden:RT.tensor bf16) (w:packed)
  (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) ()
  requires BQ.result_cells call inputs tokens hidden w physical prompt permission
  ensures output_cells Base call inputs tokens hidden w physical prompt permission
{
  input_views_agree call inputs w physical;
  unfold BQ.result_cells call inputs tokens hidden w physical prompt permission;
  with args. assert pure (args.BE.tokens == RT.data tokens /\ args.BE.hidden == RT.data hidden) **
    BS.exported call inputs args w physical prompt permission;
  unfold BS.exported call inputs args w physical prompt permission;
  with m values. assert Shared.caller_postcondition call I64.v (fun x -> x) permission
    (BI.parameter_values w) (BI.parameter_cells inputs) prompt (BI.prompt_storage inputs physical)
    (BR.run call w prompt m) (BR.output call w prompt m) (BS.output_at call args values)
    (BK.caller call w prompt m);
  let b:buffers call = {tokens = args.BE.tokens; hidden = args.BE.hidden};
  base_output_storage_agrees call args values;
  assert pure (BS.output_at call args values == output_storage call b values);
  rewrite each (BI.parameter_values w) as (Layout.parameter_values w);
  rewrite each (BI.parameter_cells inputs) as (Layout.parameter_cells inputs);
  rewrite each (BI.prompt_storage inputs physical) as (Layout.prompt_storage inputs physical);
  rewrite each (BS.output_at call args values) as (output_storage call b values);
  unfold Shared.caller_postcondition call I64.v (fun x -> x) permission
    (Layout.parameter_values w) (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical)
    (BR.run call w prompt m) (BR.output call w prompt m) (output_storage call b values)
    (BK.caller call w prompt m);
  introduce exists* (m:base_observation call) (values:lseq I64.t call.Model.decode_steps).
    Shared.safety_caller_postcondition call I64.v (fun x -> x) permission
      (Layout.parameter_values w) (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical)
      (BR.run call w prompt m) (BR.output call w prompt m) (output_storage call b values)
      (BK.caller call w prompt m) **
    pure (Shared.numerical_prefix call (Layout.parameter_values w) prompt
      (BR.run call w prompt m) (Model.calls call)) with m values;
  fold observed_payload Base call inputs b w physical prompt permission;
  introduce exists* (b:buffers call).
    pure (b.tokens == RT.data tokens /\ b.hidden == RT.data hidden) **
    observed_payload Base call inputs b w physical prompt permission with b;
  fold output_cells Base call inputs tokens hidden w physical prompt permission;
}

inline_for_extraction noextract
ghost
fn optimized_cells_refine (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (tokens:RT.tensor I64.t) (hidden:RT.tensor bf16) (w:packed)
  (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) ()
  requires OQ.result_cells call inputs tokens hidden w physical prompt permission
  ensures output_cells Optimized call inputs tokens hidden w physical prompt permission
{
  unfold OQ.result_cells call inputs tokens hidden w physical prompt permission;
  with args. assert pure (args.OE.tokens == RT.data tokens /\ args.OE.hidden == RT.data hidden) **
    OR.exported call inputs args w physical prompt permission;
  unfold OR.exported call inputs args w physical prompt permission;
  with p values. assert Shared.safety_caller_postcondition call I64.v (fun x -> x) permission
    (Layout.parameter_values w) (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical)
    (Execution.run w prompt p) (Execution.caller w prompt p p.Execution.count).D.returned
    (OR.output_addresses call args values) (Execution.caller w prompt p p.Execution.count);
  let b:buffers call = {tokens = args.OE.tokens; hidden = args.OE.hidden};
  optimized_output_storage_agrees call args values;
  assert pure (OR.output_addresses call args values == output_storage call b values);
  rewrite each (OR.output_addresses call args values) as (output_storage call b values);
  introduce exists* (p:optimized_observation call) (values:lseq I64.t call.Model.decode_steps).
    Shared.safety_caller_postcondition call I64.v (fun x -> x) permission
      (Layout.parameter_values w) (Layout.parameter_cells inputs) prompt (Layout.prompt_storage inputs physical)
      (Execution.run w prompt p) (Execution.caller w prompt p p.Execution.count).D.returned
      (output_storage call b values) (Execution.caller w prompt p p.Execution.count) **
    pure (optimized_numerical_premise call w prompt p ==>
      Shared.numerical_prefix call (Layout.parameter_values w) prompt
        (Execution.run w prompt p) (Model.calls call)) with p values;
  fold observed_payload Optimized call inputs b w physical prompt permission;
  introduce exists* (b:buffers call).
    pure (b.tokens == RT.data tokens /\ b.hidden == RT.data hidden) **
    observed_payload Optimized call inputs b w physical prompt permission with b;
  fold output_cells Optimized call inputs tokens hidden w physical prompt permission;
}

inline_for_extraction noextract
ghost
fn base_gpu_cells_refine (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (tokens:RT.tensor I64.t) (hidden:RT.tensor bf16) (w:packed)
  (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) ()
  requires on gpu_loc (BQ.result_cells call inputs tokens hidden w physical prompt permission)
  ensures on gpu_loc (output_cells Base call inputs tokens hidden w physical prompt permission)
{
  Pulse.Lib.Core.ghost_impersonate_core #emp_inames gpu_loc
    (BQ.result_cells call inputs tokens hidden w physical prompt permission)
    (output_cells Base call inputs tokens hidden w physical prompt permission)
    (base_cells_refine call inputs tokens hidden w physical prompt permission);
}

inline_for_extraction noextract
ghost
fn optimized_gpu_cells_refine (call:Model.invocation) (inputs:NativeInput.arguments call.Model.prompt_length)
  (tokens:RT.tensor I64.t) (hidden:RT.tensor bf16) (w:packed)
  (physical:lseq I64.t call.Model.prompt_length)
  (prompt:D.vec (D.token 248320) call.Model.prompt_length) (permission:perm) ()
  requires on gpu_loc (OQ.result_cells call inputs tokens hidden w physical prompt permission)
  ensures on gpu_loc (output_cells Optimized call inputs tokens hidden w physical prompt permission)
{
  Pulse.Lib.Core.ghost_impersonate_core #emp_inames gpu_loc
    (OQ.result_cells call inputs tokens hidden w physical prompt permission)
    (output_cells Optimized call inputs tokens hidden w physical prompt permission)
    (optimized_cells_refine call inputs tokens hidden w physical prompt permission);
}

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
{
  input_views_agree call inputs w physical;
  rewrite pre call inputs stream epoch w physical prompt permission
    as BP.pre call inputs stream epoch w physical prompt permission;
  let result = B.program_refines call inputs steps stream #epoch
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal physical))
    #(FStar.Ghost.hide (FStar.Ghost.reveal prompt)) #permission;
  unfold BP.post call inputs stream epoch w physical prompt permission result;
  unfold BT.post call inputs stream epoch w physical prompt permission result;
  match result {
    BT.Failed error -> {
      rewrite Pledge.pledge0 (Epoch.epoch_done stream epoch)
        (BQ.source_input call inputs w physical prompt permission)
        as Pledge.pledge0 (Epoch.epoch_done stream epoch)
          (on gpu_loc (input_cells call inputs w physical prompt permission));
      fold post Base call inputs stream epoch w physical prompt permission (Failed error);
      Failed error
    }
    BT.Returned tokens hidden -> {
      Pledge.rewrite_pledge
        (on gpu_loc (BQ.result_cells call inputs tokens hidden w physical prompt permission))
        (on gpu_loc (output_cells Base call inputs tokens hidden w physical prompt permission))
        (base_gpu_cells_refine call inputs tokens hidden w physical prompt permission);
      fold post Base call inputs stream epoch w physical prompt permission (Returned tokens hidden);
      Returned tokens hidden
    }
  }
}

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
{
  rewrite pre call inputs stream epoch w physical prompt permission
    as OP.pre call inputs stream epoch w physical prompt permission;
  let result = O.program_refines call inputs steps stream #epoch
    #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal physical))
    #(FStar.Ghost.hide (FStar.Ghost.reveal prompt)) #permission;
  unfold OP.post call inputs stream epoch w physical prompt permission result;
  match result {
    OP.Failed error -> {
      fold post Optimized call inputs stream epoch w physical prompt permission (Failed error);
      Failed error
    }
    OP.Returned tokens hidden -> {
      Pledge.rewrite_pledge
        (on gpu_loc (OQ.result_cells call inputs tokens hidden w physical prompt permission))
        (on gpu_loc (output_cells Optimized call inputs tokens hidden w physical prompt permission))
        (optimized_gpu_cells_refine call inputs tokens hidden w physical prompt permission);
      fold post Optimized call inputs stream epoch w physical prompt permission (Returned tokens hidden);
      Returned tokens hidden
    }
  }
}

#pop-options
