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
module Base = KPlay.Base.Functional
module Optimized = KPlay.Optimized.Functional
module BaseNative = KPlay.Proof.BaseNative
module Binding = KPlay.Proof.NativeBindings
module Layout = KPlay.Proof.SourceLayouts
module Program = KPlay.Proof.SourceProgram
module Capacity = KPlay.Proof.SourceCapacity

#push-options "--z3rlimit 10"

(* Review boundary: both entry points below call their checked implementations.
   Their contracts differ only in completion epoch (one launch versus three).
   Allocation/return failures, input preservation, ownership and frame recovery
   are retained. Output cells satisfy the SAME relation, not bitwise equality:
   Shared.task_relation existentially hides each execution's certified native
   projection/softmax and normalization observations. No equal-row premise is
   imposed on callers, and no cross-execution numerical equality is claimed. *)

inline_for_extraction noextract
let input_models_agree (xs:token_values) (ws:weight_values)
  : Lemma (
      Base.token_model xs == Layout.logical_tokens xs /\
      Base.weight_model ws == Layout.logical_weights ws)
  = assert_norm (
      Base.token_model xs == Layout.logical_tokens xs /\
      Base.weight_model ws == Layout.logical_weights ws)

(* Exact relation between the review postconditions. Allocation failure does
   not advance either epoch. After a launch, aligning completion epochs makes
   the entire postconditions equal, including return-construction failure.
   This compares predicates on the SAME result, not results of two runs. *)
inline_for_extraction noextract
let allocation_failure_contracts_agree
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument bf16)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  (error:RT.allocation_error)
  : Lemma (
      after_call BaseRole stream epoch tokens weights xs ws fp fw x w frame
        (AllocationFailed error) ==
      after_call OptimizedRole stream epoch tokens weights xs ws fp fw x w frame
        (AllocationFailed error))
  = ()

inline_for_extraction noextract
let returned_contracts_agree
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument bf16)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  (result:Results.construction_result i32 i64 f32 i32)
  : Lemma (
      after_call BaseRole stream (Epoch.epoch_next (Epoch.epoch_next epoch))
        tokens weights xs ws fp fw x w frame (Returned result) ==
      after_call OptimizedRole stream epoch
        tokens weights xs ws fp fw x w frame (Returned result))
  = ()

inline_for_extraction noextract unfold
let base_outcome (result:Base.source_result) : outcome =
  match result with
  | Base.Allocation_failed error -> AllocationFailed error
  | Base.Native_return result -> Returned result

inline_for_extraction noextract unfold
let optimized_outcome (result:Program.outcome) : outcome =
  match result with
  | Program.AllocationFailed error -> AllocationFailed error
  | Program.Returned result -> Returned result

(* These adapters are proofs over the original contracts, not new axioms. *)
inline_for_extraction noextract
ghost
fn pack_output
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (value:vector)
  (cells:Native.output_values) (out:Shared.public_outputs) ()
  requires Native.output_cells (handles value) cells **
    pure (Native.output_binding (handles value) cells out /\
      Shared.task_relation x w x w out)
  ensures output x w value
{
  fold output_payload x w value cells out;
  Pulse.Lib.Core.intro_exists (fun out -> output_payload x w value cells out) out;
  Pulse.Lib.Core.intro_exists
    (fun cells -> exists* out. output_payload x w value cells out) cells;
  fold output x w value;
}

inline_for_extraction noextract
ghost
fn base_output_refines
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (value:vector)
  (inputs frame:slprop) ()
  requires (inputs ** Base.program_output x w value) ** frame
  ensures (inputs ** output x w value) ** frame
{
  unfold Base.program_output x w value;
  with cells. assert Native.output_cells (handles value) cells;
  pack_output x w value cells (BaseNative.public_view cells) ();
}

inline_for_extraction noextract
ghost
fn optimized_output_refines
  (xs:token_values) (ws:weight_values)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (value:vector)
  (inputs frame:slprop)
  (models:squash (Layout.logical_tokens xs == x /\ Layout.logical_weights ws == w)) ()
  requires Binding.output_post (handles value) xs ws ** (inputs ** frame)
  ensures (inputs ** output x w value) ** frame
{
  unfold Binding.output_post (handles value) xs ws;
  with state. assert Binding.output_payload (handles value) xs ws state;
  unfold Binding.output_payload (handles value) xs ws state;
  pack_output x w value (Binding.output_values state) (Capacity.public_output state) ();
}

inline_for_extraction noextract
ghost
fn base_contract_refines
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens weights:Input.argument bf16)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  (result:Base.source_result) ()
  requires Base.program_post stream epoch tokens weights xs ws fp fw x w frame result
  ensures after_call BaseRole stream epoch tokens weights xs ws fp fw x w frame
    (base_outcome result)
{
  unfold Base.program_post stream epoch tokens weights xs ws fp fw x w frame result;
  match result {
    Base.Allocation_failed _ -> {
      fold after_call BaseRole stream epoch tokens weights xs ws fp fw x w frame
        (base_outcome result);
    }
    Base.Native_return returned -> {
      match returned {
        Results.Failed _ -> {
          fold after_call BaseRole stream epoch tokens weights xs ws fp fw x w frame
            (base_outcome result);
        }
        Results.Constructed value -> {
          Pulse.Lib.Core.star_assoc (Native.input_cells tokens weights fp fw xs ws)
            (Base.program_output x w value) frame;
          Pulse.Lib.Core.star_assoc (Native.input_cells tokens weights fp fw xs ws)
            (output x w value) frame;
          Pledge.rewrite_pledge
            ((Native.input_cells tokens weights fp fw xs ws **
              Base.program_output x w value) ** frame)
            ((Native.input_cells tokens weights fp fw xs ws **
              output x w value) ** frame)
            (base_output_refines x w value
              (Native.input_cells tokens weights fp fw xs ws) frame);
          fold after_call BaseRole stream epoch tokens weights xs ws fp fw x w frame
            (base_outcome result);
        }
      }
    }
  }
}

inline_for_extraction noextract
ghost
fn optimized_contract_refines
  (stream:Stream.stream_t) (epoch:Epoch.epoch_t)
  (tokens:Binding.token_argument) (weights:Binding.weight_argument)
  (xs:token_values) (ws:weight_values) (fp fw:perm)
  (x:Shared.token_matrix) (w:Shared.weight_matrix) (frame:slprop)
  (models:squash (Layout.logical_tokens xs == x /\ Layout.logical_weights ws == w))
  (result:Program.outcome) ()
  requires cpu ** Input.reference tokens ** Input.reference weights **
    Program.after_call stream epoch tokens weights xs ws fp fw frame result
  ensures after_call OptimizedRole stream epoch tokens weights xs ws fp fw x w frame
    (optimized_outcome result)
{
  unfold Program.after_call stream epoch tokens weights xs ws fp fw frame result;
  match result {
    Program.AllocationFailed _ -> {
      fold after_call OptimizedRole stream epoch tokens weights xs ws fp fw x w frame
        (optimized_outcome result);
    }
    Program.Returned returned -> {
      match returned {
        Results.Failed _ -> {
          fold after_call OptimizedRole stream epoch tokens weights xs ws fp fw x w frame
            (optimized_outcome result);
        }
        Results.Constructed value -> {
          Pulse.Lib.Core.star_assoc (Native.input_cells tokens weights fp fw xs ws)
            (output x w value) frame;
          Pledge.rewrite_pledge
            (Binding.output_post (handles value) xs ws **
              (Native.input_cells tokens weights fp fw xs ws ** frame))
            ((Native.input_cells tokens weights fp fw xs ws **
              output x w value) ** frame)
            (optimized_output_refines xs ws x w value
              (Native.input_cells tokens weights fp fw xs ws) frame models);
          fold after_call OptimizedRole stream epoch tokens weights xs ws fp fw x w frame
            (optimized_outcome result);
        }
      }
    }
  }
}

(* The two audit-facing signatures: identical except BaseRole/OptimizedRole.
   Optimized acquires its own parameter references internally; the caller's
   references are framed, so both wrappers preserve the same caller resources. *)
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
{
  unfold before_call stream epoch tokens weights xs ws fp fw x w frame;
  let result = Base.program_refines tokens weights stream #epoch #fp #fw #xs #ws #x #w frame ();
  base_contract_refines stream epoch tokens weights xs ws fp fw x w frame result ();
  base_outcome result
}

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
{
  unfold before_call stream epoch tokens weights xs ws fp fw x w frame;
  Base.input_models_are_bound tokens weights xs ws x w;
  input_models_agree xs ws;
  assert pure (Layout.logical_tokens xs == x /\ Layout.logical_weights ws == w);
  unfold Native.call_resources stream epoch tokens weights fp fw xs ws frame;
  fold Program.before_call stream epoch tokens weights xs ws fp fw frame;
  let result = Optimized.program_refines stream #epoch tokens weights #xs #ws #fp #fw frame ();
  optimized_contract_refines stream epoch tokens weights xs ws fp fw x w frame () result ();
  optimized_outcome result
}

#pop-options
