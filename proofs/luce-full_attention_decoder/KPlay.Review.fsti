module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional

(* Base uses compact, time-major prefixes and an explicit rotary buffer.
   Optimized uses full, head-major caches and computes rotary values internally.
   Its shared numerical guarantee is conditional on the actual live scores
   being finite. Cache retention and ownership are unconditional for both. *)
noextract type implementation = | Base | Optimized
inline_for_extraction noextract unfold
let weight_count : pos = 18352640
inline_for_extraction noextract unfold
let previous_count (i:implementation) (position:nat) : nat =
  match i with | Base -> position * 512 | Optimized -> if position = 0 then 0 else 1048576
inline_for_extraction noextract unfold
let output_count (i:implementation) (position:nat) : pos =
  match i with | Base -> (position + 1) * 512 | Optimized -> 1048576
inline_for_extraction noextract
let cache_index (i:implementation) (n:pos{n <= 2048})
  (head:Shared.index 2) (time:Shared.index n) (coordinate:Shared.index 256)
  : Shared.index (output_count i (n - 1)) =
  match i with
  | Base -> 512 * time + 256 * head + coordinate
  | Optimized -> head * 524288 + time * 256 + coordinate
inline_for_extraction noextract
let history (i:implementation) (n:pos{n <= 2048})
  (keys values:Shared.vector (output_count i (n - 1)))
  : h:Shared.kv_history{h.length == n} = {
  Shared.length = n;
  Shared.keys = (fun h t d -> keys (cache_index i n h t d));
  Shared.values = (fun h t d -> values (cache_index i n h t d))
}

noextract noeq type logical_inputs (i:implementation) = {
  call: Shared.invocation;
  packed_weights: Shared.vector weight_count;
  input_hidden: Shared.vector 1024;
  old_keys: Shared.vector (previous_count i call.position);
  old_values: Shared.vector (previous_count i call.position);
  frame: Shared.hybrid_cache
}
inline_for_extraction noextract
let before (i:implementation) (input:logical_inputs i)
  : c:Shared.hybrid_cache{Shared.domain input.call c} =
  Shared.full_profile_position_bound input.call;
  {
    Shared.kv = (fun l ->
      if l = input.call.layer then
        (if input.call.position = 0 then None
         else Some (history i input.call.position input.old_keys input.old_values))
      else input.frame.kv l);
    Shared.convolution = input.frame.convolution;
    Shared.recurrent = input.frame.recurrent
  }
inline_for_extraction noextract
let result (i:implementation) (input:logical_inputs i)
  (hidden:Shared.vector 1024)
  (keys values:Shared.vector (output_count i input.call.position)) : Shared.layer_result =
  Shared.full_profile_position_bound input.call;
  {
    Shared.hidden = hidden;
    Shared.cache = Shared.replace_kv input.frame input.call.layer
      (history i (input.call.position + 1) keys values)
  }
inline_for_extraction noextract
let represents (#n:nat) (physical:lseq bf16 n) (logical:Shared.vector n) : prop =
  forall (j:Shared.index n). Seq.index physical j %~ logical j
inline_for_extraction noextract
let represented (i:implementation) (input:logical_inputs i)
  (weights:lseq bf16 weight_count) (hidden:lseq bf16 1024)
  (keys values:lseq bf16 (previous_count i input.call.position)) : prop =
  represents #weight_count weights input.packed_weights /\ represents #1024 hidden input.input_hidden /\
  represents #(previous_count i input.call.position) keys input.old_keys /\
  represents #(previous_count i input.call.position) values input.old_values
inline_for_extraction noextract
let retained (i:implementation) (position:nat)
  (old:lseq bf16 (previous_count i position))
  (next:lseq bf16 (output_count i position)) : prop =
  match i with
  | Base -> forall (j:Shared.index (position * 512)). Seq.index next j == Seq.index old j
  | Optimized -> forall (j:Shared.index 1048576).
      (j % 524288) / 256 <> position ==>
      Seq.index next j == (if position = 0 then (zero <: bf16) else Seq.index old j)

inline_for_extraction noextract unfold
let score_observation (i:implementation) (position:nat) : Type0 =
  match i with | Base -> unit | Optimized -> lseq f32 (8 * (position + 1))
inline_for_extraction noextract
let numerical_premise (i:implementation) (position:nat)
  : (score_observation i position -> prop) =
  match i with
  | Base -> fun (_:unit) -> True
  | Optimized -> fun (scores:lseq f32 (8 * (position + 1))) ->
      forall (j:Shared.index (8 * (position + 1))). Finite? (kind (Seq.index scores j))
inline_for_extraction noextract
let semantic_post (i:implementation) (input:logical_inputs i)
  (hidden:lseq bf16 1024) (keys values:lseq bf16 (output_count i input.call.position)) : prop =
  exists (h:Shared.vector 1024)
    (k v:Shared.vector (output_count i input.call.position)).
    represents #1024 hidden h /\
    represents #(output_count i input.call.position) keys k /\
    represents #(output_count i input.call.position) values v /\
    Shared.postcondition input.call (B.weights_view input.packed_weights) input.input_hidden
      (before i input) (result i input h k v)
inline_for_extraction noextract
let post (i:implementation) (input:logical_inputs i)
  (old_keys old_values:lseq bf16 (previous_count i input.call.position))
  (hidden:lseq bf16 1024) (keys values:lseq bf16 (output_count i input.call.position))
  (scores:score_observation i input.call.position) : prop =
  retained i input.call.position old_keys keys /\
  retained i input.call.position old_values values /\
  (numerical_premise i input.call.position scores ==> semantic_post i input hidden keys values)

inline_for_extraction noextract unfold
let device_array (a:Type0) (n:nat) = p:larray a n{is_global_array p}
inline_for_extraction noextract
let state_buffer (i:implementation) (position:nat) =
  p:array bf16{
    if position = 0 then p == A.null
    else is_global_array p /\ A.length p == previous_count i position}
inline_for_extraction noextract
let state_owns (i:implementation) (position:nat)
  (ptr:state_buffer i position) (fraction:perm)
  (values:lseq bf16 (previous_count i position)) : slprop =
  if position = 0 then emp else on gpu_loc (A.pts_to ptr #fraction values)
noextract noeq type rotary_buffer = {
  pointer: device_array bf16 64;
  values: erased (lseq bf16 64)
}
inline_for_extraction noextract unfold
let auxiliary (i:implementation) : Type0 = match i with | Base -> rotary_buffer | Optimized -> unit
inline_for_extraction noextract
let auxiliary_owns (i:implementation) (fraction:perm) : (auxiliary i -> slprop) =
  match i with
  | Base -> fun (r:rotary_buffer) -> on gpu_loc (A.pts_to r.pointer #fraction r.values)
  | Optimized -> fun (_:unit) -> emp
inline_for_extraction noextract
let auxiliary_represents (i:implementation) (position:nat) : (auxiliary i -> prop) =
  match i with
  | Base -> fun (r:rotary_buffer) ->
      (forall (j:Shared.index 32). Seq.index r.values j %~ Shared.rope_cos position j) /\
      (forall (j:Shared.index 32). Seq.index r.values (32 + j) %~ Shared.rope_sin position j)
  | Optimized -> fun (_:unit) -> True

fn base
  (#input:erased (logical_inputs Base)) (position:i64{I64.v position == input.call.position})
  (weights:device_array bf16 weight_count) (hidden:device_array bf16 1024)
  (old_keys old_values:state_buffer Base input.call.position)
  (aux:auxiliary Base)
  (out_hidden:device_array bf16 1024)
  (new_keys new_values:device_array bf16 (output_count Base input.call.position))
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 1024))
  (#k #v:erased (lseq bf16 (previous_count Base input.call.position)))
  (#pw #ph #pk #pv #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns Base input.call.position old_keys pk k **
    state_owns Base input.call.position old_values pv v ** auxiliary_owns Base pr aux
  requires pure (represented Base input w h k v /\ auxiliary_represents Base input.call.position aux) **
    on gpu_loc (A.pts_to_uninit out_hidden 1024 **
      A.pts_to_uninit new_keys (output_count Base input.call.position) **
      A.pts_to_uninit new_values (output_count Base input.call.position))
  returns scores:erased (score_observation Base input.call.position)
  ensures exists* (nh:lseq bf16 1024) (nk nv:lseq bf16 (output_count Base input.call.position)).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to new_keys nk ** A.pts_to new_values nv) **
    pure (post Base input k v nh nk nv scores)

fn optimized
  (#input:erased (logical_inputs Optimized)) (position:i64{I64.v position == input.call.position})
  (weights:device_array bf16 weight_count) (hidden:device_array bf16 1024)
  (old_keys old_values:state_buffer Optimized input.call.position)
  (aux:auxiliary Optimized)
  (out_hidden:device_array bf16 1024)
  (new_keys new_values:device_array bf16 (output_count Optimized input.call.position))
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 1024))
  (#k #v:erased (lseq bf16 (previous_count Optimized input.call.position)))
  (#pw #ph #pk #pv #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns Optimized input.call.position old_keys pk k **
    state_owns Optimized input.call.position old_values pv v ** auxiliary_owns Optimized pr aux
  requires pure (represented Optimized input w h k v /\ auxiliary_represents Optimized input.call.position aux) **
    on gpu_loc (A.pts_to_uninit out_hidden 1024 **
      A.pts_to_uninit new_keys (output_count Optimized input.call.position) **
      A.pts_to_uninit new_values (output_count Optimized input.call.position))
  returns scores:erased (score_observation Optimized input.call.position)
  ensures exists* (nh:lseq bf16 1024) (nk nv:lseq bf16 (output_count Optimized input.call.position)).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to new_keys nk ** A.pts_to new_values nv) **
    pure (post Optimized input k v nh nk nv scores)

(*
Signature relation: Base uses compact time-major KV prefixes of position*512
cells and returns (position+1)*512 cells; Optimized uses full head-major
1048576-cell arrays after initialization. cache_index and history expose the
common logical heads/time/coordinate view. Initial source buffers are null
for both. Base additionally borrows a represented 64-cell rotary buffer;
optimized computes rotary values internally and takes unit. retained
specifies unchanged old cells, including optimized zero-filled noncurrent
positions at initialization.

The important semantic asymmetry is numerical_premise. Base returns an
erased unit observation and the premise is True. Optimized returns its
actual FP32 live-score observation and guarantees semantic_post only when
all 8*(position+1) scores are finite. Ownership and retained are
unconditional; the dispatcher neither assumes finiteness nor substitutes a
different score witness. Under the premise, both represented outputs satisfy
the complete shared hidden/cache transition and preserve other cache
components.

Assessment: high contract significance because dropping finiteness loses the
optimized functional conclusion; layouts alone have moderate ABI impact.
Tightness.unique determines the logical hidden/cache result, but does not
make the conditional optimized guarantee unconditional or determine native
floating bits. The dispatch body merely forwards role-specific buffers,
permissions and the same invocation to one wrapper.
*)
fn switcher
  (i:implementation) (#input:erased (logical_inputs i))
  (position:i64{I64.v position == input.call.position})
  (weights:device_array bf16 weight_count) (hidden:device_array bf16 1024)
  (old_keys old_values:state_buffer i input.call.position) (aux:auxiliary i)
  (out_hidden:device_array bf16 1024)
  (new_keys new_values:device_array bf16 (output_count i input.call.position))
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 1024))
  (#k #v:erased (lseq bf16 (previous_count i input.call.position)))
  (#pw #ph #pk #pv #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns i input.call.position old_keys pk k **
    state_owns i input.call.position old_values pv v ** auxiliary_owns i pr aux
  requires pure (represented i input w h k v /\ auxiliary_represents i input.call.position aux) **
    on gpu_loc (A.pts_to_uninit out_hidden 1024 **
      A.pts_to_uninit new_keys (output_count i input.call.position) **
      A.pts_to_uninit new_values (output_count i input.call.position))
  returns scores:erased (score_observation i input.call.position)
  ensures exists* (nh:lseq bf16 1024) (nk nv:lseq bf16 (output_count i input.call.position)).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to new_keys nk ** A.pts_to new_values nv) **
    pure (post i input k v nh nk nv scores)
{
  match i {
    Base -> {
      base #input position weights hidden old_keys old_values aux out_hidden new_keys new_values
        #w #h #k #v #pw #ph #pk #pv #pr
    }
    Optimized -> {
      optimized #input position weights hidden old_keys old_values aux out_hidden new_keys new_values
        #w #h #k #v #pw #ph #pk #pv #pr
    }
  }
}

