module KPlay.Review
#lang-pulse

open Kuiper
open FStar.Real
module Seq = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module C = Kuiper.Chest
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

(* Both wrappers use the optimized canonical packed-weight layout; the native
   base entry also permits other offsets. Base returns residual and delta
   separately, optimized one padded hidden buffer. Seven rows are logical.
   Cache observations are erased snapshots of computed temporaries, not
   persistent physical KV buffers. Optimized hidden correctness is conditional
   on finite live MMA scores of this invocation; its cache guarantee is not. *)
noextract type implementation = | Base | Optimized
inline_for_extraction noextract
let weight_count : nat = 15730944
inline_for_extraction noextract
let hidden_count (i:implementation) : nat = match i with | Base -> 7168 | Optimized -> 8192
inline_for_extraction noextract unfold
let device_array (a:Type0) (n:nat) = p:larray a n{is_global_array p}
inline_for_extraction noextract
let weights_represent (w:lseq bf16 weight_count) (p:Shared.parameters) : prop =
  (forall (c:Shared.index 1024).
    Seq.index w c %~ p.input_layernorm c /\ Seq.index w (1024 + c) %~ p.post_attention_layernorm c) /\
  (forall (c:Shared.index 128).
    Seq.index w (6293504 + c) %~ p.q_norm c /\ Seq.index w (6293632 + c) %~ p.k_norm c) /\
  (forall (r:Shared.index 2048) (c:Shared.index 1024).
    Seq.index w (2048 + r * 1024 + c) %~ p.q_proj r c) /\
  (forall (r c:Shared.index 1024).
    Seq.index w (2048 + (2048 + r) * 1024 + c) %~ p.k_proj r c /\
    Seq.index w (2048 + (3072 + r) * 1024 + c) %~ p.v_proj r c) /\
  (forall (r:Shared.index 1024) (c:Shared.index 2048).
    Seq.index w (4196352 + r * 2048 + c) %~ p.o_proj r c) /\
  (forall (r:Shared.index 3072) (c:Shared.index 1024).
    Seq.index w (6293760 + r * 1024 + c) %~ p.gate_proj r c /\
    Seq.index w (6293760 + (3072 + r) * 1024 + c) %~ p.up_proj r c) /\
  (forall (r:Shared.index 1024) (c:Shared.index 3072).
    Seq.index w (12585216 + r * 3072 + c) %~ p.down_proj r c)

noextract noeq type base_inputs = {
  fused: bool;
  residual: device_array bf16 7168;
  delta: device_array bf16 7168;
  rotary: device_array bf16 1792;
  residual_values: erased (lseq bf16 7168);
  delta_values: erased (lseq bf16 7168);
  rotary_values: erased (lseq bf16 1792);
  residual_model: erased Shared.hidden_states;
  delta_model: erased Shared.hidden_states
}
noextract noeq type optimized_inputs = {
  hidden: device_array bf16 8192;
  rotary: device_array bf16 1048576;
  position: (p:i64{0 <= I64.v p /\ I64.v p <= 4089});
  hidden_values: erased (lseq bf16 8192);
  rotary_values: erased (lseq bf16 1048576);
  hidden_model: erased (Shared.matrix 8 1024)
}
inline_for_extraction noextract unfold
let inputs (i:implementation) : Type0 = match i with | Base -> base_inputs | Optimized -> optimized_inputs
inline_for_extraction noextract
let inputs_own (i:implementation) (hidden_perm rotary_perm:perm) : (inputs i -> slprop) =
  match i with
  | Base -> fun (b:base_inputs) -> on gpu_loc (
    A.pts_to b.residual #hidden_perm b.residual_values ** A.pts_to b.delta #hidden_perm b.delta_values **
    A.pts_to b.rotary #rotary_perm b.rotary_values)
  | Optimized -> fun (b:optimized_inputs) -> on gpu_loc (
    A.pts_to b.hidden #hidden_perm b.hidden_values ** A.pts_to b.rotary #rotary_perm b.rotary_values)
inline_for_extraction noextract
let represented_inputs (i:implementation) (input:Shared.prefill_input) : (inputs i -> prop) =
  match i with
  | Base -> fun (b:base_inputs) ->
    input.hidden == (if b.fused then
      Shared.residual_add (FStar.Ghost.reveal b.residual_model) (FStar.Ghost.reveal b.delta_model)
      else FStar.Ghost.reveal b.residual_model) /\
    (forall (t:Shared.index 7) (c:Shared.index 1024).
      Seq.index b.residual_values (t * 1024 + c) %~ (FStar.Ghost.reveal b.residual_model) t c /\
      (b.fused ==> Seq.index b.delta_values (t * 1024 + c) %~ (FStar.Ghost.reveal b.delta_model) t c)) /\
    (forall (t:Shared.index 7) (c:Shared.index 128).
      Seq.index b.rotary_values (t * 128 + c) %~ input.data.cosine t c /\
      Seq.index b.rotary_values (896 + t * 128 + c) %~ input.data.sine t c)
  | Optimized -> fun (b:optimized_inputs) ->
    input.hidden == (fun (t:Shared.index 7) (c:Shared.index 1024) ->
      (FStar.Ghost.reveal b.hidden_model) t c) /\
    (forall (t:Shared.index 8) (c:Shared.index 1024).
      Seq.index b.hidden_values (t * 1024 + c) %~ (FStar.Ghost.reveal b.hidden_model) t c) /\
    (forall (t:Shared.index 7) (c:Shared.index 128).
      Seq.index b.rotary_values ((I64.v b.position + t) * 128 + c) %~ input.data.cosine t c /\
      Seq.index b.rotary_values (524288 + (I64.v b.position + t) * 128 + c) %~ input.data.sine t c)

inline_for_extraction noextract unfold
let auxiliary (i:implementation) : Type0 =
  match i with | Base -> device_array bf16 7168 | Optimized -> unit
inline_for_extraction noextract unfold
let output_cells (i:implementation) : Type0 =
  match i with | Base -> lseq bf16 7168 & lseq bf16 7168 | Optimized -> lseq bf16 8192
inline_for_extraction noextract
let output_before (i:implementation)
  : (device_array bf16 (hidden_count i) -> auxiliary i -> slprop) =
  match i with
  | Base -> fun out delta -> exists* (r d:lseq bf16 7168).
    on gpu_loc (A.pts_to out r ** A.pts_to delta d)
  | Optimized -> fun out (_:unit) -> on gpu_loc (A.pts_to_uninit out 8192)
inline_for_extraction noextract
let output_owns (i:implementation)
  : (device_array bf16 (hidden_count i) -> auxiliary i -> output_cells i -> slprop) =
  match i with
  | Base -> fun out delta (v:output_cells Base) -> on gpu_loc (A.pts_to out (fst v) ** A.pts_to delta (snd v))
  | Optimized -> fun out (_:unit) (v:output_cells Optimized) -> on gpu_loc (A.pts_to out v)
inline_for_extraction noextract
let hidden_represents (i:implementation) : (output_cells i -> Shared.hidden_states -> prop) =
  match i with
  | Base -> fun (v:output_cells Base) (h:Shared.hidden_states) -> exists (r d:Shared.hidden_states).
    h == Shared.residual_add r d /\
    (forall (t:Shared.index 7) (c:Shared.index 1024).
      Seq.index (fst v) (t * 1024 + c) %~ r t c /\ Seq.index (snd v) (t * 1024 + c) %~ d t c)
  | Optimized -> fun (v:output_cells Optimized) (h:Shared.hidden_states) -> forall (t:Shared.index 7) (c:Shared.index 1024).
    Seq.index v (t * 1024 + c) %~ h t c

inline_for_extraction noextract unfold
let observation (i:implementation) : Type0 =
  match i with | Base -> B.cache_observation | Optimized -> O.cache_observation
inline_for_extraction noextract
let cache_represents (i:implementation)
  : (observation i -> Shared.heads 8 -> Shared.heads 8 -> prop) =
  match i with
  | Base -> fun (o:B.cache_observation) (k v:Shared.heads 8) ->
    forall (t:Shared.index 7) (h:Shared.index 8) (c:Shared.index 128).
      Seq.index o.B.qk_values (t * 3072 + 2048 + h * 128 + c) %~ k t h c /\
      Seq.index o.B.qkv_values (t * 4096 + 3072 + h * 128 + c) %~ v t h c
  | Optimized -> fun (o:O.cache_observation) (k v:Shared.heads 8) ->
    forall (t:Shared.index 7) (h:Shared.index 8) (c:Shared.index 128).
      Seq.index o.O.rotated_values (t * 4096 + h * 512 + 256 + c) %~ k t h c /\
      Seq.index o.O.rotated_values (t * 4096 + h * 512 + 384 + c) %~ v t h c
inline_for_extraction noextract
let expected_keys (input:Shared.prefill_input) : Shared.heads 8 =
  Shared.rotary input.data (Shared.normalize_heads input.weight.k_norm
    (Shared.split_heads #8 (Shared.linear_rows input.weight.k_proj
      (Shared.normalize_hidden input.weight.input_layernorm input.hidden))))
inline_for_extraction noextract
let expected_values (input:Shared.prefill_input) : Shared.heads 8 =
  Shared.split_heads #8 (Shared.linear_rows input.weight.v_proj
    (Shared.normalize_hidden input.weight.input_layernorm input.hidden))
inline_for_extraction noextract
let numerical_premise (i:implementation) : (observation i -> prop) =
  match i with
  | Base -> fun (_:B.cache_observation) -> True
  | Optimized -> fun (o:O.cache_observation) ->
    forall (h:Shared.index 8) (r:Shared.index 14) (c:Shared.index 7{c <= r / 2}).
      Finite? (kind (C.acc2 (O.source_mma_scores o.O.rotated_values h 8) r c))
inline_for_extraction noextract
let padding_after (w:Shared.parameters) (incoming:Shared.vector 1024) : GTot (Shared.vector 1024) =
  let n = Shared.rms_norm w.post_attention_layernorm incoming in
  let gate = Shared.linear w.gate_proj n in
  let up = Shared.linear w.up_proj n in
  let act : Shared.vector 3072 = fun c -> Shared.silu (gate c) *. up c in
  let down = Shared.linear w.down_proj act in
  fun c -> incoming c +. down c
inline_for_extraction noextract
let padding_post (i:implementation) (input:Shared.prefill_input) : (inputs i -> output_cells i -> prop) =
  match i with
  | Base -> fun (_:base_inputs) (_:output_cells Base) -> True
  | Optimized -> fun (b:optimized_inputs) (v:output_cells Optimized) -> forall (c:Shared.index 1024).
    Seq.index v (7168 + c) %~ padding_after input.weight ((FStar.Ghost.reveal b.hidden_model) 7) c
inline_for_extraction noextract
let result (input:Shared.prefill_input) (h:Shared.hidden_states) (k v:Shared.heads 8)
  : Shared.layer_result = {
  Shared.hidden_after = h;
  Shared.cache_after = Shared.update_cache input.cache_before input.layer k v
}
inline_for_extraction noextract
let modeled_post (i:implementation) (input:Shared.prefill_input) (b:inputs i)
  (cells:output_cells i) (o:observation i) (h:Shared.hidden_states) (k v:Shared.heads 8) : prop =
  hidden_represents i cells h /\ cache_represents i o k v /\
  Shared.postcondition input (result input h k v) /\ padding_post i input b cells
inline_for_extraction noextract
let semantic_post (i:implementation) (input:Shared.prefill_input) (b:inputs i)
  (cells:output_cells i) (o:observation i) : prop =
  exists (h:Shared.hidden_states) (k v:Shared.heads 8). modeled_post i input b cells o h k v
inline_for_extraction noextract
let post (i:implementation) (input:Shared.prefill_input) (b:inputs i)
  (cells:output_cells i) (o:observation i) : prop =
  cache_represents i o (expected_keys input) (expected_values input) /\
  (numerical_premise i o ==> semantic_post i input b cells o)

(* The retained linear receipt ties the numerical premise and returned storage
   to this consumed invocation, rather than an unrelated observation. *)
inline_for_extraction noextract
let run_inputs (input:Shared.prefill_input) (w:lseq bf16 weight_count) (b:optimized_inputs)
  : O.invocation_inputs = {
  O.native_hidden = b.hidden_values; O.native_weights = w; O.native_rotary = b.rotary_values;
  O.native_position = I64.v b.position; O.native_layer = input.layer; O.native_before = input.cache_before
}
inline_for_extraction noextract
let receipt (i:implementation) (input:Shared.prefill_input) (w:lseq bf16 weight_count)
  : (b:inputs i -> output_cells i -> observation i -> slprop) =
  match i with
  | Base -> fun (_:base_inputs) (_:output_cells Base) (_:observation Base) -> emp
  | Optimized -> fun (b:optimized_inputs) (cells:output_cells Optimized) (o:observation Optimized) ->
    exists* (run:O.invocation (run_inputs input w b)).
      O.invocation_consumed #(run_inputs input w b) run o **
      O.invocation_result #(run_inputs input w b) run o cells

fn base
  (#input:erased Shared.prefill_input) (weights:device_array bf16 weight_count)
  (b:inputs Base) (out:device_array bf16 (hidden_count Base)) (aux:auxiliary Base)
  (#w:erased (lseq bf16 weight_count)) (#pw #ph #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w) ** inputs_own Base ph pr b
  requires pure (weights_represent w input.Shared.weight /\ represented_inputs Base input b) ** output_before Base out aux
  returns o:erased (observation Base)
  ensures exists* (cells:output_cells Base).
    output_owns Base out aux cells ** receipt Base input w b cells o ** pure (post Base input b cells o)

fn optimized
  (#input:erased Shared.prefill_input) (weights:device_array bf16 weight_count)
  (b:inputs Optimized) (out:device_array bf16 (hidden_count Optimized)) (aux:auxiliary Optimized)
  (#w:erased (lseq bf16 weight_count)) (#pw #ph #pr:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w) ** inputs_own Optimized ph pr b
  requires pure (weights_represent w input.Shared.weight /\ represented_inputs Optimized input b) ** output_before Optimized out aux
  returns o:erased (observation Optimized)
  ensures exists* (cells:output_cells Optimized).
    output_owns Optimized out aux cells ** receipt Optimized input w b cells o ** pure (post Optimized input b cells o)

(*
Signature relation: both wrappers restrict weights to the canonical
15730944-cell packing and the same represented Shared.prefill_input; the
native base entry permits additional offset layouts outside this review.
Base consumes residual/delta plus a compact rotary table and returns two
7168-cell residual/delta arrays. Optimized consumes padded hidden/full
rotary storage and returns one 8192-cell hidden array. represented_inputs
and hidden_represents expose the logical sum, seven active rows and rotary
position mapping; padding_post gives the optimized eighth-row target.

Both return erased observations of computed temporary key/value cells, at
the explicit cache_represents indices, not persistent physical KV
allocations. Their canonical key/value guarantee is unconditional. Base
numerical_premise is True. Optimized semantic_post, including the complete
shared hidden/cache transition and padding, requires finite actual live MMA
scores. Its linear receipt ties cells and observation to this consumed
invocation; no unrelated witness can discharge that premise.

Assessment: high semantic significance for the conditional hidden result,
and moderate ABI/observation differences. Tightness.unique/cache_contents
fix logical hidden and all modeled cache contents, lengths and
initialization, but do not remove the optimized premise or native
approximation freedom. The dispatcher does not join residuals, convert
rotary layouts, manufacture cache ownership or discharge finiteness: it only
invokes the selected wrapper.
*)
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
