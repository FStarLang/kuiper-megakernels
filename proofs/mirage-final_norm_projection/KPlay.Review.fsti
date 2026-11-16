module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional

(* Both return seven normalized BF16 rows and vocabulary scores. Base sums
   residual+delta and reads packed weights; optimized consumes a prepared
   hidden input and separate weights. Its eighth input row is unused here,
   and its extra score columns represent zero, not additional vocabulary. *)
noextract type implementation = | Base | Optimized
inline_for_extraction noextract unfold
let width : pos = 1024
inline_for_extraction noextract unfold
let rows : pos = 7
inline_for_extraction noextract unfold
let vocabulary : pos = 151936
inline_for_extraction noextract unfold
let stride (i:implementation) : pos =
  match i with | Base -> vocabulary | Optimized -> 153600
inline_for_extraction noextract unfold
let device_array (a:Type0) (n:nat) = p:larray a n{is_global_array p}

noextract noeq type base_inputs = {
  extent: B.packed_extent;
  norm_offset: B.weight_offset (U64.v extent) width;
  projection_offset: B.weight_offset (U64.v extent) (vocabulary * width);
  packed: device_array bf16 (U64.v extent);
  residual: device_array bf16 (rows * width);
  delta: device_array bf16 (rows * width);
  packed_values: erased (lseq bf16 (U64.v extent));
  residual_values: erased (lseq bf16 (rows * width));
  delta_values: erased (lseq bf16 (rows * width));
  packed_model: erased (j:nat{j < U64.v extent} -> GTot real);
  residual_model: erased Shared.hidden_rows;
  delta_model: erased Shared.hidden_rows
}
noextract noeq type optimized_inputs = {
  hidden: device_array bf16 (8 * width);
  gamma: device_array bf16 width;
  embedding: device_array bf16 (vocabulary * width);
  hidden_values: erased (lseq bf16 (8 * width));
  gamma_values: erased (lseq bf16 width);
  embedding_values: erased (lseq bf16 (vocabulary * width));
  hidden_model: erased (lseq real (8 * width));
  gamma_model: erased (lseq real width);
  embedding_model: erased (lseq real (vocabulary * width))
}
inline_for_extraction noextract unfold
let inputs (i:implementation) : Type0 =
  match i with | Base -> base_inputs | Optimized -> optimized_inputs

inline_for_extraction noextract
let logical_input (i:implementation) : (inputs i -> GTot Shared.inputs) =
  match i with
  | Base -> fun (b:base_inputs) -> {
      Shared.hidden_in = (fun p d ->
        (FStar.Ghost.reveal b.residual_model) p d +. (FStar.Ghost.reveal b.delta_model) p d);
      Shared.norm_weight = (fun d -> (FStar.Ghost.reveal b.packed_model) (I64.v b.norm_offset + d));
      Shared.embedding = (fun v d ->
        (FStar.Ghost.reveal b.packed_model) (I64.v b.projection_offset + v * width + d))
    }
  | Optimized -> fun (b:optimized_inputs) -> {
      Shared.hidden_in = (fun p d -> Seq.index b.hidden_model (p * width + d));
      Shared.norm_weight = (fun d -> Seq.index b.gamma_model d);
      Shared.embedding = (fun v d -> Seq.index b.embedding_model (v * width + d))
    }
inline_for_extraction noextract
let represented (i:implementation) : (inputs i -> GTot prop) =
  match i with
  | Base -> fun (b:base_inputs) ->
      (forall (j:nat{j < U64.v b.extent}).
        Seq.index b.packed_values j %~ (FStar.Ghost.reveal b.packed_model) j) /\
      (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
        Seq.index b.residual_values (p * width + d) %~ (FStar.Ghost.reveal b.residual_model) p d /\
        Seq.index b.delta_values (p * width + d) %~ (FStar.Ghost.reveal b.delta_model) p d)
  | Optimized -> fun (b:optimized_inputs) ->
      b.hidden_values %~ b.hidden_model /\ b.gamma_values %~ b.gamma_model /\
      b.embedding_values %~ b.embedding_model
inline_for_extraction noextract
let input_ownership (i:implementation) (p0 p1 p2:perm) : (inputs i -> slprop) =
  match i with
  | Base -> fun (b:base_inputs) -> on gpu_loc (
      A.pts_to b.packed #p0 b.packed_values **
      A.pts_to b.residual #p1 b.residual_values ** A.pts_to b.delta #p2 b.delta_values)
  | Optimized -> fun (b:optimized_inputs) -> on gpu_loc (
      A.pts_to b.hidden #p0 b.hidden_values **
      A.pts_to b.gamma #p1 b.gamma_values ** A.pts_to b.embedding #p2 b.embedding_values)

noextract noeq type observation = {
  normalized: Shared.hidden_rows;
  logits: Shared.prompt_index -> GTot Shared.vocabulary_logits
}
inline_for_extraction noextract
let observation_post (i:implementation) (x:Shared.inputs)
  (hidden:lseq bf16 (rows * width)) (scores:lseq bf16 (rows * stride i))
  (observed:observation) : GTot prop =
    (forall (p:Shared.prompt_index) (d:Shared.hidden_index).
      Seq.index hidden (p * width + d) %~ observed.normalized p d) /\
    (forall (p:Shared.prompt_index) (v:Shared.vocab_index).
      Seq.index scores (p * stride i + v) %~ observed.logits p v) /\
    Shared.postcondition x ()
      {Shared.normalized_hidden = observed.normalized; Shared.final_logits = observed.logits 6} () /\
    (forall (p:Shared.prompt_index) (v:Shared.vocab_index).
      observed.logits p v == Shared.tied_projection x.embedding (observed.normalized p) v)
inline_for_extraction noextract
let post (i:implementation) (x:Shared.inputs)
  (hidden:lseq bf16 (rows * width)) (scores:lseq bf16 (rows * stride i)) : GTot prop =
  (exists (observed:observation). observation_post i x hidden scores observed) /\
  (forall (p:Shared.prompt_index) (c:nat{vocabulary <= c /\ c < stride i}).
    Seq.index scores (p * stride i + c) %~ 0.0R)

fn base
  (b:inputs Base) (#x:erased Shared.inputs) (#p0 #p1 #p2:perm)
  (hidden:device_array bf16 (rows * width))
  (scores:device_array bf16 (rows * stride Base))
  preserves cpu ** input_ownership Base p0 p1 p2 b
  requires pure (represented Base b /\ logical_input Base b == x) **
    (exists* (h:lseq bf16 (rows * width)) (s:lseq bf16 (rows * stride Base)).
      on gpu_loc (A.pts_to hidden h ** A.pts_to scores s))
  ensures exists* (h:lseq bf16 (rows * width)) (s:lseq bf16 (rows * stride Base)).
    on gpu_loc (A.pts_to hidden h ** A.pts_to scores s) ** pure (post Base x h s)

fn optimized
  (b:inputs Optimized) (#x:erased Shared.inputs) (#p0 #p1 #p2:perm)
  (hidden:device_array bf16 (rows * width))
  (scores:device_array bf16 (rows * stride Optimized))
  preserves cpu ** input_ownership Optimized p0 p1 p2 b
  requires pure (represented Optimized b /\ logical_input Optimized b == x) **
    (exists* (h:lseq bf16 (rows * width)) (s:lseq bf16 (rows * stride Optimized)).
      on gpu_loc (A.pts_to hidden h ** A.pts_to scores s))
  ensures exists* (h:lseq bf16 (rows * width)) (s:lseq bf16 (rows * stride Optimized)).
    on gpu_loc (A.pts_to hidden h ** A.pts_to scores s) ** pure (post Optimized x h s)

(*
Signature relation: inputs Base carries packed weights/offsets and two
seven-row residual/delta buffers; inputs Optimized carries a prepared eight-
row hidden buffer and separate gamma/embedding weights. represented and
logical_input i b == x require these different physical inputs to describe
the SAME logical hidden sum and weights. Both return seven normalized BF16
rows. Score stride is 151936 for Base and 153600 for Optimized, with
explicit fractional ownership preservation for each input.

observation_post relates actual returned cells to Shared.postcondition and
additionally fixes projection of ALL seven normalized rows, not only the
last-row logits used for selection. Optimized padded score columns represent
zero; that is an approximation target, not a promise of identical zero bit
patterns. Its eighth incoming hidden row has no role in the seven-row
logical task. The dispatcher does not compute residual addition, unpack
weights, or prepare the optimized input.

Assessment: moderate ABI/input-preparation significance; no different
normalization/projection task once the logical-input relation holds.
Tightness.unique fixes shared normalized rows, final logits and framed
state; observation_post determines the other row targets too. Native BF16
cells retain approximation freedom, so this is relative functional
correctness, not bitwise equality or comparison preservation. Each visible
branch does exactly one wrapper call.
*)
fn switcher
  (i:implementation) (b:inputs i) (#x:erased Shared.inputs) (#p0 #p1 #p2:perm)
  (hidden:device_array bf16 (rows * width))
  (scores:device_array bf16 (rows * stride i))
  preserves cpu ** input_ownership i p0 p1 p2 b
  requires pure (represented i b /\ logical_input i b == x) **
    (exists* (h:lseq bf16 (rows * width)) (s:lseq bf16 (rows * stride i)).
      on gpu_loc (A.pts_to hidden h ** A.pts_to scores s))
  ensures exists* (h:lseq bf16 (rows * width)) (s:lseq bf16 (rows * stride i)).
    on gpu_loc (A.pts_to hidden h ** A.pts_to scores s) ** pure (post i x h s)
{
  match i {
    Base -> { base b #x #p0 #p1 #p2 hidden scores }
    Optimized -> { optimized b #x #p0 #p1 #p2 hidden scores }
  }
}

