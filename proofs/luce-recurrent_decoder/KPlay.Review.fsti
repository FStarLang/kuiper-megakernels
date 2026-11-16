module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module Approx = Kuiper.Approximates.Base
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional

(* Same complete shared decoder/cache transition, with three ABI differences:
   convolution cells are BF16/FP32; recurrent key/value axes are transposed;
   optimized has no input-state allocation at position zero, whereas base
   requires materialized buffers representing the shared initial zero state. *)
noextract type implementation = | Base | Optimized
noextract type position_t = p:i64{0 <= I64.v p /\ I64.v p < 2048}
inline_for_extraction noextract
let weight_count : pos = 21555360
inline_for_extraction noextract
let hidden_count : pos = 1024
inline_for_extraction noextract
let recurrent_count : pos = 262144
inline_for_extraction noextract
let convolution_count : pos = 24576
inline_for_extraction noextract unfold
let convolution_type (i:implementation) : Type0 =
  match i with | Base -> bf16 | Optimized -> f32
inline_for_extraction noextract unfold
let represents_cell (i:implementation) : (convolution_type i -> real -> GTot prop) =
  match i with
  | Base -> fun (x:bf16) r -> Approx.v_approximates x r
  | Optimized -> fun (x:f32) r -> Approx.v_approximates x r

inline_for_extraction noextract unfold
let previous_count (i:implementation) (position:position_t) (count:pos)
  : n:nat{n == 0 \/ n == count} =
  match i with | Base -> count | Optimized -> if I64.v position = 0 then 0 else count
inline_for_extraction noextract
let state_buffer (a:Type0) (n:nat) =
  ptr:array a{n > 0 ==> A.length ptr == n /\ is_global_array ptr}
inline_for_extraction noextract
let state_owns (#a:Type0) (ptr:array a) (fraction:perm) (values:seq a) : slprop =
  if Seq.length values = 0 then emp else on gpu_loc (A.pts_to ptr #fraction values)

inline_for_extraction noextract unfold
let recurrent_index (i:implementation) (head:Shared.fin 16) (key value:Shared.fin 128)
  : Shared.fin recurrent_count =
  match i with
  | Base -> (head * 128 + key) * 128 + value
  | Optimized -> (head * 128 + value) * 128 + key

inline_for_extraction noextract
let hidden_representation (physical:lseq bf16 hidden_count) (logical:Shared.vector hidden_count)
  : GTot prop =
  forall (j:Shared.fin hidden_count). Approx.v_approximates (Seq.index physical j) (logical j)
inline_for_extraction noextract
let recurrent_representation (i:implementation) (physical:lseq f32 recurrent_count)
  (logical:Shared.recurrent_state) : GTot prop =
  forall (head:Shared.fin 16) (key value:Shared.fin 128).
    Approx.v_approximates (Seq.index physical (recurrent_index i head key value))
      (logical head key value)
inline_for_extraction noextract
let convolution_representation (i:implementation)
  (physical:lseq (convolution_type i) convolution_count)
  (logical:Shared.convolution_state) : GTot prop =
  forall (channel:Shared.fin 6144) (tap:Shared.fin 4).
    represents_cell i (Seq.index physical (channel * 4 + tap)) (logical channel tap)

(* The packed-weight layout is common; its agreement with optimized is proved
   in the implementation rather than assumed by the dispatcher. *)
inline_for_extraction noextract
let weights_representation (physical:lseq bf16 weight_count) (logical:Shared.weights)
  : GTot prop =
  B.physical_extents ();
  B.weights_representation physical logical

inline_for_extraction noextract
let input_representation (i:implementation) (position:position_t)
  (call:Shared.invocation) (weights:Shared.weights) (hidden:Shared.vector hidden_count)
  (before:Shared.model_cache{Shared.cache_domain call before})
  (w:lseq bf16 weight_count) (h:lseq bf16 hidden_count)
  (r:lseq f32 (previous_count i position recurrent_count))
  (c:lseq (convolution_type i) (previous_count i position convolution_count)) : GTot prop =
  call.cache_position == I64.v position /\
  weights_representation w weights /\ hidden_representation h hidden /\
  (if previous_count i position recurrent_count = 0 then True
   else recurrent_representation i r (B.old_recurrent call before)) /\
  (if previous_count i position convolution_count = 0 then True
   else convolution_representation i c (B.old_convolution call before))

inline_for_extraction noextract
let output_representation (i:implementation)
  (h:lseq bf16 hidden_count) (r:lseq f32 recurrent_count)
  (c:lseq (convolution_type i) convolution_count) (values:Shared.layer_values) : GTot prop =
  hidden_representation h values.hidden_value /\
  recurrent_representation i r values.recurrent_value /\
  convolution_representation i c values.convolution_value

inline_for_extraction noextract
let call_postcondition (i:implementation)
  (call:Shared.invocation) (weights:Shared.weights) (hidden:Shared.vector hidden_count)
  (before:Shared.model_cache{Shared.cache_domain call before})
  (h:lseq bf16 hidden_count) (r:lseq f32 recurrent_count)
  (c:lseq (convolution_type i) convolution_count) : GTot prop =
  exists (values:Shared.layer_values).
    output_representation i h r c values /\
    Shared.postcondition call weights hidden before
      {Shared.hidden = values.hidden_value;
       Shared.cache = Shared.install_state call.layer_index before values}

inline_for_extraction noextract
let post (i:implementation) (position:position_t)
  (w:lseq bf16 weight_count) (h:lseq bf16 hidden_count)
  (r:lseq f32 (previous_count i position recurrent_count))
  (c:lseq (convolution_type i) (previous_count i position convolution_count))
  (next_h:lseq bf16 hidden_count) (next_r:lseq f32 recurrent_count)
  (next_c:lseq (convolution_type i) convolution_count) : GTot prop =
  forall (call:Shared.invocation) (weights:Shared.weights) (hidden:Shared.vector hidden_count)
    (before:Shared.model_cache{Shared.cache_domain call before}).
    input_representation i position call weights hidden before w h r c ==>
    call_postcondition i call weights hidden before next_h next_r next_c

fn base
  (position:position_t)
  (weights:larray bf16 weight_count{is_global_array weights})
  (hidden:larray bf16 hidden_count{is_global_array hidden})
  (recurrent:state_buffer f32 (previous_count Base position recurrent_count))
  (convolution:state_buffer bf16 (previous_count Base position convolution_count))
  (out_hidden:larray bf16 hidden_count{is_global_array out_hidden})
  (out_recurrent:larray f32 recurrent_count{is_global_array out_recurrent})
  (out_convolution:larray bf16 convolution_count{is_global_array out_convolution})
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 hidden_count))
  (#r:erased (lseq f32 (previous_count Base position recurrent_count)))
  (#c:erased (lseq bf16 (previous_count Base position convolution_count)))
  (#pw #ph #pr #pc:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns recurrent pr r ** state_owns convolution pc c
  requires exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq bf16 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc)
  ensures exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq bf16 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc) **
    pure (post Base position w h r c nh nr nc)

fn optimized
  (position:position_t)
  (weights:larray bf16 weight_count{is_global_array weights})
  (hidden:larray bf16 hidden_count{is_global_array hidden})
  (recurrent:state_buffer f32 (previous_count Optimized position recurrent_count))
  (convolution:state_buffer f32 (previous_count Optimized position convolution_count))
  (out_hidden:larray bf16 hidden_count{is_global_array out_hidden})
  (out_recurrent:larray f32 recurrent_count{is_global_array out_recurrent})
  (out_convolution:larray f32 convolution_count{is_global_array out_convolution})
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 hidden_count))
  (#r:erased (lseq f32 (previous_count Optimized position recurrent_count)))
  (#c:erased (lseq f32 (previous_count Optimized position convolution_count)))
  (#pw #ph #pr #pc:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns recurrent pr r ** state_owns convolution pc c
  requires exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq f32 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc)
  ensures exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq f32 convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc) **
    pure (post Optimized position w h r c nh nr nc)

(*
Signature relation: both roles use the same packed weights, hidden shape,
FP32 recurrent-state shape and complete Shared.postcondition for the
represented layer/cache transition. convolution_type is BF16 for Base and
FP32 for Optimized. recurrent_index transposes the key/value axes between
the physical recurrent buffers; output_representation maps both back to the
same logical head/key/value state. This is a representation theorem, not an
in-place conversion of those arrays.

previous_count Base always requires materialized input-state buffers. At
position zero Optimized instead has empty input-state ownership; the logical
cache domain supplies the initial state. For base's initial call,
input_representation relates its materialized buffers to the shared zero
state. post is quantified over represented logical calls and inputs: it
guarantees call_postcondition for every such representation, not existence
of a representation for arbitrary native cells. Input fractions are
preserved.

Assessment: moderate state-ABI/initialization and numerical significance,
with no change to the complete logical transition. Tightness.unique fixes
hidden, convolution, recurrent and framed KV components for fixed logical
inputs. Native output cells still have qualitative approximation freedom;
this is conditional relative functionality, not bitwise determinism. The
dispatcher neither transposes nor initializes state: it only selects the
wrapper whose types describe those obligations.
*)
fn switcher
  (i:implementation) (position:position_t)
  (weights:larray bf16 weight_count{is_global_array weights})
  (hidden:larray bf16 hidden_count{is_global_array hidden})
  (recurrent:state_buffer f32 (previous_count i position recurrent_count))
  (convolution:state_buffer (convolution_type i) (previous_count i position convolution_count))
  (out_hidden:larray bf16 hidden_count{is_global_array out_hidden})
  (out_recurrent:larray f32 recurrent_count{is_global_array out_recurrent})
  (out_convolution:larray (convolution_type i) convolution_count{is_global_array out_convolution})
  (#w:erased (lseq bf16 weight_count)) (#h:erased (lseq bf16 hidden_count))
  (#r:erased (lseq f32 (previous_count i position recurrent_count)))
  (#c:erased (lseq (convolution_type i) (previous_count i position convolution_count)))
  (#pw #ph #pr #pc:perm)
  preserves cpu ** on gpu_loc (A.pts_to weights #pw w ** A.pts_to hidden #ph h) **
    state_owns recurrent pr r ** state_owns convolution pc c
  requires exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq (convolution_type i) convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc)
  ensures exists* (nh:lseq bf16 hidden_count) (nr:lseq f32 recurrent_count)
    (nc:lseq (convolution_type i) convolution_count).
    on gpu_loc (A.pts_to out_hidden nh ** A.pts_to out_recurrent nr ** A.pts_to out_convolution nc) **
    pure (post i position w h r c nh nr nc)
{
  match i {
    Base -> {
      base position weights hidden recurrent convolution out_hidden out_recurrent out_convolution
        #w #h #r #c #pw #ph #pr #pc
    }
    Optimized -> {
      optimized position weights hidden recurrent convolution out_hidden out_recurrent out_convolution
        #w #h #r #c #pw #ph #pr #pc
    }
  }
}

