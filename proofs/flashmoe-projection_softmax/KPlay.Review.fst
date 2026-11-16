module KPlay.Review
#lang-pulse

open Kuiper
open Pulse.Lib.Send
module Seq = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module C = Kuiper.Chest
module G = Kuiper.Spec.GEMM
module SM = Kuiper.Spec.Softmax
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let base_projection_agrees (#h #e:pos) (x:lseq bf16 h) (w:lseq bf16 (e * h))
  : Lemma (
      B.projection_row x w ==
      C.chest2_row (model_projection
        (C.to_real_chest (row_major 1 h x))
        (C.to_real_chest (row_major e h w))) 0) =
  B.chest2_extensional (B.token_view x) (row_major 1 h x);
  O.weight_model_transpose h e w;
  assert_norm (O.weight_view h e w == B.weight_view w);
  assert_norm (model_projection (C.to_real_chest (row_major 1 h x))
    (C.to_real_chest (row_major e h w)) ==
    G.matmul (C.to_real_chest (row_major 1 h x))
      (C.mk2 #real #h #e (fun k j -> C.acc2 (C.to_real_chest (row_major e h w)) j k)));
  let cell (j:natlt e)
    : Lemma (
        C.acc1 (B.projection_row x w) j ==
        C.acc2 (model_projection (C.to_real_chest (row_major 1 h x))
          (C.to_real_chest (row_major e h w))) 0 j) =
    G.lemma_matmul_index (C.to_real_chest (row_major 1 h x))
      (C.mk2 #real #h #e (fun k j -> C.acc2 (C.to_real_chest (row_major e h w)) j k)) 0 j;
    G.__gmatmul_single_congr 0.0R ( *. ) ( +. )
      (C.to_real_chest (B.token_view x)) (C.to_real_chest (B.weight_view w))
      (C.to_real_chest (row_major 1 h x))
      (C.mk2 #real #h #e (fun k j -> C.acc2 (C.to_real_chest (row_major e h w)) j k))
      0 j 0 j h
  in
  FStar.Classical.forall_intro cell;
  B.chest1_extensional (B.projection_row x w)
    (C.chest2_row (model_projection
      (C.to_real_chest (row_major 1 h x))
      (C.to_real_chest (row_major e h w))) 0)

inline_for_extraction noextract
let target_entry (m h:nat) (e:pos)
  (x:lseq bf16 (m * h)) (w:lseq bf16 (e * h)) (t:natlt m) (j:natlt e)
  : Lemma (
      C.acc2 (target m h e x w) t j ==
      C.acc1 (SM.softmax_real (C.chest2_row
        (model_projection (C.to_real_chest (row_major m h x))
          (C.to_real_chest (row_major e h w))) t)) j) =
  assert_norm (
    C.acc2 (target m h e x w) t j ==
    C.acc1 (SM.softmax_real (C.chest2_row
      (model_projection (C.to_real_chest (row_major m h x))
        (C.to_real_chest (row_major e h w))) t)) j)

inline_for_extraction noextract
let base_post_refines (smoke:bool)
  (x:lseq bf16 (Shared.hidden_count smoke))
  (w:lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (p:lseq f32 (Shared.expert_count smoke))
  : Lemma
      (requires C.seq_to_chest1 p %~ SM.softmax_real (B.projection_row x w))
      (ensures post Base smoke x w p) =
  base_projection_agrees x w;
  let cell (t:natlt 1) (j:natlt (Shared.expert_count smoke))
    : Lemma (Seq.index p j %~
        C.acc2 (target 1 (Shared.hidden_count smoke) (Shared.expert_count smoke) x w) t j) =
    assert (t == 0);
    assert (Seq.index p j %~ C.acc1 (SM.softmax_real (B.projection_row x w)) j);
    target_entry 1 (Shared.hidden_count smoke) (Shared.expert_count smoke) x w t j
  in
  FStar.Classical.forall_intro_2 cell;
  assert (task 1 (Shared.hidden_count smoke) (Shared.expert_count smoke) x w p);
  ()

inline_for_extraction noextract
let optimized_domain (smoke:bool)
  (x:lseq bf16 (Shared.token_count smoke * Shared.hidden_count smoke))
  (w:lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  : Lemma
      (requires domain (Shared.token_count smoke) (Shared.hidden_count smoke)
        (Shared.expert_count smoke) x w)
      (ensures O.input_domain (Shared.token_count smoke) (Shared.hidden_count smoke)
        (Shared.expert_count smoke) x w) =
  if smoke then () else ()

inline_for_extraction noextract
let views_agree (#a:Type0) (m n:nat) (x:lseq a (m * n))
  : Lemma (row_major m n x == O.row_major_view m n x) =
  assert (row_major m n x == O.row_major_view m n x) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Review.row_major"; "KPlay.Optimized.Functional.row_major_view"]];
    FStar.Tactics.V2.trefl ())
inline_for_extraction noextract
let models_agree (#m #h:nat) (#e:pos)
  (x:C.chest2 real m h) (w:C.chest2 real e h)
  : Lemma (model_target x w == O.projection_softmax_real x w) =
  assert (model_target x w == O.projection_softmax_real x w) by (
    FStar.Tactics.V2.norm [zeta_full; iota; primops; delta_only [
      "KPlay.Review.model_target"; "KPlay.Review.model_projection";
      "KPlay.Optimized.Functional.projection_softmax_real";
      "KPlay.Optimized.Functional.projection_real"]];
    FStar.Tactics.V2.trefl ())
inline_for_extraction noextract
let optimized_post_refines (smoke:bool)
  (x:lseq bf16 (Shared.token_count smoke * Shared.hidden_count smoke))
  (w:lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (p:lseq f32 (Shared.token_count smoke * Shared.expert_count smoke))
  (modeled:Shared.probability_matrix smoke)
  : Lemma
      (requires O.semantic_post (Shared.token_count smoke) (Shared.hidden_count smoke)
        (Shared.expert_count smoke) x x w w p modeled)
      (ensures post Optimized smoke x w p) =
  views_agree (Shared.token_count smoke) (Shared.hidden_count smoke) x;
  views_agree (Shared.expert_count smoke) (Shared.hidden_count smoke) w;
  views_agree (Shared.token_count smoke) (Shared.expert_count smoke) p;
  models_agree (C.to_real_chest (row_major (Shared.token_count smoke) (Shared.hidden_count smoke) x))
    (C.to_real_chest (row_major (Shared.expert_count smoke) (Shared.hidden_count smoke) w));
  let all (rx:C.chest2 real (Shared.token_count smoke) (Shared.hidden_count smoke))
    (rw:C.chest2 real (Shared.expert_count smoke) (Shared.hidden_count smoke))
    : Lemma (model_target rx rw == O.projection_softmax_real rx rw) =
    models_agree rx rw
  in
  FStar.Classical.forall_intro_2 all;
  assert (task (Shared.token_count smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) x w p);
  ()

inline_for_extraction noextract
let packed_projection_agrees (smoke:bool)
  (x:Shared.token_matrix smoke) (w:Shared.weight_matrix smoke)
  (t:natlt (Shared.token_count smoke))
  : Lemma (
      B.projection_row (B.pack_token smoke x t) (B.pack_weights smoke w) ==
      C.chest2_row (Shared.projection smoke x w) t) =
  B.pack_weights_transposes smoke w;
  let cell (j:natlt (Shared.expert_count smoke))
    : Lemma (
        C.acc1 (B.projection_row (B.pack_token smoke x t) (B.pack_weights smoke w)) j ==
        C.acc1 (C.chest2_row (Shared.projection smoke x w) t) j) =
    Shared.projection_entry smoke x w t j;
    G.__gmatmul_single_congr 0.0R ( *. ) ( +. )
      (C.to_real_chest (B.token_view (B.pack_token smoke x t)))
      (C.to_real_chest (B.weight_view (B.pack_weights smoke w)))
      (C.to_real_chest x) (C.to_real_chest w) 0 j t j (Shared.hidden_count smoke)
  in
  FStar.Classical.forall_intro cell;
  B.chest1_extensional
    (B.projection_row (B.pack_token smoke x t) (B.pack_weights smoke w))
    (C.chest2_row (Shared.projection smoke x w) t)

let base_batch_refines smoke x w p =
  let row (t:natlt (Shared.token_count smoke))
    : Lemma (forall (j:natlt (Shared.expert_count smoke)).
        Seq.index (p t) j %~ C.acc2 (Shared.projection_softmax smoke x w) t j) =
    base_projection_agrees (B.pack_token smoke x t) (B.pack_weights smoke w);
    packed_projection_agrees smoke x w t;
    assert (task 1 (Shared.hidden_count smoke) (Shared.expert_count smoke)
      (B.pack_token smoke x t) (B.pack_weights smoke w) (p t));
    let cell (j:natlt (Shared.expert_count smoke))
      : Lemma (Seq.index (p t) j %~ C.acc2 (Shared.projection_softmax smoke x w) t j) =
      assert (Seq.index (p t) j %~
        C.acc2 (target 1 (Shared.hidden_count smoke) (Shared.expert_count smoke)
          (B.pack_token smoke x t) (B.pack_weights smoke w)) 0 j);
      target_entry 1 (Shared.hidden_count smoke) (Shared.expert_count smoke)
        (B.pack_token smoke x t) (B.pack_weights smoke w) 0 j;
      assert_norm (C.acc2 (Shared.projection_softmax smoke x w) t j ==
        C.acc1 (SM.softmax_real (C.chest2_row (Shared.projection smoke x w) t)) j)
    in
    FStar.Classical.forall_intro cell
  in
  FStar.Classical.forall_intro row

ghost
fn frame_to_native (#smoke:bool) (b:base_buffers smoke)
  requires frame_inputs b
  ensures on gpu_loc (B.input_frame_ownership
    b.input_counts b.input_indices b.input_probabilities b.input_survival b.frame)
{
  unfold frame_inputs b;
  if (I64.v b.token = 0) {
    on_emp_intro gpu_loc;
    rewrite on gpu_loc emp as on gpu_loc (B.input_frame_ownership
      b.input_counts b.input_indices b.input_probabilities b.input_survival b.frame);
  } else {
    rewrite on gpu_loc (
      A.pts_to b.input_counts b.frame.B.previous_counts **
      A.pts_to b.input_indices b.frame.B.previous_indices **
      A.pts_to b.input_probabilities b.frame.B.previous_probabilities **
      A.pts_to b.input_survival b.frame.B.previous_survival)
      as on gpu_loc (B.input_frame_ownership
        b.input_counts b.input_indices b.input_probabilities b.input_survival b.frame);
  }
}
ghost
fn frame_from_native (#smoke:bool) (b:base_buffers smoke)
  requires on gpu_loc (B.input_frame_ownership
    b.input_counts b.input_indices b.input_probabilities b.input_survival b.frame)
  ensures frame_inputs b
{
  if (I64.v b.token = 0) {
    rewrite on gpu_loc (B.input_frame_ownership
      b.input_counts b.input_indices b.input_probabilities b.input_survival b.frame)
      as on gpu_loc emp;
    on_emp_elim gpu_loc;
    rewrite emp as frame_inputs b;
  } else {
    rewrite on gpu_loc (B.input_frame_ownership
      b.input_counts b.input_indices b.input_probabilities b.input_survival b.frame)
      as on gpu_loc (
        A.pts_to b.input_counts b.frame.B.previous_counts **
        A.pts_to b.input_indices b.frame.B.previous_indices **
        A.pts_to b.input_probabilities b.frame.B.previous_probabilities **
        A.pts_to b.input_survival b.frame.B.previous_survival);
    rewrite on gpu_loc (
      A.pts_to b.input_counts b.frame.B.previous_counts **
      A.pts_to b.input_indices b.frame.B.previous_indices **
      A.pts_to b.input_probabilities b.frame.B.previous_probabilities **
      A.pts_to b.input_survival b.frame.B.previous_survival) as frame_inputs b;
  }
}

fn base
  (smoke:bool)
  (tokens:device_array bf16 (rows Base smoke * Shared.hidden_count smoke))
  (weights:device_array bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (probabilities:device_array f32 (rows Base smoke * Shared.expert_count smoke))
  (aux:auxiliary Base smoke)
  (#x:erased (lseq bf16 (rows Base smoke * Shared.hidden_count smoke)))
  (#w:erased (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)))
  preserves cpu ** on gpu_loc (A.pts_to tokens x ** A.pts_to weights w)
  requires pure (domain (rows Base smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) x w) **
    before Base smoke aux **
    (exists* (old:lseq f32 (rows Base smoke * Shared.expert_count smoke)).
      on gpu_loc (A.pts_to probabilities old))
  ensures exists* (p:lseq f32 (rows Base smoke * Shared.expert_count smoke)).
    on gpu_loc (A.pts_to probabilities p) ** after Base smoke aux ** pure (post Base smoke x w p)
{
  unfold before Base smoke aux;
  frame_to_native aux;
  B.run_profile_row smoke aux.token
    #(FStar.Ghost.hide (FStar.Ghost.reveal x)) #(FStar.Ghost.hide (FStar.Ghost.reveal w))
    #(FStar.Ghost.hide (FStar.Ghost.reveal aux.frame))
    tokens weights aux.input_counts aux.input_indices aux.input_probabilities aux.input_survival
    probabilities aux.output_counts aux.output_indices aux.output_probabilities aux.output_survival;
  with p. assert on gpu_loc (A.pts_to probabilities p);
  let _ = base_post_refines smoke x w p;
  assert pure (post Base smoke x w p);
  frame_from_native aux;
  fold after Base smoke aux;
}

fn optimized
  (smoke:bool)
  (tokens:device_array bf16 (rows Optimized smoke * Shared.hidden_count smoke))
  (weights:device_array bf16 (Shared.expert_count smoke * Shared.hidden_count smoke))
  (probabilities:device_array f32 (rows Optimized smoke * Shared.expert_count smoke))
  (aux:auxiliary Optimized smoke)
  (#x:erased (lseq bf16 (rows Optimized smoke * Shared.hidden_count smoke)))
  (#w:erased (lseq bf16 (Shared.expert_count smoke * Shared.hidden_count smoke)))
  preserves cpu ** on gpu_loc (A.pts_to tokens x ** A.pts_to weights w)
  requires pure (domain (rows Optimized smoke) (Shared.hidden_count smoke) (Shared.expert_count smoke) x w) **
    before Optimized smoke aux **
    (exists* (old:lseq f32 (rows Optimized smoke * Shared.expert_count smoke)).
      on gpu_loc (A.pts_to probabilities old))
  ensures exists* (p:lseq f32 (rows Optimized smoke * Shared.expert_count smoke)).
    on gpu_loc (A.pts_to probabilities p) ** after Optimized smoke aux ** pure (post Optimized smoke x w p)
{
  unfold before Optimized smoke aux;
  let _ = O.profile_dimensions smoke;
  let _ = optimized_domain smoke x w;
  let m = (if smoke then 256UL else 1024UL);
  let h = (if smoke then 512UL else 5120UL);
  let e = (if smoke then 8UL else 16UL);
  let modeled = O.kplay_run m h e tokens weights probabilities
    #_ #_ #(FStar.Ghost.hide (FStar.Ghost.reveal x)) #(FStar.Ghost.hide (FStar.Ghost.reveal w));
  with p. assert on gpu_loc (A.pts_to probabilities p);
  let _ = optimized_post_refines smoke x w p modeled;
  assert pure (post Optimized smoke x w p);
  fold after Optimized smoke aux;
}

#pop-options
