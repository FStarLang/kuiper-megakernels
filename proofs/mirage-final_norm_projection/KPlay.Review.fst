module KPlay.Review
#lang-pulse

open Kuiper
module Seq = FStar.Seq
module I64 = FStar.Int64
module U64 = FStar.UInt64
module A = Pulse.Lib.Array
module Shared = KPlay.SharedSpec
module B = KPlay.Base.Functional
module O = KPlay.Optimized.Functional

#push-options "--z3rlimit 10 --fuel 4 --ifuel 2"

inline_for_extraction noextract
let base_snapshot (b:base_inputs) : GTot (B.input_snapshot (U64.v b.extent)) = {
  B.packed_values = b.packed_values;
  B.residual_values = b.residual_values;
  B.delta_values = b.delta_values
}
inline_for_extraction noextract
let base_model (b:base_inputs) : GTot (B.input_denotation (U64.v b.extent)) = {
  B.packed = b.packed_model;
  B.residual = b.residual_model;
  B.delta = b.delta_model
}
inline_for_extraction noextract
let base_input_agrees (b:base_inputs)
  : Lemma (
      (represented Base b ==> B.input_representation (base_snapshot b) (base_model b)) /\
      logical_input Base b == B.map_inputs b.norm_offset b.projection_offset (base_model b)) =
  assert_norm (logical_input Base b ==
    B.map_inputs b.norm_offset b.projection_offset (base_model b))
inline_for_extraction noextract
let optimized_input_agrees (b:optimized_inputs)
  : Lemma (
      (represented Optimized b ==>
        O.input_representation b.hidden_values b.gamma_values b.embedding_values
          b.hidden_model b.gamma_model b.embedding_model) /\
      logical_input Optimized b == O.map_inputs b.hidden_model b.gamma_model b.embedding_model) =
  assert_norm (logical_input Optimized b ==
    O.map_inputs b.hidden_model b.gamma_model b.embedding_model)

inline_for_extraction noextract
let base_post_refines (b:base_inputs)
  (hidden:lseq bf16 (rows * width)) (scores:lseq bf16 (rows * stride Base))
  : Lemma
      (requires B.postcondition b.norm_offset b.projection_offset (base_model b)
        (base_snapshot b) (base_snapshot b) hidden scores)
      (ensures post Base (logical_input Base b) hidden scores) =
  base_input_agrees b;
  let point (o:B.output_denotation)
    : Lemma
        (requires B.output_representation hidden scores o /\
          Shared.postcondition (logical_input Base b) (B.map_state (base_snapshot b))
            (B.map_outputs o) (B.map_state (base_snapshot b)) /\
          B.all_rows_projection (logical_input Base b) o)
        (ensures post Base (logical_input Base b) hidden scores) =
    let observed:observation = {normalized = o.B.hidden; logits = o.B.scores} in
    assert (observation_post Base (logical_input Base b) hidden scores observed)
  in
  FStar.Classical.ghost_lemma point

inline_for_extraction noextract
let optimized_post_refines (b:optimized_inputs)
  (hidden:lseq bf16 (rows * width)) (scores:lseq bf16 (rows * stride Optimized))
  : Lemma
      (requires represented Optimized b /\
        O.shared_refinement_post b.hidden_values b.gamma_values b.embedding_values hidden scores)
      (ensures post Optimized (logical_input Optimized b) hidden scores) =
  optimized_input_agrees b;
  let point (o:O.output_denotation)
    : Lemma
        (requires O.output_representation hidden scores o /\
          Shared.postcondition (logical_input Optimized b)
            (O.map_state b.hidden_values b.gamma_values b.embedding_values)
            (O.map_outputs o)
            (O.map_state b.hidden_values b.gamma_values b.embedding_values) /\
          O.all_scores (logical_input Optimized b) o)
        (ensures post Optimized (logical_input Optimized b) hidden scores) =
    let observed:observation = {
      normalized = o.O.hidden;
      logits = (fun p v -> o.O.scores p v)
    } in
    assert (observation_post Optimized (logical_input Optimized b) hidden scores observed);
    assert (forall (p:Shared.prompt_index) (c:nat{vocabulary <= c /\ c < stride Optimized}).
      Seq.index scores (p * stride Optimized + c) %~ 0.0R)
  in
  FStar.Classical.ghost_lemma point

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
{
  let _ = base_input_agrees b;
  unfold input_ownership Base p0 p1 p2 b;
  B.kplay_run b.extent b.packed b.residual b.delta b.norm_offset b.projection_offset
    #(FStar.Ghost.hide (base_snapshot b)) #(FStar.Ghost.hide (base_model b))
    #p0 #p1 #p2 hidden scores;
  with h s. assert on gpu_loc (A.pts_to hidden h ** A.pts_to scores s);
  let _ = base_post_refines b h s;
  assert pure (post Base x h s);
  fold input_ownership Base p0 p1 p2 b;
}

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
{
  unfold input_ownership Optimized p0 p1 p2 b;
  with old_h old_s. assert on gpu_loc (A.pts_to hidden old_h ** A.pts_to scores old_s);
  O.kplay_run 8192UL 1024UL 151936UL 7ul 153600ul b.hidden b.gamma b.embedding hidden scores
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.hidden_values))
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.gamma_values))
    #(FStar.Ghost.hide (FStar.Ghost.reveal b.embedding_values))
    #(FStar.Ghost.hide old_h) #(FStar.Ghost.hide old_s) #p0 #p1 #p2;
  with h s. assert on gpu_loc (A.pts_to hidden h ** A.pts_to scores s);
  let _ = optimized_post_refines b h s;
  assert pure (post Optimized x h s);
  fold input_ownership Optimized p0 p1 p2 b;
}

#pop-options
