module KPlay.Functional

(* Optimized routing family: one serial selected-weight / capacity / routing
   update (inputs/optimized/routing.cu, route_token).

   ABI (target.json order): row f32[8], selected i32[] (1 cell),
   selected_probability f32[] (1 cell), counts i32[8], token_indices i64[8,32],
   probabilities f32[8,32], survival i32[256,8], token (int64 scalar); then the
   fresh outputs counts' i32[8], token_indices' i64[8,32], probabilities'
   f32[8,32], survival' i32[256,8]. All tensors are flat row-major.

   The only requirements are ownership and buffer lengths. The value-level
   premises (selected expert and token in range, counts in [0, token]) are
   premises inside optimized_routing_relation (G6). *)

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq

fn kplay_run
  (row : larray f32 8 { is_global row })
  (sel : larray i32 1 { is_global sel })
  (sp : larray f32 1 { is_global sp })
  (counts : larray i32 8 { is_global counts })
  (idx : larray i64 256 { is_global idx })
  (probs : larray f32 256 { is_global probs })
  (surv : larray i32 2048 { is_global surv })
  (token : i64)
  (out_counts : larray i32 8 { is_global out_counts })
  (out_idx : larray i64 256 { is_global out_idx })
  (out_probs : larray f32 256 { is_global out_probs })
  (out_surv : larray i32 2048 { is_global out_surv })
  (#prow #psel #psp #pcounts #pidx #pprobs #psurv : perm)
  (#vrow : erased (S.seq f32))
  (#vsel : erased (S.seq i32))
  (#vsp : erased (S.seq f32))
  (#vcounts : erased (S.seq i32))
  (#vidx : erased (S.seq i64))
  (#vprobs : erased (S.seq f32))
  (#vsurv : erased (S.seq i32))
  (#ocounts : erased (S.seq i32))
  (#oidx : erased (S.seq i64))
  (#oprobs : erased (S.seq f32))
  (#osurv : erased (S.seq i32))
  preserves
    cpu **
    on gpu_loc (row |-> Frac prow vrow) **
    on gpu_loc (sel |-> Frac psel vsel) **
    on gpu_loc (sp |-> Frac psp vsp) **
    on gpu_loc (counts |-> Frac pcounts vcounts) **
    on gpu_loc (idx |-> Frac pidx vidx) **
    on gpu_loc (probs |-> Frac pprobs vprobs) **
    on gpu_loc (surv |-> Frac psurv vsurv)
  requires
    on gpu_loc (out_counts |-> ocounts) **
    on gpu_loc (out_idx |-> oidx) **
    on gpu_loc (out_probs |-> oprobs) **
    on gpu_loc (out_surv |-> osurv)
  ensures
    exists* (c' : S.seq i32) (i' : S.seq i64) (p' : S.seq f32) (v' : S.seq i32).
      on gpu_loc (out_counts |-> c') **
      on gpu_loc (out_idx |-> i') **
      on gpu_loc (out_probs |-> p') **
      on gpu_loc (out_surv |-> v') **
      pure (optimized_routing_relation vrow vsel vsp vcounts vidx vprobs vsurv
              (FStar.Int64.v token) c' i' p' v')
