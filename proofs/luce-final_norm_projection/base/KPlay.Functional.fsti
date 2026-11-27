module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module BF = Kuiper.BFloat16

(* Host entrypoint of the base `final_norm_projection` family. ABI order:
   table [248320*1024], normw [1024], residual [1024] (bf16); outputs
   hidden [1024], logits [248320] (bf16). Premises are ownership and buffer
   lengths only; all numerical content is in base_final_norm_projection_relation. *)
fn kplay_run
  (table : larray BF.t table_len { is_global table })
  (normw : larray BF.t 1024 { is_global normw })
  (residual : larray BF.t 1024 { is_global residual })
  (hidden : larray BF.t 1024 { is_global hidden })
  (logits : larray BF.t 248320 { is_global logits })
  (#ft #fw #fr : perm)
  (#st #sn #sr : erased (S.seq BF.t))
  (#h0 #l0 : erased (S.seq BF.t))
  preserves cpu
  preserves on gpu_loc (table |-> Frac ft st)
  preserves on gpu_loc (normw |-> Frac fw sn)
  preserves on gpu_loc (residual |-> Frac fr sr)
  requires on gpu_loc (hidden |-> h0)
  requires on gpu_loc (logits |-> l0)
  ensures exists* (h l : S.seq BF.t).
    on gpu_loc (hidden |-> h) **
    on gpu_loc (logits |-> l) **
    pure (base_final_norm_projection_relation st sn sr h l)
