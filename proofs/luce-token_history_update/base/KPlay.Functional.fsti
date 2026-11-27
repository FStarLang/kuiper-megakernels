module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module BF = Kuiper.BFloat16

(* Width of one history row (hidden size). *)
inline_for_extraction noextract
let row : sz = 1024sz

(* ABI: input0 old tokens [S], input1 old history [S,1024], input2 hidden [1024],
   input3 selected (rank 0), input4 prompt length P, input5 position t;
   output0 tokens [S], output1 history [S,1024]. [n] is S = numel(input0). *)
fn kplay_run
  (n : sz { SZ.fits (SZ.v n * 1024) })
  (old_tokens : larray i64 (SZ.v n) { is_global old_tokens })
  (old_history : larray bf16 (SZ.v n * 1024) { is_global old_history })
  (hidden : larray bf16 1024 { is_global hidden })
  (selected : larray i64 1 { is_global selected })
  (pP : i64 { is_global selected })
  (t : i64)
  (tokens : larray i64 (SZ.v n) { is_global tokens })
  (history : larray bf16 (SZ.v n * 1024) { is_global history })
  (#p0 #p1 #p2 #p3 : perm)
  (#s0 : erased (S.seq i64))
  (#s1 #s2 : erased (S.seq bf16))
  (#s3 : erased (S.seq i64))
  (#o0 : erased (S.seq i64))
  (#o1 : erased (S.seq bf16))
  preserves
    cpu **
    on gpu_loc (old_tokens |-> Frac p0 s0) **
    on gpu_loc (old_history |-> Frac p1 s1) **
    on gpu_loc (hidden |-> Frac p2 s2) **
    on gpu_loc (selected |-> Frac p3 s3)
  requires
    on gpu_loc (tokens |-> o0) **
    on gpu_loc (history |-> o1)
  ensures
    exists* (tokens' : S.seq i64) (history' : S.seq bf16).
      on gpu_loc (tokens |-> tokens') **
      on gpu_loc (history |-> history') **
      pure (base_token_history_update_relation s0 s1 s2 s3 (I64.v pP) (I64.v t) tokens' history')
