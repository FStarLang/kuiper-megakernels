module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = FStar.SizeT
module I64 = FStar.Int64

(* Host entrypoint of the optimized `initialize` family. ABI order:
   prompt [plen] (int64), steps (int64); outputs recurrent state, convolution
   state, keys, values, tokens [steps], history [steps*1024]. Premises are
   ownership and buffer lengths only; all domain premises are inside
   optimized_initialize_relation. *)
fn kplay_run
  (prompt : array i64 { is_global prompt })
  (plen : SZ.t)
  (steps : i64)
  (rc : array f32 { is_global rc })
  (cv : array f32 { is_global cv })
  (k : array bf16 { is_global k })
  (v : array bf16 { is_global v })
  (tokens : array i64 { is_global tokens })
  (history : array bf16 { is_global history })
  (#pp : perm)
  (#sp : erased (S.seq i64))
  (#src #scv : erased (S.seq f32))
  (#sk #sv : erased (S.seq bf16))
  (#stok : erased (S.seq i64))
  (#shist : erased (S.seq bf16))
  preserves
    cpu **
    on gpu_loc (prompt |-> Frac pp sp)
  requires
    on gpu_loc (rc |-> src) **
    on gpu_loc (cv |-> scv) **
    on gpu_loc (k |-> sk) **
    on gpu_loc (v |-> sv) **
    on gpu_loc (tokens |-> stok) **
    on gpu_loc (history |-> shist) **
    pure (S.length sp == SZ.v plen /\
          S.length src == opt_rec_len /\ S.length scv == opt_conv_len /\
          S.length sk == opt_kv_len /\ S.length sv == opt_kv_len /\
          S.length stok == I64.v steps /\ S.length shist == I64.v steps * 1024)
  ensures
    exists* (src' scv' : S.seq f32) (sk' sv' : S.seq bf16) (stok' : S.seq i64) (shist' : S.seq bf16).
      on gpu_loc (rc |-> src') **
      on gpu_loc (cv |-> scv') **
      on gpu_loc (k |-> sk') **
      on gpu_loc (v |-> sv') **
      on gpu_loc (tokens |-> stok') **
      on gpu_loc (history |-> shist') **
      pure (optimized_initialize_relation sp (I64.v steps) src' scv' sk' sv' stok' shist')
