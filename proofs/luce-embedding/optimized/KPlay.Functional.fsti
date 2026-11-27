module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module F32 = Kuiper.Float32
module BF = Kuiper.BFloat16

(* Host entrypoint of the optimized `embedding` family. ABI order:
   table [248320*1024] (bf16), prompt [plen] (int64), plen, tokens [tlen]
   (int64), tlen, position (int64); outputs hidden [1024], cos [32], sin [32]
   (float32). Premises are ownership and buffer lengths only; all domain
   premises are inside optimized_embedding_relation. *)
fn kplay_run
  (table : larray BF.t table_len { is_global table })
  (prompt : array I64.t { is_global prompt })
  (plen : SZ.t)
  (tokens : array I64.t { is_global tokens })
  (tlen : SZ.t)
  (position : I64.t)
  (hidden : larray F32.t 1024 { is_global hidden })
  (cosine : larray F32.t 32 { is_global cosine })
  (sine : larray F32.t 32 { is_global sine })
  (#f : perm)
  (#st : erased (S.seq BF.t))
  (#sp #sk : erased (S.seq I64.t))
  (#h0 #c0 #s0 : erased (S.seq F32.t))
  preserves cpu
  preserves on gpu_loc (table |-> Frac f st)
  preserves on gpu_loc (prompt |-> Frac f sp)
  preserves on gpu_loc (tokens |-> Frac f sk)
  requires on gpu_loc (hidden |-> h0)
  requires on gpu_loc (cosine |-> c0)
  requires on gpu_loc (sine |-> s0)
  requires pure (S.length sp == SZ.v plen /\ S.length sk == SZ.v tlen)
  ensures exists* (h c s : S.seq F32.t).
    on gpu_loc (hidden |-> h) **
    on gpu_loc (cosine |-> c) **
    on gpu_loc (sine |-> s) **
    pure (optimized_embedding_relation st sp sk (I64.v position) h c s)
