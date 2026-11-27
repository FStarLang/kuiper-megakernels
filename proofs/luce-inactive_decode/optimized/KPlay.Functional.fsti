module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module SS = KPlay.SharedSpec

inline_for_extraction noextract let rec_n : nat = 4718592
inline_for_extraction noextract let conv_n : nat = 442368
inline_for_extraction noextract let kv_n : nat = 6291456
inline_for_extraction noextract let row_n : nat = 1024

(* Inputs are borrowed read-only (fraction f); outputs are fresh, fully owned
   buffers whose final contents satisfy the frozen family relation, with
   arguments in target ABI order (six inputs, p, l, six outputs). *)
fn kplay_run
  (state_in : larray F32.t rec_n) (conv_in : larray F32.t conv_n)
  (key_in value_in : larray BF.t kv_n)
  (steps : SZ.t) (hist_n : SZ.t)
  (tokens_in : larray I64.t (SZ.v steps)) (history_in : larray BF.t (SZ.v hist_n))
  (state_out : larray F32.t rec_n) (conv_out : larray F32.t conv_n)
  (key_out value_out : larray BF.t kv_n)
  (tokens_out : larray I64.t (SZ.v steps)) (history_out : larray BF.t (SZ.v hist_n))
  (p l : I64.t)
  (#f : perm)
  (#rc #cv : erased (S.seq F32.t)) (#k #v : erased (S.seq BF.t))
  (#tokens : erased (S.seq I64.t)) (#history : erased (S.seq BF.t))
  (#rc0 #cv0 : erased (S.seq F32.t)) (#k0 #v0 : erased (S.seq BF.t))
  (#tokens0 : erased (S.seq I64.t)) (#history0 : erased (S.seq BF.t))
  preserves cpu
  preserves on gpu_loc (state_in |-> Frac f rc)
  preserves on gpu_loc (conv_in |-> Frac f cv)
  preserves on gpu_loc (key_in |-> Frac f k)
  preserves on gpu_loc (value_in |-> Frac f v)
  preserves on gpu_loc (tokens_in |-> Frac f tokens)
  preserves on gpu_loc (history_in |-> Frac f history)
  requires on gpu_loc (state_out |-> rc0)
  requires on gpu_loc (conv_out |-> cv0)
  requires on gpu_loc (key_out |-> k0)
  requires on gpu_loc (value_out |-> v0)
  requires on gpu_loc (tokens_out |-> tokens0)
  requires on gpu_loc (history_out |-> history0)
  requires pure (SZ.v hist_n == SZ.v steps * row_n)
  ensures exists* (rc' cv' : S.seq F32.t) (k' v' : S.seq BF.t) (tokens' : S.seq I64.t) (history' : S.seq BF.t).
    on gpu_loc (state_out |-> rc') **
    on gpu_loc (conv_out |-> cv') **
    on gpu_loc (key_out |-> k') **
    on gpu_loc (value_out |-> v') **
    on gpu_loc (tokens_out |-> tokens') **
    on gpu_loc (history_out |-> history') **
    pure (SS.optimized_inactive_decode_relation rc cv k v tokens history (I64.v p) (I64.v l)
                                                rc' cv' k' v' tokens' history')
