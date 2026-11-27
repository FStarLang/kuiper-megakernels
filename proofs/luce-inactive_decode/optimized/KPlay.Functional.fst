module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF = Kuiper.BFloat16
module I64 = FStar.Int64
module SS = KPlay.SharedSpec
module A = Kuiper.Array.Core

(* Kuiper ships no sized instance for int64; memcpy byte counts use sizeof(int64_t). *)
inline_for_extraction noextract
instance sized_i64 : sized I64.t = { size = 8sz; default = 0L }

ghost
fn gpu_len (#t:Type0) (#n:erased nat) (a:larray t n) (#f:perm) (#s:erased (S.seq t))
  preserves on gpu_loc (a |-> Frac f s)
  ensures pure (S.length s == reveal n)
{
  ghost_impersonate gpu_loc (on gpu_loc (a |-> Frac f s))
    (on gpu_loc (a |-> Frac f s) ** pure (S.length s == reveal n)) fn _ {
    on_elim (a |-> Frac f s);
    Pulse.Lib.Array.pts_to_len a;
    on_intro (a |-> Frac f s);
  }
}

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
{
  gpu_len key_in; gpu_len value_in; gpu_len tokens_in;
  gpu_len history_in; gpu_len state_in; gpu_len conv_in;
  (* Source serial order: KV, tokens, history, recurrent state, convolution state. *)
  A.memcpy_device_to_device key_out key_in (SZ.uint32_to_sizet 6291456ul);
  A.memcpy_device_to_device value_out value_in (SZ.uint32_to_sizet 6291456ul);
  A.memcpy_device_to_device tokens_out tokens_in steps;
  A.memcpy_device_to_device history_out history_in hist_n;
  A.memcpy_device_to_device state_out state_in (SZ.uint32_to_sizet 4718592ul);
  A.memcpy_device_to_device conv_out conv_in (SZ.uint32_to_sizet 442368ul);
}
