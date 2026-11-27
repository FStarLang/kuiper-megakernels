module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module SZ = Kuiper.SizeT
module I64 = FStar.Int64

(* Base-role embedding_rope: residual row selection (prompt embedding or
   token-table lookup) and the 32 BF16 RoPE cosine/sine coefficients at
   position t. Arguments are the actual buffers in target ABI order (table,
   prompt embeddings, tokens, position; residual, cosine, sine) with their
   element counts. Only ownership and lengths are required; every domain
   premise lives inside base_embedding_rope_relation. *)
fn kplay_run
  (tab_len emb_len tok_len : sz)
  (table : larray bf16 (SZ.v tab_len) { is_global table })
  (emb : larray bf16 (SZ.v emb_len) { is_global emb })
  (tokens : larray I64.t (SZ.v tok_len) { is_global tokens })
  (t : I64.t)
  (residual : larray bf16 1024 { is_global residual })
  (cos_out : larray bf16 32 { is_global cos_out })
  (sin_out : larray bf16 32 { is_global sin_out })
  (#ptab #pemb #ptok : perm)
  (#stab #semb : erased (seq bf16))
  (#stok : erased (seq I64.t))
  (#r0 #c0 #s0 : erased (seq bf16))
  preserves
    cpu **
    on gpu_loc (table |-> Frac ptab stab) **
    on gpu_loc (emb |-> Frac pemb semb) **
    on gpu_loc (tokens |-> Frac ptok stok)
  requires
    on gpu_loc (residual |-> r0) **
    on gpu_loc (cos_out |-> c0) **
    on gpu_loc (sin_out |-> s0)
  ensures
    exists* (r c s : seq bf16).
      on gpu_loc (residual |-> r) **
      on gpu_loc (cos_out |-> c) **
      on gpu_loc (sin_out |-> s) **
      pure (base_embedding_rope_relation stab semb stok (I64.v t) r c s)
