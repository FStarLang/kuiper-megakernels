module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = FStar.SizeT
module I64 = FStar.Int64

inline_for_extraction noextract let _ = ()

(* Clamped allocation extents; under the relation's domain (P,S >= 1) they are
   exactly P*1024, 442368 + (P+S-1)*6144, S and S*1024. *)
noextract
let snat (x:int) : nat = if x >= 0 then x else 0

noextract
let cache_extent (p:nat) (s:int) : nat = snat (442368 + (p + s - 1) * 6144)

(* One physical thread: gather prompt embedding rows, zero cache, recurrent
   state, tokens and history (initialize_state<<<1,1>>>). *)
fn kplay_run
  (table : array bf16 { is_global table })
  (prompt : array i64 { is_global prompt })
  (p : sz)
  (steps : i64)
  (emb : array bf16 { is_global emb })
  (cache : array bf16 { is_global cache })
  (recurrent : array f32 { is_global recurrent })
  (tokens : array i64 { is_global tokens })
  (history : array bf16 { is_global history })
  (#ft #fp : perm)
  (#ts : erased (S.seq bf16))
  (#ps : erased (S.seq i64))
  (#e0 #c0 #h0 : erased (S.seq bf16))
  (#r0 : erased (S.seq f32))
  (#t0 : erased (S.seq i64))
  preserves
    cpu **
    on gpu_loc (table |-> Frac ft ts) **
    on gpu_loc (prompt |-> Frac fp ps)
  requires
    on gpu_loc (emb |-> e0) **
    on gpu_loc (cache |-> c0) **
    on gpu_loc (recurrent |-> r0) **
    on gpu_loc (tokens |-> t0) **
    on gpu_loc (history |-> h0) **
    pure (S.length ts == table_len /\
          S.length ps == SZ.v p /\
          S.length e0 == SZ.v p * 1024 /\
          S.length c0 == cache_extent (SZ.v p) (I64.v steps) /\
          S.length r0 == opt_rec_len /\
          S.length t0 == snat (I64.v steps) /\
          S.length h0 == snat (I64.v steps) * 1024)
  ensures
    exists* (e c h : S.seq bf16) (r : S.seq f32) (t : S.seq i64).
      on gpu_loc (emb |-> e) **
      on gpu_loc (cache |-> c) **
      on gpu_loc (recurrent |-> r) **
      on gpu_loc (tokens |-> t) **
      on gpu_loc (history |-> h) **
      pure (base_initialize_relation ts ps (I64.v steps) e c r t h)
