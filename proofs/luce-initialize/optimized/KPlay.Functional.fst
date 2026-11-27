module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = FStar.SizeT
module I64 = FStar.Int64
module U64 = FStar.UInt64
module Cast = FStar.Int.Cast
module F = Kuiper.Floating

(*** Specification of the kernel's final buffer contents *)

noextract
let tok_spec (sp : S.seq i64) (n : nat) : S.seq i64 =
  if n > 0 && S.length sp > 0 then S.upd (S.create n 0L) 0 (S.index sp 0) else S.create n 0L

let zeros_all_zero (#t:Type) {| F.floating t |} (n : nat)
  : Lemma (all_zero (S.create n (zero #t)))
  = ()

let kernel_satisfies (sp : S.seq i64) (n : nat)
  : Lemma (optimized_initialize_relation sp n
             (S.create opt_rec_len (zero #f32)) (S.create opt_conv_len (zero #f32))
             (S.create opt_kv_len (zero #bf16)) (S.create opt_kv_len (zero #bf16))
             (tok_spec sp n) (S.create (n * 1024) (zero #bf16)))
  = zeros_all_zero #f32 opt_rec_len;
    zeros_all_zero #f32 opt_conv_len;
    zeros_all_zero #bf16 opt_kv_len;
    zeros_all_zero #bf16 (n * 1024)

(*** Device code *)

inline_for_extraction noextract
fn fill (#t:Type0) (a : array t) (n : SZ.t) (x : t) (#s : erased (S.seq t))
  requires a |-> s ** pure (S.length s == SZ.v n)
  ensures a |-> S.create (SZ.v n) x
{
  pts_to_len a;
  let mut i = 0sz;
  while (!i <^ n)
    invariant live i ** pure (SZ.v !i <= SZ.v n)
    invariant exists* (s' : S.seq t).
      a |-> s' **
      pure (S.length s' == SZ.v n /\ (forall (j:nat). j < SZ.v !i /\ j < S.length s' ==> S.index s' j == x))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    pts_to_len a;
    a.(vi) <- x;
    i := vi +^ 1sz;
  };
  with s'. assert a |-> s';
  S.lemma_eq_intro s' (S.create (SZ.v n) x);
  ()
}

inline_for_extraction noextract
fn kf
  (prompt : array i64 { is_global prompt })
  (plen : SZ.t)
  (ntok : SZ.t)
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
    gpu **
    prompt |-> Frac pp sp
  requires
    rc |-> src ** cv |-> scv ** k |-> sk ** v |-> sv **
    tokens |-> stok ** history |-> shist **
    pure (S.length sp == SZ.v plen /\
          S.length src == opt_rec_len /\ S.length scv == opt_conv_len /\
          S.length sk == opt_kv_len /\ S.length sv == opt_kv_len /\
          S.length stok == SZ.v ntok /\ S.length shist == SZ.v ntok * 1024 /\
          SZ.fits (SZ.v ntok * 1024))
  ensures
    rc |-> S.create opt_rec_len (zero #f32) **
    cv |-> S.create opt_conv_len (zero #f32) **
    k |-> S.create opt_kv_len (zero #bf16) **
    v |-> S.create opt_kv_len (zero #bf16) **
    tokens |-> tok_spec sp (SZ.v ntok) **
    history |-> S.create (SZ.v ntok * 1024) (zero #bf16)
{
  fill rc (SZ.uint32_to_sizet 4718592ul) (zero #f32);
  fill cv (SZ.uint32_to_sizet 442368ul) (zero #f32);
  fill k (SZ.uint32_to_sizet 6291456ul) (zero #bf16);
  fill v (SZ.uint32_to_sizet 6291456ul) (zero #bf16);
  fill tokens ntok 0L;
  pts_to_len prompt;
  pts_to_len tokens;
  if (SZ.gt ntok 0sz && SZ.gt plen 0sz)
  {
    let p0 = prompt.(0sz);
    tokens.(0sz) <- p0;
    rewrite (tokens |-> S.upd (S.create (SZ.v ntok) 0L) 0 p0)
         as (tokens |-> tok_spec sp (SZ.v ntok));
  }
  else
  {
    rewrite (tokens |-> S.create (SZ.v ntok) 0L)
         as (tokens |-> tok_spec sp (SZ.v ntok));
  };
  fill history (SZ.mul ntok 1024sz) (zero #bf16);
}

instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

(*** Host entrypoint *)

let i64_to_nat_bound (x : i64)
  : Lemma (requires I64.v x >= 0) (ensures U64.v (Cast.int64_to_uint64 x) == I64.v x)
  = FStar.Math.Lemmas.pow2_lt_compat 64 63;
    FStar.Math.Lemmas.small_mod (I64.v x) (pow2 64)

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
{
  gpu_pts_to_ref_located history;
  gpu_pts_to_ref_located tokens;
  i64_to_nat_bound steps;
  let ntok = SZ.uint64_to_sizet (Cast.int64_to_uint64 steps);
  kernel_satisfies sp (SZ.v ntok);
  launch_kernel_1
    (fun () ->
      kf prompt plen ntok rc cv k v tokens history
        #pp #sp #src #scv #sk #sv #stok #shist);
}
