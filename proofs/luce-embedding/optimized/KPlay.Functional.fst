module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module U32 = FStar.UInt32
module F32 = Kuiper.Float32
module BF = Kuiper.BFloat16
module Fast = Kuiper.Float32.FastMath
module Cast = FStar.Int.Cast
module RO = Kuiper.Float.Realops
module Pow = FStar.Math.Pow
module Exp = FStar.Math.Exp
module Trig = Kuiper.Real.Trigonometry
module Casts = Kuiper.Float.Casts
module R = FStar.Real

(*** Real-level facts *)

let log2_e7 : real = RO.r_log2 10000000.0R

let freq_r (p:int) (i:nat) : real =
  R.of_int p *. (1.0R /. Pow.exp2 ((R.of_int (2 * i) *. (1.0R /. 64.0R)) *. log2_e7))

let freq_ok (p:int) (i:nat{i < 32})
  : Lemma (S.index (rope_cos opt_numerics p) i == Trig.cos (freq_r p i) /\
           S.index (rope_sin opt_numerics p) i == Trig.sin (freq_r p i))
  = RO.log_base_nonzero 2.0R;
    assert (at inv_freq_exp2 i ==
            1.0R /. Pow.exp2 ((R.of_int (2 * i) /. 64.0R) *. log2_e7));
    assert (R.of_int (2 * i) /. 64.0R == R.of_int (2 * i) *. (1.0R /. 64.0R))

(* log2(1e7) computed as logf(1e7) / logf(2) *)
inline_for_extraction noextract
let log2_e7_f32 () : f32 = div (flog (of_int #f32 10000000L)) (flog (of_int #f32 2L))

let log2_e7_f32_ok ()
  : Lemma (log2_e7_f32 () %~ log2_e7)
  = approx_of_int #f32;
    approx_apply (of_int #f32) RO.r_of_int 10000000L 10000000L;
    approx_apply (of_int #f32) RO.r_of_int 2L 2L;
    approx_flog #f32;
    assert (flog (of_int #f32 10000000L) %~ Exp.log 10000000.0R);
    assert (flog (of_int #f32 2L) %~ Exp.log 2.0R);
    RO.log_base_nonzero 2.0R;
    approx_div #f32;
    assert (div (flog (of_int #f32 10000000L)) (flog (of_int #f32 2L)) %~
            (Exp.log 10000000.0R /. Exp.log 2.0R))

let of_int_ok (x:I64.t)
  : Lemma (of_int #f32 x %~ R.of_int (I64.v x))
  = approx_of_int #f32;
    approx_apply (of_int #f32) RO.r_of_int x x

let neg_ok (x:f32) (r:real)
  : Lemma (requires x %~ r)
          (ensures neg x %~ (0.0R -. r) /\ Pow.exp2 (0.0R -. r) == 1.0R /. Pow.exp2 r)
  = approx_neg #f32;
    Pow.pow_neg 2.0R r

(*** Embedding row *)

let row_ok (st:S.seq BF.t) (tok:int) (h:S.seq F32.t) : prop =
  S.length h == 1024 /\
  (forall (j:nat). j < 1024 ==>
    0 <= tok * 1024 + j /\ tok * 1024 + j < S.length st /\
    S.index h j == Casts.fcast (S.index st (tok * 1024 + j)))

let row_embed (st:S.seq BF.t) (tok:int) (h:S.seq F32.t) (rt:S.seq real)
  : Lemma (requires row_ok st tok h /\ st %~ rt)
          (ensures h %~ embed rt tok)
  = let aux (j:nat{j < 1024})
      : Lemma (S.index h j %~ S.index (embed rt tok) j)
      = Casts.fcast_approx #BF.t #F32.t (S.index st (tok * 1024 + j)) (S.index rt (tok * 1024 + j))
    in
    FStar.Classical.forall_intro aux

let emb_premise (st:S.seq BF.t) (sp sk:S.seq I64.t) (p:int) : prop =
  S.length st == table_len /\ 1 <= S.length sp /\ 1 <= S.length sk /\
  0 <= p /\ p < 12 /\ p < S.length sp + S.length sk - 1 /\
  valid_token (opt_emb_token sp sk p)

(*** Device code *)

inline_for_extraction noextract
fn select_token
  (prompt : array I64.t) (plen : SZ.t)
  (tokens : array I64.t) (tlen : SZ.t)
  (pos : SZ.t)
  (#f : perm) (#sp #sk : erased (S.seq I64.t))
  preserves prompt |-> Frac f sp ** tokens |-> Frac f sk
  requires pure (S.length sp == SZ.v plen /\ S.length sk == SZ.v tlen)
  returns t : I64.t
  ensures pure (1 <= SZ.v plen /\ 1 <= SZ.v tlen /\ SZ.v pos < SZ.v plen + SZ.v tlen - 1 ==>
                I64.v t == opt_emb_token sp sk (SZ.v pos))
{
  pts_to_len prompt;
  pts_to_len tokens;
  if (pos = 0sz) {
    if (0sz <^ tlen) { tokens.(0sz) } else { 0L }
  } else if (pos <^ plen) {
    prompt.(pos)
  } else {
    let q = pos -^ plen;
    if (q <^ tlen) { tokens.(q) } else { 0L }
  }
}

inline_for_extraction noextract
fn copy_row
  (table : larray BF.t table_len)
  (hidden : larray F32.t 1024)
  (t : I64.t { valid_token (I64.v t) })
  (#f : perm) (#st : erased (S.seq BF.t)) (#h0 : erased (S.seq F32.t))
  preserves table |-> Frac f st
  requires hidden |-> h0
  ensures exists* h. hidden |-> h ** pure (row_ok st (I64.v t) h)
{
  pts_to_len table;
  pts_to_len hidden;
  let base = SZ.uint32_to_sizet (Cast.int64_to_uint32 t) *^ 1024sz;
  let mut j = 0sz;
  while (!j <^ 1024sz)
    invariant exists* (vj : SZ.t) (h : S.seq F32.t).
      j |-> vj ** hidden |-> h **
      pure (SZ.v vj <= 1024 /\ S.length h == 1024 /\
            (forall (k:nat). k < SZ.v vj ==>
              S.index h k == Casts.fcast (S.index st (I64.v t * 1024 + k))))
    decreases (1024 - SZ.v !j)
  {
    let vj = !j;
    let x = table.(base +^ vj);
    pts_to_len hidden;
    hidden.(vj) <- Casts.fcast #BF.t #F32.t x;
    j := vj +^ 1sz;
  };
}

inline_for_extraction noextract
fn embed_hidden
  (table : larray BF.t table_len)
  (prompt : array I64.t) (plen : SZ.t)
  (tokens : array I64.t) (tlen : SZ.t)
  (position : I64.t)
  (hidden : larray F32.t 1024)
  (#f : perm) (#st : erased (S.seq BF.t)) (#sp #sk : erased (S.seq I64.t))
  (#h0 : erased (S.seq F32.t))
  preserves table |-> Frac f st ** prompt |-> Frac f sp ** tokens |-> Frac f sk
  requires hidden |-> h0 ** pure (S.length sp == SZ.v plen /\ S.length sk == SZ.v tlen)
  ensures exists* h. hidden |-> h **
    pure (emb_premise st sp sk (I64.v position) ==>
          row_ok st (opt_emb_token sp sk (I64.v position)) h)
{
  if (I64.gte position 0L && I64.lt position 12L) {
    let pos = SZ.uint32_to_sizet (Cast.int64_to_uint32 position);
    let t = select_token prompt plen tokens tlen pos;
    if (I64.gte t 0L && I64.lt t 248320L) {
      copy_row table hidden t;
    } else {
      ()
    }
  } else {
    ()
  }
}

inline_for_extraction noextract
fn rope_freq (pf : f32) (pr : erased int) (i : SZ.t { SZ.v i < 32 }) (l : f32)
  preserves gpu
  requires pure (pf %~ R.of_int pr /\ l %~ log2_e7)
  returns r : f32
  ensures pure (r %~ freq_r pr (SZ.v i))
{
  let ii = Cast.uint32_to_int64 (SZ.sizet_to_u32 (2sz *^ i));
  let fi = of_int #f32 ii;
  of_int_ok ii;
  let c64 = of_int #f32 64L;
  of_int_ok 64L;
  let r64 = F32.rcp_approx_ftz c64;
  let a = F32.mul_rn_ftz fi r64;
  let b = F32.mul_rn_ftz a l;
  (* ptxas compiles the source's div.approx(p, ex2(m)) as p * ex2(-m). *)
  let nb = neg b;
  neg_ok b ((R.of_int (2 * SZ.v i) *. (1.0R /. 64.0R)) *. log2_e7);
  let e = F32.exp2_approx_ftz nb;
  F32.mul_rn_ftz pf e
}

inline_for_extraction noextract
fn rope
  (position : I64.t)
  (plen : SZ.t)
  (cosine : larray F32.t 32)
  (sine : larray F32.t 32)
  (#c0 #s0 : erased (S.seq F32.t))
  preserves gpu
  requires cosine |-> c0 ** sine |-> s0
  ensures exists* c s. cosine |-> c ** sine |-> s **
    pure (SZ.v plen >= 1 ==>
          (c %~ rope_cos opt_numerics (I64.v position) /\
           s %~ rope_sin opt_numerics (I64.v position)))
{
  pts_to_len cosine;
  pts_to_len sine;
  let pf = of_int #f32 position;
  of_int_ok position;
  let l = log2_e7_f32 ();
  log2_e7_f32_ok ();
  (* Runtime start index (0 whenever the relation's premise holds): keeps the
     rotary loop from being unrolled into compile-time-folded approximations. *)
  let start = (if (plen = 0sz) then 1sz else 0sz);
  let mut i = start;
  while (!i <^ 32sz)
    invariant exists* (vi : SZ.t) (c s : S.seq F32.t).
      i |-> vi ** cosine |-> c ** sine |-> s **
      pure (SZ.v start <= SZ.v vi /\ SZ.v vi <= 32 /\ S.length c == 32 /\ S.length s == 32 /\
            (forall (k:nat). SZ.v start <= k /\ k < SZ.v vi ==>
              S.index c k %~ S.index (rope_cos opt_numerics (I64.v position)) k /\
              S.index s k %~ S.index (rope_sin opt_numerics (I64.v position)) k))
    decreases (32 - SZ.v !i)
  {
    let vi = !i;
    let fr = rope_freq pf (I64.v position) vi l;
    freq_ok (I64.v position) (SZ.v vi);
    let cv = Fast.cos fr;
    let sv = Fast.sin fr;
    pts_to_len cosine;
    pts_to_len sine;
    cosine.(vi) <- cv;
    sine.(vi) <- sv;
    i := vi +^ 1sz;
  };
}

let finish (st:S.seq BF.t) (sp sk:S.seq I64.t) (p:int) (h c s:S.seq F32.t)
  : Lemma (requires (emb_premise st sp sk p ==> row_ok st (opt_emb_token sp sk p) h) /\
                    S.length h == 1024 /\
                    (1 <= S.length sp ==> (c %~ rope_cos opt_numerics p /\ s %~ rope_sin opt_numerics p)))
          (ensures optimized_embedding_relation st sp sk p h c s)
  = let aux (rt:S.seq real)
      : Lemma (requires emb_premise st sp sk p /\ st %~ rt)
              (ensures h %~ embed rt (opt_emb_token sp sk p))
      = row_embed st (opt_emb_token sp sk p) h rt
    in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

inline_for_extraction noextract
fn kernel
  (table : larray BF.t table_len)
  (prompt : array I64.t) (plen : SZ.t)
  (tokens : array I64.t) (tlen : SZ.t)
  (position : I64.t)
  (hidden : larray F32.t 1024)
  (cosine : larray F32.t 32)
  (sine : larray F32.t 32)
  (#f : perm) (#st : erased (S.seq BF.t)) (#sp #sk : erased (S.seq I64.t))
  (#h0 #c0 #s0 : erased (S.seq F32.t))
  preserves gpu ** table |-> Frac f st ** prompt |-> Frac f sp ** tokens |-> Frac f sk
  requires hidden |-> h0 ** cosine |-> c0 ** sine |-> s0 **
    pure (S.length sp == SZ.v plen /\ S.length sk == SZ.v tlen)
  ensures exists* h c s. hidden |-> h ** cosine |-> c ** sine |-> s **
    pure (optimized_embedding_relation st sp sk (I64.v position) h c s)
{
  embed_hidden table prompt plen tokens tlen position hidden;
  rope position plen cosine sine;
  with h. assert hidden |-> h;
  with c. assert cosine |-> c;
  with s. assert sine |-> s;
  pts_to_len hidden;
  finish st sp sk (I64.v position) h c s;
}

(* The generic instance targets visibility_of arr; launch_kernel_1 needs gpu_of. *)
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

(*** Host entrypoint *)

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
{
  launch_kernel_1
    (fun () ->
      kernel table prompt plen tokens tlen position hidden cosine sine
        #f #st #sp #sk #h0 #c0 #s0);
}
