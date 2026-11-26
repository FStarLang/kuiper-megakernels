module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = FStar.SizeT
module I32 = FStar.Int32
module I64 = FStar.Int64
module F32 = Kuiper.Float32
module Cast = FStar.Int.Cast

(* ------------------------------------------------------------------ *)
(* Pure model of the update                                            *)

(* The selected expert used for indexing (0 when out of range: the
   relation then holds vacuously). *)
noextract
let sidx (s:I32.t) : n:nat{n < 8} =
  if 0 <= I32.v s && I32.v s < 8 then I32.v s else 0

noextract
let new_count (s r:I32.t) : I32.t =
  if 0 <= I32.v s && I32.v s < 8 && I32.v r < 2147483647 then I32.add r 1l else r

noextract
let keep_b (s r:I32.t) (t:I64.t) : bool =
  0 <= I32.v s && I32.v s < 8 && 0 <= I32.v r && I32.v r < 32 &&
  0 <= I64.v t && I64.v t < 256

noextract
let kslot (s r:I32.t) (t:I64.t) : n:nat{n < 256} =
  if keep_b s r t then 32 * sidx s + I32.v r else 0

noextract
let jsurv (s r:I32.t) (t:I64.t) : n:nat{n < 2048} =
  if keep_b s r t then 8 * I64.v t + sidx s else 0

(* fdividef(q, __fadd_rn(0.0f, p)) *)
noextract
let combine_lemma (q p:F32.t)
  : Lemma (optimized_combine_ok q p (div q (add zero p)))
  = let aux (rq rp:real)
      : Lemma (q %~ rq /\ p %~ rp /\ (0.0R +. rp) =!= 0.0R ==>
               div q (add zero p) %~ (rq /. (0.0R +. rp)))
      = introduce (q %~ rq /\ p %~ rp /\ (0.0R +. rp) =!= 0.0R) ==>
                  (div q (add zero p) %~ (rq /. (0.0R +. rp)))
        with (
          a0 #F32.t;
          a_add (zero #F32.t) p 0.0R rp;
          approx_div #F32.t
        )
    in
    Classical.forall_intro_2 aux

#push-options "--z3rlimit 50"
noextract
let routing_lemma
  (vrow vsp:S.seq F32.t) (vsel vcounts vsurv:S.seq I32.t) (vidx:S.seq I64.t)
  (vprobs:S.seq F32.t) (t:I64.t)
  : Lemma
    (requires S.length vrow == 8 /\ S.length vsel == 1 /\ S.length vsp == 1 /\
              S.length vcounts == 8 /\ S.length vidx == 256 /\
              S.length vprobs == 256 /\ S.length vsurv == 2048)
    (ensures (
      let s = S.index vsel 0 in
      let r = S.index vcounts (sidx s) in
      let prob = div (S.index vrow (sidx s)) (add zero (S.index vsp 0)) in
      let k = kslot s r t in
      let j = jsurv s r t in
      let kp = keep_b s r t in
      optimized_routing_relation vrow vsel vsp vcounts vidx vprobs vsurv (I64.v t)
        (S.upd vcounts (sidx s) (new_count s r))
        (S.upd vidx k (if kp then t else S.index vidx k))
        (S.upd vprobs k (if kp then prob else S.index vprobs k))
        (S.upd vsurv j (if kp then 1l else S.index vsurv j))))
  = let s = S.index vsel 0 in
    combine_lemma (S.index vrow (sidx s)) (S.index vsp 0)
#pop-options

(* ------------------------------------------------------------------ *)
(* Native helpers                                                      *)

inline_for_extraction noextract
let i32_to_sz (x:I32.t{0 <= I32.v x}) : (y:SZ.t{SZ.v y == I32.v x}) =
  FStar.UInt.pow2_values 32;
  FStar.Math.Lemmas.small_mod (I32.v x) (pow2 32);
  SZ.uint32_to_sizet (Cast.int32_to_uint32 x)

inline_for_extraction noextract
let i64_to_sz (x:I64.t{0 <= I64.v x /\ I64.v x < 256}) : (y:SZ.t{SZ.v y == I64.v x}) =
  FStar.UInt.pow2_values 32;
  FStar.Math.Lemmas.small_mod (I64.v x) (pow2 32);
  SZ.uint32_to_sizet (Cast.int64_to_uint32 x)

inline_for_extraction noextract
fn copy_into
  (#t:Type0) (n:sz)
  (src : larray t (SZ.v n))
  (dst : larray t (SZ.v n))
  (#p : perm)
  (#vs : erased (S.seq t))
  (#vd : erased (S.seq t))
  preserves gpu ** src |-> Frac p vs
  requires dst |-> vd
  ensures dst |-> vs
{
  pts_to_len src;
  pts_to_len dst;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j : sz) (s : S.seq t).
      i |-> j ** dst |-> s **
      pure (SZ.v j <= SZ.v n /\ S.length s == SZ.v n /\
            (forall (k:nat). k < SZ.v j ==> S.index s k == S.index vs k))
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    let x = src.(j);
    dst.(j) <- x;
    i := j +^ 1sz;
  };
  with s. assert (dst |-> s);
  assert (pure (S.equal s vs));
  rewrite (dst |-> s) as (dst |-> vs);
}

(* ------------------------------------------------------------------ *)
(* The kernel (single thread)                                          *)

inline_for_extraction noextract
fn kf
  (row : larray f32 8)
  (sel : larray i32 1)
  (sp : larray f32 1)
  (counts : larray i32 8)
  (idx : larray i64 256)
  (probs : larray f32 256)
  (surv : larray i32 2048)
  (token : i64)
  (out_counts : larray i32 8)
  (out_idx : larray i64 256)
  (out_probs : larray f32 256)
  (out_surv : larray i32 2048)
  (#prow #psel #psp #pcounts #pidx #pprobs #psurv : perm)
  (#vrow : erased (S.seq f32))
  (#vsel : erased (S.seq i32))
  (#vsp : erased (S.seq f32))
  (#vcounts : erased (S.seq i32))
  (#vidx : erased (S.seq i64))
  (#vprobs : erased (S.seq f32))
  (#vsurv : erased (S.seq i32))
  (#ocounts : erased (S.seq i32))
  (#oidx : erased (S.seq i64))
  (#oprobs : erased (S.seq f32))
  (#osurv : erased (S.seq i32))
  preserves
    gpu **
    (row |-> Frac prow vrow) **
    (sel |-> Frac psel vsel) **
    (sp |-> Frac psp vsp) **
    (counts |-> Frac pcounts vcounts) **
    (idx |-> Frac pidx vidx) **
    (probs |-> Frac pprobs vprobs) **
    (surv |-> Frac psurv vsurv)
  requires
    (out_counts |-> ocounts) **
    (out_idx |-> oidx) **
    (out_probs |-> oprobs) **
    (out_surv |-> osurv)
  ensures
    exists* (c' : S.seq i32) (i' : S.seq i64) (p' : S.seq f32) (v' : S.seq i32).
      (out_counts |-> c') **
      (out_idx |-> i') **
      (out_probs |-> p') **
      (out_surv |-> v') **
      pure (optimized_routing_relation vrow vsel vsp vcounts vidx vprobs vsurv
              (I64.v token) c' i' p' v')
{
  pts_to_len row;
  pts_to_len sel;
  pts_to_len sp;
  pts_to_len counts;
  pts_to_len idx;
  pts_to_len probs;
  pts_to_len surv;
  copy_into 8sz counts out_counts;
  copy_into 256sz idx out_idx;
  copy_into 256sz probs out_probs;
  copy_into 2048sz surv out_surv;
  let s = sel.(0sz);
  let p = sp.(0sz);
  let vs = (I32.lte 0l s && I32.lt s 8l);
  let si : sz = (if vs then i32_to_sz s else 0sz);
  let q = row.(si);
  let cw = add #f32 zero p;
  let prob = div #f32 q cw;
  let r = counts.(si);
  let nc = (if (vs && I32.lt r 2147483647l) then I32.add r 1l else r);
  out_counts.(si) <- nc;
  let keep = (vs && I32.lte 0l r && I32.lt r 32l && I64.lte 0L token && I64.lt token 256L);
  let k : sz = (if keep then (32sz *^ si) +^ i32_to_sz r else 0sz);
  let oi = idx.(k);
  out_idx.(k) <- (if keep then token else oi);
  let op = probs.(k);
  out_probs.(k) <- (if keep then prob else op);
  let j : sz = (if keep then (8sz *^ i64_to_sz token) +^ si else 0sz);
  let ov = surv.(j);
  out_surv.(j) <- (if keep then 1l else ov);
  routing_lemma vrow vsp vsel vcounts vsurv vidx vprobs token;
  assert (pure (SZ.v si == sidx s /\ nc == new_count s r /\ keep == keep_b s r token /\
                SZ.v k == kslot s r token /\ SZ.v j == jsurv s r token));
  ()
}

(* The generic instance targets visibility_of arr; launch_kernel_1 needs gpu_of. *)
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (row : larray f32 8 { is_global row })
  (sel : larray i32 1 { is_global sel })
  (sp : larray f32 1 { is_global sp })
  (counts : larray i32 8 { is_global counts })
  (idx : larray i64 256 { is_global idx })
  (probs : larray f32 256 { is_global probs })
  (surv : larray i32 2048 { is_global surv })
  (token : i64)
  (out_counts : larray i32 8 { is_global out_counts })
  (out_idx : larray i64 256 { is_global out_idx })
  (out_probs : larray f32 256 { is_global out_probs })
  (out_surv : larray i32 2048 { is_global out_surv })
  (#prow #psel #psp #pcounts #pidx #pprobs #psurv : perm)
  (#vrow : erased (S.seq f32))
  (#vsel : erased (S.seq i32))
  (#vsp : erased (S.seq f32))
  (#vcounts : erased (S.seq i32))
  (#vidx : erased (S.seq i64))
  (#vprobs : erased (S.seq f32))
  (#vsurv : erased (S.seq i32))
  (#ocounts : erased (S.seq i32))
  (#oidx : erased (S.seq i64))
  (#oprobs : erased (S.seq f32))
  (#osurv : erased (S.seq i32))
  preserves
    cpu **
    on gpu_loc (row |-> Frac prow vrow) **
    on gpu_loc (sel |-> Frac psel vsel) **
    on gpu_loc (sp |-> Frac psp vsp) **
    on gpu_loc (counts |-> Frac pcounts vcounts) **
    on gpu_loc (idx |-> Frac pidx vidx) **
    on gpu_loc (probs |-> Frac pprobs vprobs) **
    on gpu_loc (surv |-> Frac psurv vsurv)
  requires
    on gpu_loc (out_counts |-> ocounts) **
    on gpu_loc (out_idx |-> oidx) **
    on gpu_loc (out_probs |-> oprobs) **
    on gpu_loc (out_surv |-> osurv)
  ensures
    exists* (c' : S.seq i32) (i' : S.seq i64) (p' : S.seq f32) (v' : S.seq i32).
      on gpu_loc (out_counts |-> c') **
      on gpu_loc (out_idx |-> i') **
      on gpu_loc (out_probs |-> p') **
      on gpu_loc (out_surv |-> v') **
      pure (optimized_routing_relation vrow vsel vsp vcounts vidx vprobs vsurv
              (I64.v token) c' i' p' v')
{
  launch_kernel_1
    (fun () ->
      kf row sel sp counts idx probs surv token out_counts out_idx out_probs out_surv
        #prow #psel #psp #pcounts #pidx #pprobs #psurv
        #vrow #vsel #vsp #vcounts #vidx #vprobs #vsurv
        #ocounts #oidx #oprobs #osurv);
}
