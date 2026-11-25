module KPlay.Functional
#lang-pulse
open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout
open Kuiper.Injection
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, l2_col_major }
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
module MMA = Kuiper.TensorCore.MMA
module SZ = FStar.SizeT
open Kuiper.Tensor.Layout.Alg { l1_forward }
open Kuiper.Float.Casts
module Fast = Kuiper.Float32.FastMath
module ML = FStar.Math.Lemmas
module S = FStar.Seq
open KPlay.SharedSpec
open Kuiper.Float.Realops
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Spec.GEMM
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module Classical = FStar.Classical
module BF = Kuiper.BFloat16
#set-options "--z3rlimit 60"

(* ===== T2 ===== *)

noextract
let inv_l1 (n : nat) (p : natlt n)
  : Lemma (inverse_f (l1_forward n).imap p == idx1 p)
  = let i = inverse_f (l1_forward n).imap p in
    (l1_forward n).imap.is_inj i (idx1 p)

noextract
let inv_rm (m k : nat) (p : natlt (m * k))
  : Lemma (k > 0 /\ inverse_f (l2_row_major m k).imap p == idx2 #m #k (p / k) (p % k))
  = let i = inverse_f (l2_row_major m k).imap p in
    (l2_row_major m k).imap.is_inj i (idx2 #m #k (p / k) (p % k))

noextract
let inv_cm (m k : nat) (p : natlt (m * k))
  : Lemma (m > 0 /\ inverse_f (l2_col_major m k).imap p == idx2 #m #k (p % m) (p / m))
  = let i = inverse_f (l2_col_major m k).imap p in
    (l2_col_major m k).imap.is_inj i (idx2 #m #k (p % m) (p / m))

ghost
fn to_view (#et:Type0) (#n:nat) (a : array1 et (l1_forward n))
  (#r:nat) (#d:shape r) (l : tlayout d { is_full l /\ l.ulen == n })
  (#f:perm) (#s : chest1 et n)
  requires a |-> Frac f s
  ensures relay a l |-> Frac f (from_seq l (to_seq (l1_forward n) s))
{
  tensor_concr a;
  tensor_abs' l (core a);
}

ghost
fn from_view (#et:Type0) (#n:nat) (a : array1 et (l1_forward n))
  (#r:nat) (#d:shape r) (l : tlayout d { is_full l /\ l.ulen == n })
  (#f:perm) (#s : chest d et)
  requires relay a l |-> Frac f s
  ensures a |-> Frac f (from_seq (l1_forward n) (to_seq l s))
{
  tensor_concr (relay a l);
  rewrite each core (relay a l) as core a;
  tensor_abs' (l1_forward n) (core a);
  rewrite each from_array (l1_forward n) (core a) as a;
}

noextract
let rm_view_acc (#et:Type) (m k : pos) (s : chest1 et (m * k)) (i : natlt m) (j : natlt k)
  : Lemma (acc2 (from_seq (l2_row_major m k) (to_seq (l1_forward (m * k)) s)) i j == acc1 s (i * k + j))
  = inv_l1 (m * k) (i * k + j)

noextract
let cm_view_acc (#et:Type) (m k : pos) (s : chest1 et (m * k)) (i : natlt m) (j : natlt k)
  : Lemma (acc2 (from_seq (l2_col_major m k) (to_seq (l1_forward (m * k)) s)) i j == acc1 s (j * m + i))
  = inv_l1 (m * k) (j * m + i)

noextract
let rm_unview_acc (#et:Type) (m k : pos) (s : chest2 et m k) (p : natlt (m * k))
  : Lemma (acc1 (from_seq (l1_forward (m * k)) (to_seq (l2_row_major m k) s)) p == acc2 s (p / k) (p % k))
  = inv_rm m k p

(* ===== T3 ===== *)

(* tile maps *)
noextract
let ra_s (mode t : nat) : nat =
  if mode = 1 then
    (if t % 32 < 16 then (t / 32) * 16 + t % 32
     else 128 + ((t % 32 - 16) / 8) * 64 + (t / 32) * 8 + t % 8)
  else if mode = 2 then (t / 8) * 4 + t % 4 + ((t % 8) / 4) * 192
  else if mode = 4 then t / 16
  else t

noextract
let cb_s (mode t : nat) : nat =
  if mode = 3 || mode = 4 then t else 0

inline_for_extraction noextract
let ra_c (mode t : sz) : r:sz{SZ.v r == ra_s (SZ.v mode) (SZ.v t)} =
  if mode = 1sz then
    (if t %^ 32sz <^ 16sz then (t /^ 32sz) *^ 16sz +^ t %^ 32sz
     else 128sz +^ ((t %^ 32sz -^ 16sz) /^ 8sz) *^ 64sz +^ (t /^ 32sz) *^ 8sz +^ t %^ 8sz)
  else if mode = 2sz then (t /^ 8sz) *^ 4sz +^ t %^ 4sz +^ ((t %^ 8sz) /^ 4sz) *^ 192sz
  else if mode = 4sz then t /^ 16sz
  else t

inline_for_extraction noextract
let cb_c (mode t : sz) : r:sz{SZ.v r == cb_s (SZ.v mode) (SZ.v t)} =
  if mode = 3sz || mode = 4sz then t else 0sz

noextract
let gemm_ok (ma kk nb tt mode : nat) : prop =
  ma > 0 /\ kk > 0 /\ nb > 0 /\ tt > 0 /\ ma % 16 == 0 /\ kk % 16 == 0 /\ nb % 8 == 0 /\
  SZ.fits (ma * kk) /\ SZ.fits (kk * nb) /\ SZ.fits (16 * tt * 8) /\ SZ.fits (mode * 1) /\
  (forall (t:nat). t < tt ==> ra_s mode t < ma / 16 /\ cb_s mode t < nb / 8)

inline_for_extraction noextract
let fzero : f32 = zero

noextract
let rec chain (#ma #kk #nb : pos) (#_ : squash (ma % 16 == 0 /\ kk % 16 == 0 /\ nb % 8 == 0))
  (va : chest2 bf16 ma kk) (vb : chest2 bf16 kk nb) (r : natlt (ma / 16)) (c : natlt (nb / 8))
  (n : nat{n <= kk / 16})
  : Tot (chest2 f32 16 8) (decreases n)
  = if n = 0 then const (16 @| 8 @| INil) fzero
    else MMA.emma (chain va vb r c (n - 1))
                  (ematrix_subtile va 16 16 r (n - 1))
                  (ematrix_subtile vb 16 8 (n - 1) c)

noextract
let rec gemm_res (#ma #kk #nb #tt : pos) (mode : nat) (#_ : squash (gemm_ok ma kk nb tt mode))
  (va : chest2 bf16 ma kk) (vb : chest2 bf16 kk nb) (vc : chest2 f32 (16 * tt) 8)
  (n : nat{n <= tt})
  : Tot (chest2 f32 (16 * tt) 8) (decreases n)
  = if n = 0 then vc
    else update_tile (gemm_res mode va vb vc (n - 1)) 16 8 (n - 1) 0
           (chain va vb (ra_s mode (n - 1)) (cb_s mode (n - 1)) (kk / 16))

inline_for_extraction noextract
let sa_inst (ma kk : szp) (tr tc : sz)
  (#_ : squash (ma % 16 == 0 /\ kk % 16 == 0 /\ SZ.fits (ma * kk) /\ tr < ma / 16 /\ tc < kk / 16))
  : strided_row_major (subtile_layout (l2_row_major ma kk) 16 16 tr tc)
  = strided_row_major_subtile (l2_row_major ma kk)
      #() #(strided_row_major_l2_row_major #(hide (SZ.v ma)) #(hide (SZ.v kk)) #() #(Mkconcrete_sz kk))
      (hide 16) (hide 16) (hide (SZ.v tr)) (hide (SZ.v tc)) #(Mkconcrete_sz 16sz) #(Mkconcrete_sz 16sz) #(Mkconcrete_sz tr) #(Mkconcrete_sz tc)

inline_for_extraction noextract
let sb_inst (kk nb : szp) (tr tc : sz)
  (#_ : squash (kk % 16 == 0 /\ nb % 8 == 0 /\ SZ.fits (kk * nb) /\ tr < kk / 16 /\ tc < nb / 8))
  : strided_col_major (subtile_layout (l2_col_major kk nb) 16 8 tr tc)
  = strided_col_major_subtile (l2_col_major kk nb)
      #() #(strided_col_major_l2_col_major #(hide (SZ.v kk)) #(hide (SZ.v nb)) #() #(Mkconcrete_sz kk))
      (hide 16) (hide 8) (hide (SZ.v tr)) (hide (SZ.v tc)) #(Mkconcrete_sz 16sz) #(Mkconcrete_sz 8sz) #(Mkconcrete_sz tr) #(Mkconcrete_sz tc)

inline_for_extraction noextract
let sc_inst (tt : szp) (tr : sz)
  (#_ : squash (SZ.fits (16 * tt * 8) /\ tr < tt))
  : strided_row_major (subtile_layout (l2_row_major (16 * tt) 8) 16 8 tr 0)
  = strided_row_major_subtile (l2_row_major (16 * tt) 8)
      #() #(strided_row_major_l2_row_major #(hide (16 * SZ.v tt)) #(hide 8) #() #(Mkconcrete_sz 8sz))
      (hide 16) (hide 8) (hide (SZ.v tr)) (hide 0) #(Mkconcrete_sz 16sz) #(Mkconcrete_sz 8sz) #(Mkconcrete_sz tr) #(Mkconcrete_sz 0sz)

#push-options "--z3rlimit 100"
inline_for_extraction noextract
fn gemm_body (ma kk nb tt : szp) (mode : sz) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array2 bf16 (l2_row_major ma kk))
  (b : array2 bf16 (l2_col_major kk nb))
  (c : array2 f32 (l2_row_major (16 * tt) 8))
  (#fa #fb : perm) (#va : chest2 bf16 ma kk) (#vb : chest2 bf16 kk nb) (#vc : chest2 f32 (16 * tt) 8)
  preserves gpu ** a |-> Frac fa va ** b |-> Frac fb vb
  requires c |-> Frac (1.0R /. 32) vc
  ensures c |-> Frac (1.0R /. 32) (gemm_res mode va vb vc tt)
{
  let fr = MMA.alloc_fragment ();
  let mut t : szle tt = 0sz;
  while (!t <^ tt)
    invariant live t
    invariant exists* v. fr |-> v
    invariant c |-> Frac (1.0R /. 32) (gemm_res mode va vb vc !t)
    decreases (tt - !t)
  {
    let tv = !t;
    let rat : (r:sz{SZ.v r < ma / 16}) = ra_c mode tv;
    let cbt : (r:sz{SZ.v r < nb / 8}) = cb_c mode tv;
    MMA.fill fr fzero;
    let mut ch : szle (kk / 16) = 0sz;
    while (!ch <^ kk /^ 16sz)
      invariant live ch
      invariant fr |-> chain va vb (SZ.v rat) (SZ.v cbt) (SZ.v !ch)
      decreases (kk / 16 - !ch)
    {
      let chv = !ch;
      let ta = array2_extract_tile_ro' a 16 16 (hide (SZ.v rat)) (hide (SZ.v chv));
      let tb = array2_extract_tile_ro' b 16 8 (hide (SZ.v chv)) (hide (SZ.v cbt));
      MMA.mma_sync #_ #(sa_inst ma kk rat chv) ta #_ #(sb_inst kk nb chv cbt) tb fr;
      Pulse.Lib.Trade.elim_trade (array2_subtile b 16 8 (SZ.v chv) (SZ.v cbt) |-> Frac fb (ematrix_subtile vb 16 8 (SZ.v chv) (SZ.v cbt))) _;
      Pulse.Lib.Trade.elim_trade (array2_subtile a 16 16 (SZ.v rat) (SZ.v chv) |-> Frac fa (ematrix_subtile va 16 16 (SZ.v rat) (SZ.v chv))) _;
      ch := chv +^ 1sz;
    };
    let tc = array2_extract_tile_st c 16 8 (hide (SZ.v tv)) (hide 0);
    with vt0. assert fr |-> vt0;
    MMA.store fr #_ #(sc_inst tt tv) tc;
    with vt. assert fr |-> vt;
    Pulse.Lib.Forall.elim_forall vt;
    Pulse.Lib.Trade.elim_trade _ _;
    t := tv +^ 1sz;
  };
  with v. assert fr |-> v;
  drop_ (fr |-> v);
}
#pop-options

(* ===== T4 ===== *)

noextract
let from_to (#et:Type) (#r : nat) (#d : shape r) (l : full_tlayout d) (c : chest d et)
  : Lemma (from_seq l (to_seq l c) == c)
          [SMTPat (from_seq l (to_seq l c))]
  = assert (Kuiper.Chest.equal (from_seq l (to_seq l c)) c)

noextract
let vw (#et:Type) (#r:nat) (#d:shape r) (n:nat) (l : full_tlayout d{l.ulen == n}) (s : chest1 et n) : chest d et =
  from_seq l (to_seq (l1_forward n) s)

noextract
let uvw (#et:Type) (#r:nat) (#d:shape r) (n:nat) (l : full_tlayout d{l.ulen == n}) (s : chest d et) : chest1 et n =
  from_seq (l1_forward n) (to_seq l s)

noextract
let uvw_vw (#et:Type) (#r:nat) (#d:shape r) (n:nat) (l : full_tlayout d{l.ulen == n}) (s : chest1 et n)
  : Lemma (uvw n l (vw n l s) == s)
  = to_from l (to_seq (l1_forward n) s);
    from_to (l1_forward n) s

noextract
let gemm_out (ma kk nb tt : pos) (mode : nat) (#pf : squash (gemm_ok ma kk nb tt mode))
  (sa : chest1 bf16 (ma * kk)) (sb : chest1 bf16 (kk * nb)) (sc : chest1 f32 (16 * tt * 8))
  : chest1 f32 (16 * tt * 8)
  = uvw (16 * tt * 8) (l2_row_major (16 * tt) 8)
      (gemm_res mode (vw (ma * kk) (l2_row_major ma kk) sa) (vw (kk * nb) (l2_col_major kk nb) sb)
                     (vw (16 * tt * 8) (l2_row_major (16 * tt) 8) sc) tt)

noextract
let gk_pre (ma kk nb tt : pos) (mode : nat) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array1 bf16 (l1_forward (ma * kk))) (b : array1 bf16 (l1_forward (kk * nb)))
  (c : array1 f32 (l1_forward (16 * tt * 8)))
  (fa fb : perm) (sa : chest1 bf16 (ma * kk)) (sb : chest1 bf16 (kk * nb)) (sc : chest1 f32 (16 * tt * 8))
  (tid : nat) : slprop =
  relay a (l2_row_major ma kk) |-> Frac (fa /. 32) (vw (ma * kk) (l2_row_major ma kk) sa) **
  relay b (l2_col_major kk nb) |-> Frac (fb /. 32) (vw (kk * nb) (l2_col_major kk nb) sb) **
  relay c (l2_row_major (16 * tt) 8) |-> Frac (1.0R /. 32) (vw (16 * tt * 8) (l2_row_major (16 * tt) 8) sc)

noextract
let gk_post (ma kk nb tt : pos) (mode : nat) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array1 bf16 (l1_forward (ma * kk))) (b : array1 bf16 (l1_forward (kk * nb)))
  (c : array1 f32 (l1_forward (16 * tt * 8)))
  (fa fb : perm) (sa : chest1 bf16 (ma * kk)) (sb : chest1 bf16 (kk * nb)) (sc : chest1 f32 (16 * tt * 8))
  (tid : nat) : slprop =
  relay a (l2_row_major ma kk) |-> Frac (fa /. 32) (vw (ma * kk) (l2_row_major ma kk) sa) **
  relay b (l2_col_major kk nb) |-> Frac (fb /. 32) (vw (kk * nb) (l2_col_major kk nb) sb) **
  relay c (l2_row_major (16 * tt) 8) |-> Frac (1.0R /. 32)
    (gemm_res mode (vw (ma * kk) (l2_row_major ma kk) sa) (vw (kk * nb) (l2_col_major kk nb) sb)
                     (vw (16 * tt * 8) (l2_row_major (16 * tt) 8) sc) tt)

inline_for_extraction noextract
fn gemm_kf (ma kk nb tt : szp) (mode : sz) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array1 bf16 (l1_forward (ma * kk))) (b : array1 bf16 (l1_forward (kk * nb)))
  (c : array1 f32 (l1_forward (16 * tt * 8)))
  (fa fb : perm) (sa : erased (chest1 bf16 (ma * kk))) (sb : erased (chest1 bf16 (kk * nb)))
  (sc : erased (chest1 f32 (16 * tt * 8)))
  (tid : szlt 32sz) ()
  norewrite
  requires gpu ** gk_pre ma kk nb tt mode a b c fa fb sa sb sc tid ** thread_id 32sz tid
  ensures gpu ** gk_post ma kk nb tt mode a b c fa fb sa sb sc tid ** thread_id 32sz tid
{
  unfold (gk_pre ma kk nb tt mode a b c fa fb sa sb sc (SZ.v tid));
  gemm_body ma kk nb tt mode (relay a (l2_row_major ma kk)) (relay b (l2_col_major kk nb))
    (relay c (l2_row_major (16 * SZ.v tt) 8));
  fold (gk_post ma kk nb tt mode a b c fa fb sa sb sc (SZ.v tid));
}

ghost
fn gemm_setup (ma kk nb tt : pos) (mode : nat) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array1 bf16 (l1_forward (ma * kk))) (b : array1 bf16 (l1_forward (kk * nb)))
  (c : array1 f32 (l1_forward (16 * tt * 8)))
  (fa fb : perm) (sa : chest1 bf16 (ma * kk)) (sb : chest1 bf16 (kk * nb)) (sc : chest1 f32 (16 * tt * 8))
  ()
  norewrite
  requires a |-> Frac fa sa ** b |-> Frac fb sb ** c |-> sc
  ensures (forall+ (tid : natlt 32). gk_pre ma kk nb tt mode a b c fa fb sa sb sc tid) ** emp
{
  to_view a (l2_row_major ma kk);
  to_view b (l2_col_major kk nb);
  to_view c (l2_row_major (16 * tt) 8);
  tensor_share_n (relay a (l2_row_major ma kk)) 32;
  tensor_share_n (relay b (l2_col_major kk nb)) 32;
  tensor_share_n (relay c (l2_row_major (16 * tt) 8)) 32;
  forevery_zip3 #(natlt 32)
    (fun _ -> relay a (l2_row_major ma kk) |-> Frac (fa /. 32) (vw (ma * kk) (l2_row_major ma kk) sa))
    (fun _ -> relay b (l2_col_major kk nb) |-> Frac (fb /. 32) (vw (kk * nb) (l2_col_major kk nb) sb))
    (fun _ -> relay c (l2_row_major (16 * tt) 8) |-> Frac (1.0R /. 32) (vw (16 * tt * 8) (l2_row_major (16 * tt) 8) sc));
  forevery_ext #(natlt 32) _ (fun tid -> gk_pre ma kk nb tt mode a b c fa fb sa sb sc tid);
}

ghost
fn gemm_teardown (ma kk nb tt : pos) (mode : nat) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array1 bf16 (l1_forward (ma * kk))) (b : array1 bf16 (l1_forward (kk * nb)))
  (c : array1 f32 (l1_forward (16 * tt * 8)))
  (fa fb : perm) (sa : chest1 bf16 (ma * kk)) (sb : chest1 bf16 (kk * nb)) (sc : chest1 f32 (16 * tt * 8))
  ()
  norewrite
  requires (forall+ (tid : natlt 32). gk_post ma kk nb tt mode a b c fa fb sa sb sc tid) ** emp
  ensures a |-> Frac fa sa ** b |-> Frac fb sb ** c |-> gemm_out ma kk nb tt mode sa sb sc
{
  forevery_ext #(natlt 32) (fun tid -> gk_post ma kk nb tt mode a b c fa fb sa sb sc tid)
    (fun _ -> relay a (l2_row_major ma kk) |-> Frac (fa /. 32) (vw (ma * kk) (l2_row_major ma kk) sa) **
      relay b (l2_col_major kk nb) |-> Frac (fb /. 32) (vw (kk * nb) (l2_col_major kk nb) sb) **
      relay c (l2_row_major (16 * tt) 8) |-> Frac (1.0R /. 32)
        (gemm_res mode (vw (ma * kk) (l2_row_major ma kk) sa) (vw (kk * nb) (l2_col_major kk nb) sb)
                     (vw (16 * tt * 8) (l2_row_major (16 * tt) 8) sc) tt));
  forevery_unzip3 #(natlt 32) _ _ _;
  tensor_gather_n (relay a (l2_row_major ma kk)) 32;
  tensor_gather_n (relay b (l2_col_major kk nb)) 32;
  tensor_gather_n (relay c (l2_row_major (16 * tt) 8)) 32 #1.0R;
  from_view a (l2_row_major ma kk);
  from_view b (l2_col_major kk nb);
  from_view c (l2_row_major (16 * tt) 8);
  uvw_vw (ma * kk) (l2_row_major ma kk) sa;
  uvw_vw (kk * nb) (l2_col_major kk nb) sb;
  rewrite each from_seq (l1_forward (ma * kk)) (to_seq (l2_row_major ma kk) (vw (ma * kk) (l2_row_major ma kk) sa)) as sa;
  rewrite each from_seq (l1_forward (kk * nb)) (to_seq (l2_col_major kk nb) (vw (kk * nb) (l2_col_major kk nb) sb)) as sb;
  rewrite each from_seq (l1_forward (16 * tt * 8)) (to_seq (l2_row_major (16 * tt) 8)
                (gemm_res mode
                    (vw (ma * kk) (l2_row_major ma kk) sa)
                    (vw (kk * nb) (l2_col_major kk nb) sb)
                    (vw (16 * tt * 8) (l2_row_major (16 * tt) 8) sc)
                    tt))
    as gemm_out ma kk nb tt mode sa sb sc;
}

inline_for_extraction noextract
let gemm_kdesc (ma kk nb tt : szp) (mode : sz) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array1 bf16 (l1_forward (ma * kk)) {is_global a}) (b : array1 bf16 (l1_forward (kk * nb)) {is_global b})
  (c : array1 f32 (l1_forward (16 * tt * 8)) {is_global c})
  (fa fb : perm) (sa : erased (chest1 bf16 (ma * kk))) (sb : erased (chest1 bf16 (kk * nb)))
  (sc : erased (chest1 f32 (16 * tt * 8)))
  : kernel_desc_1_n
      (a |-> Frac fa sa ** b |-> Frac fb sb ** c |-> sc)
      (a |-> Frac fa sa ** b |-> Frac fb sb ** c |-> gemm_out ma kk nb tt mode sa sb sc)
  = {
    nthr = 32sz;
    frame = emp;
    kpre = gk_pre ma kk nb tt mode a b c fa fb sa sb sc;
    kpost = gk_post ma kk nb tt mode a b c fa fb sa sb sc;
    f = gemm_kf ma kk nb tt mode a b c fa fb sa sb sc;
    block_setup = gemm_setup ma kk nb tt mode a b c fa fb sa sb sc;
    block_teardown = gemm_teardown ma kk nb tt mode a b c fa fb sa sb sc;
    kpost_sendable = solve;
    kpre_sendable = solve;
    full_post_sendable = solve;
    full_pre_sendable = solve;
  }

inline_for_extraction noextract
fn gemm_launch (ma kk nb tt : szp) (mode : sz) (#pf : squash (gemm_ok ma kk nb tt mode))
  (a : array1 bf16 (l1_forward (ma * kk)) {is_global a}) (b : array1 bf16 (l1_forward (kk * nb)) {is_global b})
  (c : array1 f32 (l1_forward (16 * tt * 8)) {is_global c})
  (#fa #fb : perm) (#sa : erased (chest1 bf16 (ma * kk))) (#sb : erased (chest1 bf16 (kk * nb)))
  (#sc : erased (chest1 f32 (16 * tt * 8)))
  preserves cpu ** on gpu_loc (a |-> Frac fa sa) ** on gpu_loc (b |-> Frac fb sb)
  requires on gpu_loc (c |-> sc)
  ensures on gpu_loc (c |-> gemm_out ma kk nb tt mode sa sb sc)
{
  launch_sync (gemm_kdesc ma kk nb tt mode a b c fa fb sa sb sc);
}

(* ===== B0 ===== *)


inline_for_extraction noextract
let tob (x : f32) : bf16 = fcast x
inline_for_extraction noextract
let tof (x : bf16) : f32 = fcast x

noextract
let upd_acc1 (#et:Type) (#n:nat) (s : chest1 et n) (i j : natlt n) (v : et)
  : Lemma (acc1 (upd s (i, ()) v) j == (if j = i then v else acc1 s j))
          [SMTPat (acc1 (upd s (i, ()) v) j)]
  = ()

noextract
let inv_step (#et:Type) (#n:nat) (s : chest1 et n) (k : natlt n) (v : et) (f : natlt n -> GTot et)
  : Lemma (requires (forall (j:natlt n). j < k ==> acc1 s j == f j) /\ v == f k)
          (ensures forall (j:natlt n). j < k + 1 ==> acc1 (upd s (k, ()) v) j == f j)
  = ()

inline_for_extraction noextract
fn rd (#et:Type0) (#n:erased nat{SZ.fits n}) (a : array1 et (l1_forward n)) (i : sz{SZ.v i < n}) (#f:perm) (#s:erased (chest1 et n))
  preserves a |-> Frac f s
  returns v : et
  ensures pure (v == acc1 s (SZ.v i))
{
  Kuiper.Tensor.read a ((i <: szlt n), ()) #f
}

inline_for_extraction noextract
fn wr (#et:Type0) (#n:erased nat{SZ.fits n}) (a : array1 et (l1_forward n)) (i : sz{SZ.v i < n}) (v : et) (#s:erased (chest1 et n))
  requires a |-> s
  ensures a |-> upd s (SZ.v i, ()) v
{
  Kuiper.Tensor.write a ((i <: szlt n), ()) v
}

noextract
let mod_spec_lem (a : nat{SZ.fits a}) (b : pos{SZ.fits b}) : Lemma (SZ.mod_spec a b == a % b) =
  ML.euclidean_division_definition a b

noextract
let ms (a : sz) (b : pos{b < 65536}) : Lemma (SZ.fits b /\ SZ.mod_spec (SZ.v a) b == SZ.v a % b) =
  FStar.UInt.pow2_values 16;
  SZ.fits_at_least_16 b;
  mod_spec_lem (SZ.v a) b

inline_for_extraction noextract
let ( %% ) (a : sz) (b : sz{0 < SZ.v b /\ SZ.v b < 65536}) : (r:sz{SZ.v r == SZ.v a % SZ.v b}) =
  ms a (SZ.v b);
  a %^ b

inline_for_extraction noextract
let n64k : (x:szp{SZ.v x == 65536}) = SZ.uint32_to_sizet 65536ul

inline_for_extraction noextract
let fz : f32 = zero

inline_for_extraction noextract
let eps : f32 = div one (of_int 1000000L)

(* float constants pass through a local so that karamel does not fold float ops *)
inline_for_extraction noextract
fn mk_eps ()
  preserves gpu
  returns e : f32
  ensures pure (e == eps)
{
  let mut d : f32 = of_int 1000000L;
  let dv = !d;
  div one dv
}

noextract
let capx (#t:Type) {| scalar t, real_like t |} (#n:nat) (s : chest1 t n) (r : S.seq real) : prop =
  forall (i:natlt n). acc1 s i %~ at r i

noextract
let zo_apx () : Lemma ((zero #f32) %~ 0.0R /\ (one #f32) %~ 1.0R) =
  let _ : squash (v_approximates (zero #f32) r_zero) = a0 #f32 in
  let _ : squash (v_approximates (one #f32) r_one) = a1 #f32 in
  ()

noextract
let tobz_apx () : Lemma (tob zero %~ 0.0R) = zo_apx ()

noextract
let oi (n : FStar.Int64.t) : Lemma ((of_int #f32 n) %~ FStar.Real.of_int (FStar.Int64.v n)) =
  let _ = approx_of_int #f32 in
  assert (n %~ n)

(* ===== R1 ===== *)

(* ---------------- sdot ---------------- *)

noextract
let rec sdot_split (m p : nat) (a : S.seq real) (ao sa : int) (b : S.seq real) (bo sb : int)
  : Lemma (ensures sdot (m + p) a ao sa b bo sb ==
                   sdot m a ao sa b bo sb +. sdot p a (ao + m * sa) sa b (bo + m * sb) sb)
          (decreases p)
  = if p = 0 then ()
    else begin
      sdot_split m (p - 1) a ao sa b bo sb;
      ML.distributivity_add_left m (p - 1) sa;
      ML.distributivity_add_left m (p - 1) sb
    end

noextract
let rec sdot_ext (n : nat) (a : S.seq real) (ao sa : int) (b : S.seq real) (bo sb : int)
  (a' : S.seq real) (ao' sa' : int) (b' : S.seq real) (bo' sb' : int)
  : Lemma (requires forall (k:nat). k < n ==>
             at a (ao + k * sa) *. at b (bo + k * sb) == at a' (ao' + k * sa') *. at b' (bo' + k * sb'))
          (ensures sdot n a ao sa b bo sb == sdot n a' ao' sa' b' bo' sb')
          (decreases n)
  = if n = 0 then () else sdot_ext (n - 1) a ao sa b bo sb a' ao' sa' b' bo' sb'

noextract
let rec sdot_zero (n : nat) (a : S.seq real) (ao sa : int) (b : S.seq real) (bo sb : int)
  : Lemma (requires forall (k:nat). k < n ==> at a (ao + k * sa) *. at b (bo + k * sb) == 0.0R)
          (ensures sdot n a ao sa b bo sb == 0.0R)
          (decreases n)
  = if n = 0 then () else sdot_zero (n - 1) a ao sa b bo sb

(* padded tail: terms k in [m, n) vanish *)
noextract
let sdot_pad (m n : nat) (a : S.seq real) (ao sa : int) (b : S.seq real) (bo sb : int)
  : Lemma (requires m <= n /\ (forall (k:nat). m <= k /\ k < n ==> at a (ao + k * sa) *. at b (bo + k * sb) == 0.0R))
          (ensures sdot n a ao sa b bo sb == sdot m a ao sa b bo sb)
  = sdot_split m (n - m) a ao sa b bo sb;
    let aux (k:nat{k < n - m}) : Lemma (at a ((ao + m * sa) + k * sa) *. at b ((bo + m * sb) + k * sb) == 0.0R)
      = ML.distributivity_add_left m k sa;
        ML.distributivity_add_left m k sb
    in
    FStar.Classical.forall_intro aux;
    sdot_zero (n - m) a (ao + m * sa) sa b (bo + m * sb) sb

(* ===== G1 ===== *)


noextract
let rmA (ma kk : pos) (ra : S.seq real) : chest2 real ma kk = mk2 (fun x k -> at ra (x * kk + k))
noextract
let cmB (kk nb : pos) (rb : S.seq real) : chest2 real kk nb = mk2 (fun k y -> at rb (y * kk + k))

noextract
let rec rchain (#ma #kk #nb : pos) (#_ : squash (ma % 16 == 0 /\ kk % 16 == 0 /\ nb % 8 == 0))
  (ra : chest2 real ma kk) (rb : chest2 real kk nb) (r : natlt (ma / 16)) (c : natlt (nb / 8))
  (n : nat{n <= kk / 16})
  : Tot (chest2 real 16 8) (decreases n)
  = if n = 0 then const (16 @| 8 @| INil) 0.0R
    else matplus (rchain ra rb r c (n - 1))
                 (matmul (ematrix_subtile ra 16 16 r (n - 1)) (ematrix_subtile rb 16 8 (n - 1) c))

noextract
let rec chain_apx (#ma #kk #nb : pos) (#pf : squash (ma % 16 == 0 /\ kk % 16 == 0 /\ nb % 8 == 0))
  (va : chest2 bf16 ma kk) (vb : chest2 bf16 kk nb)
  (ra : chest2 real ma kk) (rb : chest2 real kk nb) (r : natlt (ma / 16)) (c : natlt (nb / 8))
  (n : nat{n <= kk / 16})
  : Lemma (requires va %~ ra /\ vb %~ rb)
          (ensures chain #ma #kk #nb #pf va vb r c n %~ rchain #ma #kk #nb #pf ra rb r c n)
          (decreases n)
  = if n = 0 then ()
    else begin
      chain_apx va vb ra rb r c (n - 1);
      MMA.emma_approx_lemma (chain #ma #kk #nb #pf va vb r c (n - 1))
        (ematrix_subtile va 16 16 r (n - 1)) (ematrix_subtile vb 16 8 (n - 1) c)
        (rchain #ma #kk #nb #pf ra rb r c (n - 1))
        (ematrix_subtile ra 16 16 r (n - 1)) (ematrix_subtile rb 16 8 (n - 1) c)
    end

noextract
let rec msingle_sdot (#m1 #m2 : pos) (xa : chest2 real m1 16) (xb : chest2 real 16 m2)
  (i : natlt m1) (j : natlt m2) (to : nat{to <= 16})
  (a : S.seq real) (ao : int) (b : S.seq real) (bo : int)
  : Lemma (requires forall (k:natlt 16). acc2 xa i k == at a (ao + k) /\ acc2 xb k j == at b (bo + k))
          (ensures __matmul_single xa xb i j to == sdot to a ao 1 b bo 1)
          (decreases to)
  = if to = 0 then () else msingle_sdot xa xb i j (to - 1) a ao b bo

noextract
let rec rchain_val (ma kk nb : pos) (#pf : squash (ma % 16 == 0 /\ kk % 16 == 0 /\ nb % 8 == 0))
  (ra rb : S.seq real) (r : natlt (ma / 16)) (c : natlt (nb / 8))
  (n : nat{n <= kk / 16}) (i : natlt 16) (j : natlt 8)
  : Lemma (ensures acc2 (rchain #ma #kk #nb #pf (rmA ma kk ra) (cmB kk nb rb) r c n) i j ==
                   sdot (16 * n) ra ((r * 16 + i) * kk) 1 rb ((c * 8 + j) * kk) 1)
          (decreases n)
  = if n = 0 then ()
    else begin
      rchain_val ma kk nb #pf ra rb r c (n - 1) i j;
      let xa = ematrix_subtile (rmA ma kk ra) 16 16 r (n - 1) in
      let xb = ematrix_subtile (cmB kk nb rb) 16 8 (n - 1) c in
      chest_comb_acc2 #16 #8 add (rchain #ma #kk #nb #pf (rmA ma kk ra) (cmB kk nb rb) r c (n - 1)) (matmul xa xb) i j;
      lemma_matmul_index xa xb i j;
      let ao = (r * 16 + i) * kk + 16 * (n - 1) in
      let bo = (c * 8 + j) * kk + 16 * (n - 1) in
      let aux (k:natlt 16) : Lemma (acc2 xa i k == at ra (ao + k) /\ acc2 xb k j == at rb (bo + k)) =
        ML.distributivity_add_left r 1 16;
        ML.lemma_mult_le_right 16 (r + 1) (ma / 16);
        ML.distributivity_add_left (n - 1) 1 16;
        ML.lemma_mult_le_right 16 n (kk / 16);
        ML.distributivity_add_left c 1 8;
        ML.lemma_mult_le_right 8 (c + 1) (nb / 8)
      in
      FStar.Classical.forall_intro aux;
      msingle_sdot xa xb i j 16 ra ao rb bo;
      sdot_split (16 * (n - 1)) 16 ra ((r * 16 + i) * kk) 1 rb ((c * 8 + j) * kk) 1
    end

noextract
let rec gemm_res_acc (#ma #kk #nb #tt : pos) (mode : nat) (#pf : squash (gemm_ok ma kk nb tt mode))
  (va : chest2 bf16 ma kk) (vb : chest2 bf16 kk nb) (vc : chest2 f32 (16 * tt) 8)
  (n : nat{n <= tt}) (x : natlt (16 * tt)) (y : natlt 8)
  : Lemma (ensures acc2 (gemm_res mode #pf va vb vc n) x y ==
                   (if x / 16 < n then acc2 (chain va vb (ra_s mode (x / 16)) (cb_s mode (x / 16)) (kk / 16)) (x % 16) y
                    else acc2 vc x y))
          (decreases n)
  = if n = 0 then () else gemm_res_acc mode #pf va vb vc (n - 1) x y

noextract
let gemm_apx (ma kk nb tt : pos) (mode : nat) (#pf : squash (gemm_ok ma kk nb tt mode))
  (sa : chest1 bf16 (ma * kk)) (sb : chest1 bf16 (kk * nb)) (sc : chest1 f32 (16 * tt * 8))
  (ra rb : S.seq real) (p : natlt (16 * tt * 8))
  : Lemma (requires capx sa ra /\ capx sb rb)
          (ensures acc1 (gemm_out ma kk nb tt mode #pf sa sb sc) p %~
                   sdot kk ra ((ra_s mode (p / 128) * 16 + (p / 8) % 16) * kk) 1
                           rb ((cb_s mode (p / 128) * 8 + p % 8) * kk) 1)
  = let va = vw (ma * kk) (l2_row_major ma kk) sa in
    let vb = vw (kk * nb) (l2_col_major kk nb) sb in
    let vc = vw (16 * tt * 8) (l2_row_major (16 * tt) 8) sc in
    let xa : chest2 real ma kk = rmA ma kk ra in
    let xb : chest2 real kk nb = cmB kk nb rb in
    let auxa (x:natlt ma) (k:natlt kk) : Lemma (acc2 va x k %~ acc2 xa x k) =
      rm_view_acc ma kk sa x k
    in
    let auxb (k:natlt kk) (y:natlt nb) : Lemma (acc2 vb k y %~ acc2 xb k y) =
      cm_view_acc kk nb sb k y;
      ML.distributivity_add_left y 1 kk;
      ML.lemma_mult_le_right kk (y + 1) nb;
      let q : natlt (kk * nb) = y * kk + k in
      assert (acc1 sb q %~ at rb q)
    in
    FStar.Classical.forall_intro_2 auxa;
    FStar.Classical.forall_intro_2 auxb;
    lemma_approximates_intro va xa;
    lemma_approximates_intro vb xb;
    rm_unview_acc (16 * tt) 8 (gemm_res mode #pf va vb vc tt) p;
    let x = p / 8 in
    let y = p % 8 in
    ML.division_multiplication_lemma p 8 16;
    gemm_res_acc mode #pf va vb vc tt x y;
    let r = ra_s mode (x / 16) in
    let c = cb_s mode (x / 16) in
    chain_apx #ma #kk #nb #() va vb xa xb r c (kk / 16);
    rchain_val ma kk nb #() ra rb r c (kk / 16) (x % 16) y

(* ===== K1 ===== *)


noextract
let scat_bound (w : pos) (q : natlt (w * 8)) : Lemma (q / w < 8 /\ (q % w) * 8 + q / w < w * 8) =
  ML.lemma_div_mod q w;
  if q / w >= 8 then ML.lemma_mult_le_left w 8 (q / w);
  ML.lemma_mult_le_right 8 (q % w) (w - 1)

noextract
let scat_i (w : pos) (q : natlt (w * 8)) : natlt (w * 8) =
  scat_bound w q;
  (q % w) * 8 + q / w

noextract
let scat_i_def (w : pos) (q : natlt (w * 8)) : Lemma (scat_i w q == (q % w) * 8 + q / w) = ()

noextract
let scat_f (w : pos) (sc : chest1 f32 (w * 8)) (j : natlt (w * 8)) : GTot bf16 = tob (acc1 sc (scat_i w j))

(* out[n*width + p] = bf16 (c[p*8 + n]) *)
inline_for_extraction noextract
fn k_scatter (width : szp{SZ.fits (width * 8)})
  (c : array1 f32 (l1_forward (width * 8))) (out : array1 bf16 (l1_forward (width * 8)))
  (#fc : perm) (#sc : erased (chest1 f32 (width * 8))) (#so : erased (chest1 bf16 (width * 8)))
  preserves gpu ** c |-> Frac fc sc
  requires out |-> so
  ensures exists* r. out |-> r **
    pure (forall (x:natlt (width * 8)). acc1 r x == scat_f width sc x)
{
  let mut q = 0sz;
  while (!q <^ width *^ 8sz)
    invariant live q ** pure (SZ.v !q <= width * 8)
    invariant exists* s. out |-> s **
      pure (forall (j:natlt (width * 8)). j < SZ.v !q ==> acc1 s j == scat_f width sc j)
    decreases (width * 8 - SZ.v !q)
  {
    let qv = !q;
    scat_i_def (SZ.v width) (SZ.v qv);
    scat_bound (SZ.v width) (SZ.v qv);
    mod_spec_lem (SZ.v qv) (SZ.v width);
    let i1 = (qv %^ width) *^ 8sz +^ qv /^ width;
    assert (pure (SZ.v i1 == scat_i width (SZ.v qv)));
    let x = rd c i1;
    with s0. assert (out |-> s0);
    inv_step s0 (SZ.v qv) (tob x) (scat_f width sc);
    wr out qv (tob x);
    q := qv +^ 1sz;
  };
}

noextract
let silu_off (q : natlt 24576) : natlt 49152 =
  (q / 3072) * 6144 + ((q % 3072) / 64) * 128 + (q % 3072) % 64

noextract
let silu_bound (q : natlt 24576)
  : Lemma (q / 3072 < 8 /\ q % 3072 < 3072 /\ (q % 3072) / 64 < 48 /\
           (q / 3072) * 6144 + ((q % 3072) / 64) * 128 + (q % 3072) % 64 + 64 < 49152)
  = ()

noextract
let silu_f (sg : chest1 bf16 49152) (q : natlt 24576) : GTot bf16 =
  let g = tof (acc1 sg (silu_off q)) in
  let a = tob (div g (add one (fexp (neg g)))) in
  let u = tof (acc1 sg (silu_off q + 64)) in
  if q / 3072 < 7 then tob (mul (tof a) u) else tob zero

(* out[r*3072+c] = r<7 ? bf16(bf16(g/(1+exp(-g))) * up) : 0 *)
inline_for_extraction noextract
fn k_silu (gu : array1 bf16 (l1_forward 49152)) (act : array1 bf16 (l1_forward 24576))
  (#fg : perm) (#sg : erased (chest1 bf16 49152)) (#so : erased (chest1 bf16 24576))
  preserves gpu ** gu |-> Frac fg sg
  requires act |-> so
  ensures exists* so'. act |-> so' ** pure (forall (x:natlt 24576). acc1 so' x == silu_f sg x)
{
  let mut q = 0sz;
  while (!q <^ 24576sz)
    invariant live q ** pure (SZ.v !q <= 24576)
    invariant exists* s. act |-> s ** pure (forall (j:natlt 24576). j < SZ.v !q ==> acc1 s j == silu_f sg j)
    decreases (24576 - SZ.v !q)
  {
    let qv = !q;
    ms qv 3072;
    silu_bound (SZ.v qv);
    let rv = qv /^ 3072sz;
    let cv = qv %^ 3072sz;
    ms cv 64;
    let off = rv *^ 6144sz +^ (cv /^ 64sz) *^ 128sz +^ cv %^ 64sz;
    let g = tof (rd gu off);
    let a = tob (div g (add one (fexp (neg g))));
    let u = tof (rd gu (off +^ 64sz));
    let o = (if rv <^ 7sz then tob (mul (tof a) u) else tob zero);
    with s0. assert (act |-> s0);
    inv_step s0 (SZ.v qv) o (silu_f sg);
    wr act qv o;
    q := qv +^ 1sz;
  };
}

noextract
let vt_src (i : natlt 65536{i % 64 < 7}) : natlt 32768 =
  (i % 64) * 4096 + ((i / 64) / 128) * 512 + 384 + (i / 64) % 128

noextract
let vt_f (sq : chest1 bf16 32768) (i : natlt 65536) : GTot bf16 =
  if i % 64 < 7 then acc1 sq (vt_src i) else tob zero

inline_for_extraction noextract
let vt_idx (iv : sz{SZ.v iv < 65536 /\ SZ.v iv % 64 < 7}) : (r:sz{SZ.v r == vt_src (SZ.v iv)}) =
  let hc = iv /^ 64sz in
  (iv %% 64sz) *^ 4096sz +^ (hc /^ 128sz) *^ 512sz +^ 384sz +^ hc %% 128sz

inline_for_extraction noextract
fn vt_val (qkv : array1 bf16 (l1_forward 32768)) (iv : sz{SZ.v iv < 65536})
  (#fq : perm) (#sq : erased (chest1 bf16 32768))
  preserves qkv |-> Frac fq sq
  returns v : bf16
  ensures pure (v == vt_f sq (SZ.v iv))
{
  let k = iv %% 64sz;
  if (k <^ 7sz) {
    rd qkv (vt_idx iv)
  } else {
    tob zero
  }
}

#push-options "--z3rlimit 80"
(* vT[(h*128+c)*64 + k] = k<7 ? qkv[k*4096 + h*512 + 384 + c] : 0 *)
inline_for_extraction noextract
fn k_vt (qkv : array1 bf16 (l1_forward 32768)) (vt : array1 bf16 (l1_forward 65536))
  (#fq : perm) (#sq : erased (chest1 bf16 32768)) (#so : erased (chest1 bf16 65536))
  preserves gpu ** qkv |-> Frac fq sq
  requires vt |-> so
  ensures exists* so'. vt |-> so' ** pure (forall (x:natlt 65536). acc1 so' x == vt_f sq x)
{
  let mut i = 0sz;
  while (!i <^ n64k)
    invariant live i ** pure (SZ.v !i <= 65536)
    invariant exists* s. vt |-> s ** pure (forall (j:natlt 65536). j < SZ.v !i ==> acc1 s j == vt_f sq j)
    decreases (65536 - SZ.v !i)
  {
    let iv = !i;
    let x = vt_val qkv iv;
    with s0. assert (vt |-> s0);
    inv_step s0 (SZ.v iv) x (vt_f sq);
    wr vt iv x;
    i := iv +^ 1sz;
  };
}

noextract
let pvs_i (i : natlt 14336) : natlt 16384 =
  let t = i / 2048 in
  let h = (i % 2048) / 256 in
  let p = (i % 256) / 128 in
  let c = i % 128 in
  ((h * 16 + c / 8) * 16 + (t * 2 + p)) * 8 + c % 8

noextract
let pvs_bound (i : natlt 14336)
  : Lemma (i / 2048 < 7 /\ (i % 2048) / 256 < 8 /\ (i % 256) / 128 < 2 /\ i % 128 < 128 /\
           (i % 128) / 8 < 16 /\ (i % 128) % 8 < 8)
  = ()

inline_for_extraction noextract
let pvs_idx (iv : sz{SZ.v iv < 14336}) : (r:sz{SZ.v r == pvs_i (SZ.v iv)}) =
  pvs_bound (SZ.v iv);
  let t = iv /^ 2048sz in
  let h = (iv %% 2048sz) /^ 256sz in
  let p = (iv %% 256sz) /^ 128sz in
  let c = iv %% 128sz in
  ((h *^ 16sz +^ c /^ 8sz) *^ 16sz +^ (t *^ 2sz +^ p)) *^ 8sz +^ c %% 8sz

noextract
let pvs_f (sc : chest1 f32 16384) (i : natlt 16384) : GTot bf16 =
  if i < 14336 then tob (acc1 sc (pvs_i i)) else tob zero

(* attended[(r/2)*2048 + h*256 + (r%2)*128 + c] = bf16(C[((h*16 + c/8)*16 + r)*8 + c%8]) for r<14; row 7 = 0 *)
inline_for_extraction noextract
fn k_pvs (cp : array1 f32 (l1_forward 16384)) (att : array1 bf16 (l1_forward 16384))
  (#fc : perm) (#sc : erased (chest1 f32 16384)) (#so : erased (chest1 bf16 16384))
  preserves gpu ** cp |-> Frac fc sc
  requires att |-> so
  ensures exists* so'. att |-> so' ** pure (forall (x:natlt 16384). acc1 so' x == pvs_f sc x)
{
  let mut i = 0sz;
  while (!i <^ 14336sz)
    invariant live i ** pure (SZ.v !i <= 14336)
    invariant exists* s. att |-> s ** pure (forall (j:natlt 16384). j < SZ.v !i ==> acc1 s j == pvs_f sc j)
    decreases (14336 - SZ.v !i)
  {
    let iv = !i;
    let x = rd cp (pvs_idx iv);
    with s0. assert (att |-> s0);
    inv_step s0 (SZ.v iv) (tob x) (pvs_f sc);
    wr att iv (tob x);
    i := iv +^ 1sz;
  };
  let mut j = 0sz;
  while (!j <^ 2048sz)
    invariant live j ** pure (SZ.v !j <= 2048)
    invariant exists* s. att |-> s ** pure (forall (q:natlt 16384). q < 14336 + SZ.v !j ==> acc1 s q == pvs_f sc q)
    decreases (2048 - SZ.v !j)
  {
    let jv = !j;
    with s0. assert (att |-> s0);
    inv_step s0 (14336 + SZ.v jv) (tob zero) (pvs_f sc);
    wr att (14336sz +^ jv) (tob zero);
    j := jv +^ 1sz;
  };
}
#pop-options

(* ===== K2 ===== *)

inline_for_extraction noextract
let scale : f32 = div one (sqrt (of_int 128L))

inline_for_extraction noextract
let maskf : f32 = neg (mul (of_int 255L) (mul (of_int 1152921504606846976L) (of_int 1152921504606846976L)))

inline_for_extraction noextract
let ninf : f32 = neg infinity

inline_for_extraction noextract
fn mk_mask ()
  preserves gpu
  returns m : f32
  ensures pure (m == maskf)
{
  let mut a : f32 = of_int 255L;
  let mut b : f32 = of_int 1152921504606846976L;
  let av = !a;
  let bv = !b;
  neg (mul av (mul bv bv))
}

inline_for_extraction noextract
let scv (mk : f32) (x : f32) (r k : sz) : f32 =
  if k <^ 7sz && r <^ 14sz then
    tof (tob (add (tof (tob (mul (tof (tob x)) scale))) (if k >^ r /^ 2sz then mk else zero)))
  else ninf

noextract
let scale_apx () : Lemma (scale %~ attn_scale) =
  zo_apx ();
  oi 128L;
  let _ = approx_sqrt #f32 in
  let _ = approx_div #f32 in
  let c : f32 = of_int 128L in
  assert (c %~ 128.0R);
  assert (sqrt c %~ Sqrt.sqrt 128.0R);
  assert (div one (sqrt c) %~ (1.0R /. Sqrt.sqrt 128.0R))

noextract
let mask_apx () : Lemma (maskf %~ mask_min) =
  oi 255L;
  oi 1152921504606846976L;
  let _ = approx_neg #f32 in
  let a : f32 = of_int 255L in
  let b : f32 = of_int 1152921504606846976L in
  assert (a %~ 255.0R);
  assert (b %~ 1152921504606846976.0R);
  assert (mul b b %~ (1152921504606846976.0R *. 1152921504606846976.0R));
  assert (mul a (mul b b) %~ (255.0R *. (1152921504606846976.0R *. 1152921504606846976.0R)));
  assert (neg (mul a (mul b b)) %~ (0.0R -. 255.0R *. (1152921504606846976.0R *. 1152921504606846976.0R)))

(* ideal scores, softmax over the 7 valid keys of row rr = h*16 + r *)
noextract
let sc_r (rc : S.seq real) (rr k : nat) : real =
  at rc (rr * 8 + k) *. attn_scale +. (if k > (rr % 16) / 2 then mask_min else 0.0R)

noextract
let scs (rc : S.seq real) (rr : nat) : S.seq real = S.init 7 (fun (j:nat{j < 7}) -> sc_r rc rr j)

noextract
let sm_r (rc : S.seq real) (rr k : nat) : real =
  if k < 7 && rr % 16 < 14 then Exp.exp (sc_r rc rr k) /. sumexp 7 (scs rc rr) 0 else 0.0R

noextract
let sm_seq (rc : S.seq real) : S.seq real = S.init 8192 (fun (i:nat{i < 8192}) -> sm_r rc (i / 64) (i % 64))

noextract
let sk_ok (sc : chest1 f32 1024) (rr : nat) (k : nat) (s : f32) : prop =
  forall (rc : S.seq real). capx sc rc ==> s %~ sc_r rc rr k

noextract
let sk_lem (sc : chest1 f32 1024) (rr : natlt 128) (mk x : f32) (r k : sz)
  : Lemma (requires mk == maskf /\ SZ.v r == rr % 16 /\ SZ.v k < 7 /\
                    x == acc1 sc (rr * 8 + SZ.v k))
          (ensures rr % 16 < 14 ==> sk_ok sc rr (SZ.v k) (scv mk x r k))
  = if rr % 16 < 14 then
    let aux (rc : S.seq real) : Lemma (requires capx sc rc) (ensures scv mk x r k %~ sc_r rc rr (SZ.v k)) =
      zo_apx ();
      scale_apx ();
      mask_apx ();
      let xr = at rc (rr * 8 + SZ.v k) in
      assert (x %~ xr);
      let m = (if k >^ r /^ 2sz then mk else zero) in
      let mr = (if SZ.v k > (rr % 16) / 2 then mask_min else 0.0R) in
      assert (m %~ mr);
      assert (tof (tob x) %~ xr);
      assert (mul (tof (tob x)) scale %~ (xr *. attn_scale));
      assert (tof (tob (mul (tof (tob x)) scale)) %~ (xr *. attn_scale));
      assert (add (tof (tob (mul (tof (tob x)) scale))) m %~ (xr *. attn_scale +. mr));
      assert (tof (tob (add (tof (tob (mul (tof (tob x)) scale))) m)) %~ (xr *. attn_scale +. mr))
    in
    Classical.forall_intro (Classical.move_requires aux)
    else ()

noextract
let exp_ok (d e : f32) : prop = forall (xr : real). d %~ xr ==> e %~ r_exp xr

noextract
let div_ok (x y r : f32) : prop =
  forall (xr : real) (yr : real{yr =!= 0.0R}). x %~ xr /\ y %~ yr ==> r %~ r_div xr yr

inline_for_extraction noextract
fn expv (valid : bool) (s shift : f32)
  preserves gpu
  returns e : f32
  ensures pure ((valid ==> exp_ok (Fast.pure_sub_rn s shift) e) /\ (not valid ==> e == zero))
{
  if valid {
    let d = Fast.sub_rn s shift;
    Fast.exp d
  } else {
    zero
  }
}

noextract
let el_lem (s shift e : f32) (sr : real)
  : Lemma (requires s %~ sr /\ exp_ok (Fast.pure_sub_rn s shift) e)
          (ensures e %~ (Exp.exp sr /. Exp.exp (to_real shift)))
  = let c = to_real shift in
    assert (shift %~ c);
    assert (Fast.pure_sub_rn s shift %~ (sr -. c));
    assert (e %~ Exp.exp (sr -. c))

#push-options "--fuel 8"
noextract
let sumexp7 (a : S.seq real)
  : Lemma (sumexp 7 a 0 == 0.0R +. Exp.exp (at a 0) +. Exp.exp (at a 1) +. Exp.exp (at a 2) +.
                           Exp.exp (at a 3) +. Exp.exp (at a 4) +. Exp.exp (at a 5) +. Exp.exp (at a 6))
  = ()
#pop-options

noextract
let sum_div7 (a0 a1 a2 a3 a4 a5 a6 x : real)
  : Lemma (requires x >. 0.0R)
          (ensures ((a0 /. x +. a1 /. x) +. (a2 /. x +. a3 /. x)) +. ((a4 /. x +. a5 /. x) +. (a6 /. x +. 0.0R)) ==
                   (0.0R +. a0 +. a1 +. a2 +. a3 +. a4 +. a5 +. a6) /. x)
  = ()

noextract
let cancel (a s e : real)
  : Lemma (requires s >. 0.0R /\ e >. 0.0R) (ensures (a /. e) *. (1.0R /. (s /. e)) == a /. s)
  = ()

noextract
let wv (vr : bool) (e inv : f32) : bf16 = if vr then tob (mul e inv) else tob zero

noextract
let row_ok_b (sc : chest1 f32 1024) (s : chest1 bf16 8192) (rr : natlt 128) (km : nat) : prop =
  forall (rc : S.seq real). capx sc rc ==>
    (forall (k:natlt 64). k < km ==> acc1 s (rr * 64 + k) %~ sm_r rc rr k)

[@@"opaque_to_smt"] noextract
let row_ok (sc : chest1 f32 1024) (s : chest1 bf16 8192) (rr : natlt 128) (km : nat) : prop = row_ok_b sc s rr km

noextract
let row_ok_i (sc : chest1 f32 1024) (s : chest1 bf16 8192) (rr : natlt 128) (km : nat)
  : Lemma (requires row_ok_b sc s rr km) (ensures row_ok sc s rr km)
  = reveal_opaque (`%row_ok) (row_ok sc s rr km)

noextract
let row_ok_64 (sc : chest1 f32 1024) (s : chest1 bf16 8192) (rr : natlt 128) (km : nat)
  : Lemma (requires row_ok sc s rr km /\ km == 64) (ensures row_ok sc s rr 64)
  = ()

noextract
let row_ok_e (sc : chest1 f32 1024) (s : chest1 bf16 8192) (rr : natlt 128) (km : nat)
  : Lemma (requires row_ok sc s rr km) (ensures row_ok_b sc s rr km)
  = reveal_opaque (`%row_ok) (row_ok sc s rr km)

noextract
let fr (s s' : chest1 bf16 8192) (rr : natlt 128) : prop =
  forall (j:natlt 8192). (j < rr * 64 \/ j >= rr * 64 + 64) ==> acc1 s' j == acc1 s j

noextract
let fr_w1 (so s : chest1 bf16 8192) (rr : natlt 128) (o j : nat) (w : bf16)
  : Lemma (requires fr so s rr /\ o == rr * 64 /\ j < 64)
          (ensures fr so (upd s (o + j, ()) w) rr)
  = ()

noextract
let fr_w7 (so : chest1 bf16 8192) (rr : natlt 128) (o : nat) (w0 w1 w2 w3 w4 w5 w6 : bf16)
  : Lemma (requires o == rr * 64)
          (ensures fr so (upd (upd (upd (upd (upd (upd (upd so (o, ()) w0) (o + 1, ()) w1) (o + 2, ()) w2)
                     (o + 3, ()) w3) (o + 4, ()) w4) (o + 5, ()) w5) (o + 6, ()) w6) rr)
  = let s1 = upd so (o + 0, ()) w0 in
    fr_w1 so so rr o 0 w0;
    let s2 = upd s1 (o + 1, ()) w1 in
    fr_w1 so s1 rr o 1 w1;
    let s3 = upd s2 (o + 2, ()) w2 in
    fr_w1 so s2 rr o 2 w2;
    let s4 = upd s3 (o + 3, ()) w3 in
    fr_w1 so s3 rr o 3 w3;
    let s5 = upd s4 (o + 4, ()) w4 in
    fr_w1 so s4 rr o 4 w4;
    let s6 = upd s5 (o + 5, ()) w5 in
    fr_w1 so s5 rr o 5 w5;
    fr_w1 so s6 rr o 6 w6

#push-options "--z3rlimit 60"
noextract
let sm_lem (sc : chest1 f32 1024) (s : chest1 bf16 8192) (rr : natlt 128) (shift : f32)
  (s0 s1 s2 s3 s4 s5 s6 : f32) (e0 e1 e2 e3 e4 e5 e6 inv : f32)
  : Lemma (requires
             (rr % 16 < 14 ==>
               (sk_ok sc rr 0 s0 /\ sk_ok sc rr 1 s1 /\ sk_ok sc rr 2 s2 /\ sk_ok sc rr 3 s3 /\
                sk_ok sc rr 4 s4 /\ sk_ok sc rr 5 s5 /\ sk_ok sc rr 6 s6 /\
                exp_ok (Fast.pure_sub_rn s0 shift) e0 /\ exp_ok (Fast.pure_sub_rn s1 shift) e1 /\
                exp_ok (Fast.pure_sub_rn s2 shift) e2 /\ exp_ok (Fast.pure_sub_rn s3 shift) e3 /\
                exp_ok (Fast.pure_sub_rn s4 shift) e4 /\ exp_ok (Fast.pure_sub_rn s5 shift) e5 /\
                exp_ok (Fast.pure_sub_rn s6 shift) e6 /\
                div_ok one (add (add (add e0 e1) (add e2 e3)) (add (add e4 e5) (add e6 zero))) inv)) /\
             (let vr = rr % 16 < 14 in
              acc1 s (rr * 64) == wv vr e0 inv /\ acc1 s (rr * 64 + 1) == wv vr e1 inv /\
              acc1 s (rr * 64 + 2) == wv vr e2 inv /\ acc1 s (rr * 64 + 3) == wv vr e3 inv /\
              acc1 s (rr * 64 + 4) == wv vr e4 inv /\ acc1 s (rr * 64 + 5) == wv vr e5 inv /\
              acc1 s (rr * 64 + 6) == wv vr e6 inv))
          (ensures row_ok_b sc s rr 7)
  = tobz_apx ();
    if rr % 16 < 14 then begin
      let aux (rc : S.seq real) : Lemma (requires capx sc rc)
          (ensures forall (k:natlt 64). k < 7 ==> acc1 s (rr * 64 + k) %~ sm_r rc rr k) =
        zo_apx ();
        let c = to_real shift in
        let x = Exp.exp c in
        let g0 = Exp.exp (sc_r rc rr 0) in
        let g1 = Exp.exp (sc_r rc rr 1) in
        let g2 = Exp.exp (sc_r rc rr 2) in
        let g3 = Exp.exp (sc_r rc rr 3) in
        let g4 = Exp.exp (sc_r rc rr 4) in
        let g5 = Exp.exp (sc_r rc rr 5) in
        let g6 = Exp.exp (sc_r rc rr 6) in
        el_lem s0 shift e0 (sc_r rc rr 0);
        el_lem s1 shift e1 (sc_r rc rr 1);
        el_lem s2 shift e2 (sc_r rc rr 2);
        el_lem s3 shift e3 (sc_r rc rr 3);
        el_lem s4 shift e4 (sc_r rc rr 4);
        el_lem s5 shift e5 (sc_r rc rr 5);
        el_lem s6 shift e6 (sc_r rc rr 6);
        let sum = add (add (add e0 e1) (add e2 e3)) (add (add e4 e5) (add e6 zero)) in
        let tr = ((g0 /. x +. g1 /. x) +. (g2 /. x +. g3 /. x)) +. ((g4 /. x +. g5 /. x) +. (g6 /. x +. 0.0R)) in
        assert (sum %~ tr);
        let sr = scs rc rr in
        sumexp7 sr;
        let se = sumexp 7 sr 0 in
        assert (se == 0.0R +. g0 +. g1 +. g2 +. g3 +. g4 +. g5 +. g6);
        sum_div7 g0 g1 g2 g3 g4 g5 g6 x;
        assert (tr == se /. x);
        assert (inv %~ (1.0R /. tr));
        cancel g0 se x; cancel g1 se x; cancel g2 se x; cancel g3 se x;
        cancel g4 se x; cancel g5 se x; cancel g6 se x;
        assert (mul e0 inv %~ (g0 /. se));
        assert (mul e1 inv %~ (g1 /. se));
        assert (mul e2 inv %~ (g2 /. se));
        assert (mul e3 inv %~ (g3 /. se));
        assert (mul e4 inv %~ (g4 /. se));
        assert (mul e5 inv %~ (g5 /. se));
        assert (mul e6 inv %~ (g6 /. se))
      in
      Classical.forall_intro (Classical.move_requires aux)
    end else ()
#pop-options

noextract
let row_step (sc : chest1 f32 1024) (so s : chest1 bf16 8192) (rr : natlt 128) (kv : nat)
  : Lemma (requires row_ok_b sc s rr kv /\ fr so s rr /\ 7 <= kv /\ kv < 64)
          (ensures row_ok_b sc (upd s (rr * 64 + kv, ()) (tob zero)) rr (kv + 1) /\
                   fr so (upd s (rr * 64 + kv, ()) (tob zero)) rr)
  = tobz_apx ();
    let s' = upd s (rr * 64 + kv, ()) (tob zero) in
    let aux (rc : S.seq real) : Lemma (requires capx sc rc)
        (ensures forall (k:natlt 64). k < kv + 1 ==> acc1 s' (rr * 64 + k) %~ sm_r rc rr k) =
      let a2 (k : natlt 64) : Lemma (k < kv + 1 ==> acc1 s' (rr * 64 + k) %~ sm_r rc rr k) =
        if k < kv then begin
          assert (acc1 s (rr * 64 + k) %~ sm_r rc rr k);
          assert (acc1 s' (rr * 64 + k) == acc1 s (rr * 64 + k))
        end
        else if k = kv then assert (acc1 s' (rr * 64 + k) == tob zero)
        else ()
      in
      Classical.forall_intro a2
    in
    Classical.forall_intro (Classical.move_requires aux);
    let a3 (j : natlt 8192) : Lemma ((j < rr * 64 \/ j >= rr * 64 + 64) ==> acc1 s' j == acc1 so j) = () in
    Classical.forall_intro a3

noextract
let outer_aux (sc : chest1 f32 1024) (s s' : chest1 bf16 8192) (i : natlt 128) (q : natlt 128) (rc : S.seq real) (k : natlt 64)
  : Lemma (requires q < i /\ row_ok sc s q 64 /\ fr s s' i /\ capx sc rc)
          (ensures acc1 s' (q * 64 + k) %~ sm_r rc q k)
  = row_ok_e sc s q 64;
    ML.lemma_mult_le_right 64 (q + 1) i;
    assert (acc1 s' (q * 64 + k) == acc1 s (q * 64 + k))

noextract
let outer_last (sc : chest1 f32 1024) (s' : chest1 bf16 8192) (i q : natlt 128)
  : Lemma (requires q < i + 1 /\ ~(q < i) /\ row_ok sc s' i 64) (ensures row_ok sc s' q 64)
  = assert (q == i)

noextract
let outer_step (sc : chest1 f32 1024) (s s' : chest1 bf16 8192) (i : natlt 128)
  : Lemma (requires (forall (q:natlt 128). q < i ==> row_ok sc s q 64) /\ fr s s' i /\ row_ok sc s' i 64)
          (ensures forall (q:natlt 128). q < i + 1 ==> row_ok sc s' q 64)
  = let fin (q : natlt 128) : Lemma (q < i + 1 ==> row_ok sc s' q 64) =
      if q < i then begin
        let a1 (rc : S.seq real) (k : natlt 64) : Lemma (capx sc rc ==> acc1 s' (q * 64 + k) %~ sm_r rc q k) =
          Classical.move_requires (outer_aux sc s s' i q rc) k
        in
        Classical.forall_intro_2 a1;
        row_ok_i sc s' q 64
      end else Classical.move_requires (outer_last sc s' i) q
    in
    Classical.forall_intro fin

noextract
let sm_post (sc : chest1 f32 1024) (s : chest1 bf16 8192) : prop =
  forall (rc : S.seq real). capx sc rc ==> capx s (sm_seq rc)

noextract
let sm_fin (sc : chest1 f32 1024) (s : chest1 bf16 8192) (n : nat)
  : Lemma (requires n >= 128 /\ (forall (q:natlt 128). q < n ==> row_ok sc s q 64)) (ensures sm_post sc s)
  = let aux (rc : S.seq real) : Lemma (requires capx sc rc) (ensures capx s (sm_seq rc)) =
      let el (i : natlt 8192) : Lemma (acc1 s i %~ at (sm_seq rc) i) =
        ML.lemma_div_mod i 64;
        assert (row_ok sc s (i / 64) 64);
        row_ok_e sc s (i / 64) 64
      in
      Classical.forall_intro el
    in
    Classical.forall_intro (Classical.move_requires aux)

inline_for_extraction noextract
fn sm_head (cq : array1 f32 (l1_forward 1024)) (pr : array1 bf16 (l1_forward 8192)) (rr : sz{SZ.v rr < 128})
  (#fc : perm) (#sc : erased (chest1 f32 1024)) (#so : erased (chest1 bf16 8192))
  preserves gpu ** cq |-> Frac fc sc
  requires pr |-> so
  ensures exists* s'. pr |-> s' ** pure (fr so s' (SZ.v rr) /\ row_ok sc s' (SZ.v rr) 7)
{
  let r = rr %% 16sz;
  let b = rr *^ 8sz;
  let mk = mk_mask ();
  let x0 = rd cq b;
  let s0 = scv mk x0 r 0sz;
  let x1 = rd cq (b +^ 1sz);
  let s1 = scv mk x1 r 1sz;
  let x2 = rd cq (b +^ 2sz);
  let s2 = scv mk x2 r 2sz;
  let x3 = rd cq (b +^ 3sz);
  let s3 = scv mk x3 r 3sz;
  let x4 = rd cq (b +^ 4sz);
  let s4 = scv mk x4 r 4sz;
  let x5 = rd cq (b +^ 5sz);
  let s5 = scv mk x5 r 5sz;
  let x6 = rd cq (b +^ 6sz);
  let s6 = scv mk x6 r 6sz;
  let x7 = rd cq (b +^ 7sz);
  let s7 = scv mk x7 r 7sz;
  let mx = fmax (fmax (fmax (fmax (fmax (fmax (fmax (fmax ninf s0) s1) s2) s3) s4) s5) s6) s7;
  let shift = (if ieee_eq mx ninf then fz else mx);
  let vr = r <^ 14sz;
  sk_lem sc (SZ.v rr) mk x0 r 0sz;
  sk_lem sc (SZ.v rr) mk x1 r 1sz;
  sk_lem sc (SZ.v rr) mk x2 r 2sz;
  sk_lem sc (SZ.v rr) mk x3 r 3sz;
  sk_lem sc (SZ.v rr) mk x4 r 4sz;
  sk_lem sc (SZ.v rr) mk x5 r 5sz;
  sk_lem sc (SZ.v rr) mk x6 r 6sz;
  let e0 = expv vr s0 shift;
  let e1 = expv vr s1 shift;
  let e2 = expv vr s2 shift;
  let e3 = expv vr s3 shift;
  let e4 = expv vr s4 shift;
  let e5 = expv vr s5 shift;
  let e6 = expv vr s6 shift;
  let e7 = expv false s7 shift;
  let sum = add (add (add e0 e1) (add e2 e3)) (add (add e4 e5) (add e6 e7));
  let inv = Fast.divide one sum;
  let o = rr *^ 64sz;
  wr pr o (if vr then tob (mul e0 inv) else tob zero);
  wr pr (o +^ 1sz) (if vr then tob (mul e1 inv) else tob zero);
  wr pr (o +^ 2sz) (if vr then tob (mul e2 inv) else tob zero);
  wr pr (o +^ 3sz) (if vr then tob (mul e3 inv) else tob zero);
  wr pr (o +^ 4sz) (if vr then tob (mul e4 inv) else tob zero);
  wr pr (o +^ 5sz) (if vr then tob (mul e5 inv) else tob zero);
  wr pr (o +^ 6sz) (if vr then tob (mul e6 inv) else tob zero);
  with sw. assert (pr |-> sw);
  sm_lem sc sw (SZ.v rr) shift s0 s1 s2 s3 s4 s5 s6 e0 e1 e2 e3 e4 e5 e6 inv;
  row_ok_i sc sw (SZ.v rr) 7;
  fr_w7 so (SZ.v rr) (SZ.v o) (if vr then tob (mul e0 inv) else tob zero) (if vr then tob (mul e1 inv) else tob zero)
    (if vr then tob (mul e2 inv) else tob zero) (if vr then tob (mul e3 inv) else tob zero)
    (if vr then tob (mul e4 inv) else tob zero) (if vr then tob (mul e5 inv) else tob zero)
    (if vr then tob (mul e6 inv) else tob zero);
}

inline_for_extraction noextract
fn sm_tail (pr : array1 bf16 (l1_forward 8192)) (rr : sz{SZ.v rr < 128})
  (#sc : erased (chest1 f32 1024)) (#so #s0 : erased (chest1 bf16 8192))
  preserves gpu
  requires pr |-> s0 ** pure (fr so s0 (SZ.v rr) /\ row_ok sc s0 (SZ.v rr) 7)
  ensures exists* s'. pr |-> s' ** pure (fr so s' (SZ.v rr) /\ row_ok sc s' (SZ.v rr) 64)
{
  let o = rr *^ 64sz;
  let mut k = 7sz;
  while (!k <^ 64sz)
    invariant live k ** pure (7 <= SZ.v !k /\ SZ.v !k <= 64)
    invariant exists* s. pr |-> s ** pure (fr so s (SZ.v rr) /\ row_ok sc s (SZ.v rr) (SZ.v !k))
    decreases (64 - SZ.v !k)
  {
    let kv = !k;
    with sx. assert (pr |-> sx);
    row_ok_e sc sx (SZ.v rr) (SZ.v kv);
    row_step sc so sx (SZ.v rr) (SZ.v kv);
    wr pr (o +^ kv) (tob zero);
    with sy. assert (pr |-> sy);
    row_ok_i sc sy (SZ.v rr) (SZ.v kv + 1);
    k := kv +^ 1sz;
  };
  with sz. assert (pr |-> sz);
  let kf = !k;
  assert pure (SZ.v kf == 64);
  row_ok_64 sc sz (SZ.v rr) (SZ.v kf);
}

inline_for_extraction noextract
fn sm_row (cq : array1 f32 (l1_forward 1024)) (pr : array1 bf16 (l1_forward 8192)) (rr : sz{SZ.v rr < 128})
  (#fc : perm) (#sc : erased (chest1 f32 1024)) (#so : erased (chest1 bf16 8192))
  preserves gpu ** cq |-> Frac fc sc
  requires pr |-> so
  ensures exists* s'. pr |-> s' ** pure (fr so s' (SZ.v rr) /\ row_ok sc s' (SZ.v rr) 64)
{
  sm_head cq pr rr;
  sm_tail pr rr #sc #so;
}

#push-options "--z3rlimit 80"
inline_for_extraction noextract
fn k_softmax (cq : array1 f32 (l1_forward 1024)) (pr : array1 bf16 (l1_forward 8192))
  (#fc : perm) (#sc : erased (chest1 f32 1024)) (#so : erased (chest1 bf16 8192))
  preserves gpu ** cq |-> Frac fc sc
  requires pr |-> so
  ensures exists* so'. pr |-> so' ** pure (sm_post sc so')
{
  let mut i = 0sz;
  while (!i <^ 128sz)
    invariant live i ** pure (SZ.v !i <= 128)
    invariant exists* s. pr |-> s ** pure (forall (q:natlt 128). q < SZ.v !i ==> row_ok sc s q 64)
    decreases (128 - SZ.v !i)
  {
    let rr = !i;
    with s0. assert (pr |-> s0);
    sm_row cq pr rr;
    with s1. assert (pr |-> s1);
    outer_step sc s0 s1 (SZ.v rr);
    i := rr +^ 1sz;
  };
  with s. assert (pr |-> s);
  let iv = !i;
  assert pure (SZ.v iv >= 128);
  assert pure (forall (q:natlt 128). q < SZ.v iv ==> row_ok sc s q 64);
  sm_fin sc s (SZ.v iv);
  assert pure (sm_post sc s);
}
#pop-options

(* ===== K3 ===== *)

noextract
let lvl_t (#m:nat) (s : chest1 f32 m) (o d t : nat) (j : natlt m) : GTot f32 =
  if o <= j && j < o + t && j + d < m then add (acc1 s j) (acc1 s (j + d)) else acc1 s j

noextract
let lvl_f (#m:nat) (s : chest1 f32 m) (o d : nat) (j : natlt m) : GTot f32 = lvl_t s o d d j

noextract
let lvl (#m:nat) (s : chest1 f32 m) (o d : nat) : chest1 f32 m = mk1 (lvl_f s o d)

noextract
let rec tree_spec (#m:nat) (s : chest1 f32 m) (o d : nat) : Tot (chest1 f32 m) (decreases d) =
  if d = 0 then s else tree_spec (lvl s o d) o (d / 2)

noextract
let rec tree_frame (#m:nat) (s : chest1 f32 m) (o d : nat) (j : natlt m)
  : Lemma (requires j < o \/ j >= o + d) (ensures acc1 (tree_spec s o d) j == acc1 s j) (decreases d)
  = if d = 0 then () else tree_frame (lvl s o d) o (d / 2) j

noextract
let tree_unf (#m:nat) (s : chest1 f32 m) (o d : nat)
  : Lemma (requires d > 0) (ensures tree_spec s o d == tree_spec (lvl s o d) o (d / 2)) = ()

(* in-place tree reduction: for d = n/2 .. 1: a[o+t] += a[o+t+d], t<d *)
inline_for_extraction noextract
fn tree_lvl (#m : erased nat{SZ.fits m}) (a : array1 f32 (l1_forward m)) (o : sz) (dv : sz{SZ.v o + 2 * SZ.v dv <= m})
  (#s : erased (chest1 f32 m))
  requires a |-> s
  ensures a |-> lvl s (SZ.v o) (SZ.v dv)
{
  let mut t : sz = 0sz;
  while (!t <^ dv)
    invariant live t ** pure (SZ.v !t <= SZ.v dv)
    invariant exists* s1. a |-> s1 **
      pure (forall (j:natlt m). acc1 s1 j == lvl_t s (SZ.v o) (SZ.v dv) (SZ.v !t) j)
    decreases (SZ.v dv - SZ.v !t)
  {
    let tv = !t;
    let x = rd a (o +^ tv);
    let y = rd a (o +^ tv +^ dv);
    wr a (o +^ tv) (add x y);
    t := tv +^ 1sz;
  };
  with s1. assert (a |-> s1);
  assert pure (equal s1 (lvl s (SZ.v o) (SZ.v dv)));
  rewrite (a |-> s1) as (a |-> lvl s (SZ.v o) (SZ.v dv));
}

inline_for_extraction noextract
fn tree_red (#m : erased nat{SZ.fits m}) (a : array1 f32 (l1_forward m)) (o : sz) (n : szp{SZ.v o + SZ.v n <= m})
  (#s : erased (chest1 f32 m))
  requires a |-> s
  ensures a |-> tree_spec s (SZ.v o) (SZ.v n / 2)
{
  let mut d : sz = n /^ 2sz;
  while (!d >^ 0sz)
    invariant live d ** pure (2 * SZ.v !d <= SZ.v n)
    invariant exists* s1. a |-> s1 ** pure (tree_spec s1 (SZ.v o) (SZ.v !d) == tree_spec s (SZ.v o) (SZ.v n / 2))
    decreases (SZ.v !d)
  {
    let dv = !d;
    with s1. assert (a |-> s1);
    tree_unf s1 (SZ.v o) (SZ.v dv);
    tree_lvl a o dv;
    d := dv /^ 2sz;
  };
  with s1. assert (a |-> s1);
  rewrite (a |-> s1) as (a |-> tree_spec s (SZ.v o) (SZ.v n / 2));
}

(* real sums: rs g m t st = sum_{i<m} g (t + i*st) *)
noextract
let rec rs (g : nat -> GTot real) (m : nat) (t st : nat) : GTot real (decreases m) =
  if m = 0 then 0.0R else rs g (m - 1) t st +. g (t + (m - 1) * st)

noextract
let rec rs_split (g : nat -> GTot real) (m : nat) (t d : nat)
  : Lemma (ensures rs g (2 * m) t d == rs g m t (2 * d) +. rs g m (t + d) (2 * d)) (decreases m)
  = if m = 0 then ()
    else begin
      rs_split g (m - 1) t d;
      assert (rs g (2 * m) t d == rs g (2 * m - 1) t d +. g (t + (2 * m - 1) * d));
      assert (rs g (2 * m - 1) t d == rs g (2 * (m - 1)) t d +. g (t + (2 * (m - 1)) * d));
      ML.paren_mul_right (m - 1) 2 d;
      ML.paren_mul_left 2 (m - 1) d;
      assert ((m - 1) * (2 * d) == 2 * (m - 1) * d);
      assert (t + d + (m - 1) * (2 * d) == t + (2 * m - 1) * d)
    end

noextract
let rec tree_apx_k (#n:nat) (k : nat) (s : chest1 f32 n) (o : nat) (g : nat -> GTot real) (go : nat) (m : nat)
  : Lemma (requires o + 2 * pow2 k <= n /\
                    (forall (t:nat). t < 2 * pow2 k ==> acc1 s (o + t) %~ rs g m (go + t) (2 * pow2 k)))
          (ensures acc1 (tree_spec s o (pow2 k)) o %~ rs g (2 * pow2 k * m) go 1)
          (decreases k)
  = let d = pow2 k in
    let s' = lvl s o d in
    let aux (t:nat{t < d}) : Lemma (acc1 s' (o + t) %~ rs g (2 * m) (go + t) d) =
      rs_split g m (go + t) d;
      assert (acc1 s' (o + t) == add (acc1 s (o + t)) (acc1 s (o + (t + d))));
      let x = acc1 s (o + t) in
      let y = acc1 s (o + (t + d)) in
      assert (x %~ rs g m (go + t) (2 * d));
      assert (y %~ rs g m (go + (t + d)) (2 * d));
      assert (go + (t + d) == go + t + d);
      assert (add x y %~ (rs g m (go + t) (2 * d) +. rs g m (go + t + d) (2 * d)))
    in
    FStar.Classical.forall_intro aux;
    if k = 0 then begin
      assert (tree_spec s o d == tree_spec s' o 0)
    end else begin
      FStar.Math.Lemmas.pow2_double_mult (k - 1);
      assert (d / 2 == pow2 (k - 1));
      tree_apx_k (k - 1) s' o g go (2 * m);
      assert (tree_spec s o d == tree_spec s' o (pow2 (k - 1)));
      assert (2 * pow2 (k - 1) * (2 * m) == 2 * d * m)
    end

noextract
let tree_apx (#n:nat) (k : nat) (s : chest1 f32 n) (o : nat) (g : nat -> GTot real) (go : nat)
  : Lemma (requires o + 2 * pow2 k <= n /\
                    (forall (t:nat). t < 2 * pow2 k ==> acc1 s (o + t) %~ g (go + t)))
          (ensures acc1 (tree_spec s o (pow2 k)) o %~ rs g (2 * pow2 k) go 1)
  = let aux (t:nat{t < 2 * pow2 k}) : Lemma (rs g 1 (go + t) (2 * pow2 k) == g (go + t)) =
      assert (rs g 1 (go + t) (2 * pow2 k) == 0.0R +. g (go + t + 0 * (2 * pow2 k)))
    in
    FStar.Classical.forall_intro aux;
    tree_apx_k k s o g go 1

noextract
let rec rs_cat (g : nat -> GTot real) (a c t : nat)
  : Lemma (ensures rs g (a + c) t 1 == rs g a t 1 +. rs g c (t + a) 1) (decreases c)
  = if c = 0 then () else rs_cat g a (c - 1) t

noextract
let blk (g : nat -> GTot real) (k b : nat) (w : nat) : GTot real = rs g k (b + k * w) 1

noextract
let rec rs_blk (g : nat -> GTot real) (k mm b : nat)
  : Lemma (ensures rs g (mm * k) b 1 == rs (blk g k b) mm 0 1) (decreases mm)
  = if mm = 0 then ()
    else begin
      rs_blk g k (mm - 1) b;
      ML.distributivity_sub_left mm 1 k;
      rs_cat g ((mm - 1) * k) k b;
      assert (blk g k b (0 + (mm - 1) * 1) == rs g k (b + k * (mm - 1)) 1)
    end

noextract
let sq (a : S.seq real) (o : int) (j : nat) : GTot real = at a (o + j) *. at a (o + j)

noextract
let rec sumsq_rs (n : nat) (a : S.seq real) (o : int)
  : Lemma (ensures sumsq n a o == rs (sq a o) n 0 1) (decreases n)
  = if n = 0 then () else sumsq_rs (n - 1) a o

(* fma-nested 4-term partial: x3^2 + (x2^2 + (x1^2 + (x0^2 + 0))) with x_k = a[o+j+k*st] *)
noextract
let pq4 (a : S.seq real) (o : int) (st : nat) (j : nat) : GTot real =
  at a (o + j + 3 * st) *. at a (o + j + 3 * st) +.
  (at a (o + j + 2 * st) *. at a (o + j + 2 * st) +.
   (at a (o + j + st) *. at a (o + j + st) +.
    (at a (o + j) *. at a (o + j) +. 0.0R)))

noextract
let rec rs_pq4 (m : nat) (a : S.seq real) (o : int) (st : nat)
  : Lemma (ensures rs (pq4 a o st) m 0 1 ==
                   rs (sq a o) m 0 1 +. rs (sq a o) m st 1 +. rs (sq a o) m (2 * st) 1 +. rs (sq a o) m (3 * st) 1)
          (decreases m)
  = if m = 0 then ()
    else begin
      rs_pq4 (m - 1) a o st;
      let j = m - 1 in
      assert (sq a o (st + j * 1) == at a (o + j + st) *. at a (o + j + st));
      assert (sq a o (2 * st + j * 1) == at a (o + j + 2 * st) *. at a (o + j + 2 * st));
      assert (sq a o (3 * st + j * 1) == at a (o + j + 3 * st) *. at a (o + j + 3 * st));
      assert (sq a o (0 + j * 1) == at a (o + j) *. at a (o + j))
    end

noextract
let sumsq_pq4 (a : S.seq real) (o : int) (st : nat)
  : Lemma (rs (pq4 a o st) st 0 1 == sumsq (4 * st) a o)
  = rs_pq4 st a o st;
    sumsq_rs (4 * st) a o;
    rs_cat (sq a o) (3 * st) st 0;
    rs_cat (sq a o) (2 * st) st 0;
    rs_cat (sq a o) st st 0

noextract
let p16 () : Lemma (pow2 4 == 16 /\ pow2 2 == 4) =
  assert (pow2 0 == 1); assert (pow2 1 == 2); assert (pow2 2 == 4); assert (pow2 3 == 8); assert (pow2 4 == 16)

(* ===== K4 ===== *)

noextract
let nres_f (sp sr : chest1 bf16 8192) (i : nat) : GTot bf16 =
  if i < 8192 then tob (add (tof (acc1 sp i)) (tof (acc1 sr i))) else tob zero

noextract
let nx_apx (sp sr : chest1 bf16 8192) (xr : S.seq real) (b : nat) : prop =
  forall (c:nat). c < 1024 ==> nres_f sp sr (b + c) %~ at xr (b + c)

noextract
let row_ex (sp sr sn : chest1 bf16 8192) (b : nat{b + 1024 <= 8192}) : prop =
  forall (c:nat). c < 1024 ==> acc1 sn (b + c) == nres_f sp sr (b + c)

(* ---- partial sums ---- *)
noextract
let pa_b (sp sr : chest1 bf16 8192) (b t : nat) (v : f32) : prop =
  forall (xr : S.seq real). nx_apx sp sr xr b ==> v %~ pq4 xr b 256 t

[@@"opaque_to_smt"] noextract
let pa_ok (sp sr : chest1 bf16 8192) (b t : nat) (v : f32) : prop = pa_b sp sr b t v

noextract
let pa_i (sp sr : chest1 bf16 8192) (b t : nat) (v : f32)
  : Lemma (requires pa_b sp sr b t v) (ensures pa_ok sp sr b t v)
  = reveal_opaque (`%pa_ok) (pa_ok sp sr b t v)

noextract
let pa_e (sp sr : chest1 bf16 8192) (b t : nat) (v : f32) (xr : S.seq real)
  : Lemma (requires pa_ok sp sr b t v /\ nx_apx sp sr xr b) (ensures v %~ pq4 xr b 256 t)
  = reveal_opaque (`%pa_ok) (pa_ok sp sr b t v)

noextract
let fma_i (v z a : f32) (xr zr : real)
  : Lemma (requires (forall (xr yr zr : real). v_approximates v xr /\ v_approximates v yr /\ v_approximates z zr ==>
                       v_approximates a (r_fma xr yr zr)) /\ v %~ xr /\ z %~ zr)
          (ensures a %~ (xr *. xr +. zr))
  = ()

noextract
let pa_x (sp sr : chest1 bf16 8192) (b t : nat) (v0 v1 v2 v3 a0 a1 a2 a3 : f32) (xr : S.seq real)
  : Lemma (requires b + 1024 <= 8192 /\ t < 256 /\
             v0 == tof (nres_f sp sr (b + t)) /\ v1 == tof (nres_f sp sr (b + (t + 256))) /\
             v2 == tof (nres_f sp sr (b + (t + 512))) /\ v3 == tof (nres_f sp sr (b + (t + 768))) /\
             (forall (xr yr zr : real). v_approximates v0 xr /\ v_approximates v0 yr /\ v_approximates (zero #f32) zr ==>
                v_approximates a0 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v1 xr /\ v_approximates v1 yr /\ v_approximates a0 zr ==>
                v_approximates a1 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v2 xr /\ v_approximates v2 yr /\ v_approximates a1 zr ==>
                v_approximates a2 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v3 xr /\ v_approximates v3 yr /\ v_approximates a2 zr ==>
                v_approximates a3 (r_fma xr yr zr)) /\
             nx_apx sp sr xr b)
          (ensures a3 %~ pq4 xr b 256 t)
  = zo_apx ();
    assert (nres_f sp sr (b + t) %~ at xr (b + t));
    assert (nres_f sp sr (b + (t + 256)) %~ at xr (b + (t + 256)));
    assert (nres_f sp sr (b + (t + 512)) %~ at xr (b + (t + 512)));
    assert (nres_f sp sr (b + (t + 768)) %~ at xr (b + (t + 768)));
    let x0 = at xr (b + t) in
    let x1 = at xr (b + t + 256) in
    let x2 = at xr (b + t + 2 * 256) in
    let x3 = at xr (b + t + 3 * 256) in
    assert (v0 %~ x0); assert (v1 %~ x1); assert (v2 %~ x2); assert (v3 %~ x3);
    fma_i v0 zero a0 x0 0.0R;
    fma_i v1 a0 a1 x1 (x0 *. x0 +. 0.0R);
    fma_i v2 a1 a2 x2 (x1 *. x1 +. (x0 *. x0 +. 0.0R));
    fma_i v3 a2 a3 x3 (x2 *. x2 +. (x1 *. x1 +. (x0 *. x0 +. 0.0R)));
    assert (a3 %~ (x3 *. x3 +. (x2 *. x2 +. (x1 *. x1 +. (x0 *. x0 +. 0.0R)))))

noextract
let pa_lem (sp sr : chest1 bf16 8192) (b t : nat) (v0 v1 v2 v3 a0 a1 a2 a3 : f32)
  : Lemma (requires b + 1024 <= 8192 /\ t < 256 /\
             v0 == tof (nres_f sp sr (b + t)) /\ v1 == tof (nres_f sp sr (b + (t + 256))) /\
             v2 == tof (nres_f sp sr (b + (t + 512))) /\ v3 == tof (nres_f sp sr (b + (t + 768))) /\
             (forall (xr yr zr : real). v_approximates v0 xr /\ v_approximates v0 yr /\ v_approximates (zero #f32) zr ==>
                v_approximates a0 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v1 xr /\ v_approximates v1 yr /\ v_approximates a0 zr ==>
                v_approximates a1 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v2 xr /\ v_approximates v2 yr /\ v_approximates a1 zr ==>
                v_approximates a2 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v3 xr /\ v_approximates v3 yr /\ v_approximates a2 zr ==>
                v_approximates a3 (r_fma xr yr zr)))
          (ensures pa_ok sp sr b t a3)
  = Classical.forall_intro (Classical.move_requires (pa_x sp sr b t v0 v1 v2 v3 a0 a1 a2 a3));
    pa_i sp sr b t a3

(* ---- group sums ---- *)
noextract
let gs_b (sp sr : chest1 bf16 8192) (b w : nat) (v : f32) : prop =
  forall (xr : S.seq real). nx_apx sp sr xr b ==> v %~ blk (pq4 xr b 256) 32 0 w

[@@"opaque_to_smt"] noextract
let gs_ok (sp sr : chest1 bf16 8192) (b w : nat) (v : f32) : prop = gs_b sp sr b w v

noextract
let gs_i (sp sr : chest1 bf16 8192) (b w : nat) (v : f32)
  : Lemma (requires gs_b sp sr b w v) (ensures gs_ok sp sr b w v)
  = reveal_opaque (`%gs_ok) (gs_ok sp sr b w v)

noextract
let gs_e (sp sr : chest1 bf16 8192) (b w : nat) (v : f32) (xr : S.seq real)
  : Lemma (requires gs_ok sp sr b w v /\ nx_apx sp sr xr b) (ensures v %~ blk (pq4 xr b 256) 32 0 w)
  = reveal_opaque (`%gs_ok) (gs_ok sp sr b w v)

noextract
let grp_x (sp sr : chest1 bf16 8192) (b : nat) (s : chest1 f32 256) (wv : nat{wv < 8}) (xr : S.seq real)
  : Lemma (requires (forall (j:natlt 256). wv * 32 <= j ==> pa_ok sp sr b j (acc1 s j)) /\ nx_apx sp sr xr b)
          (ensures acc1 (tree_spec s (wv * 32) 16) (wv * 32) %~ blk (pq4 xr b 256) 32 0 wv)
  = p16 ();
    let g = pq4 xr b 256 in
    let aux (t:nat{t < 32}) : Lemma (acc1 s (wv * 32 + t) %~ g (wv * 32 + t)) =
      pa_e sp sr b (wv * 32 + t) (acc1 s (wv * 32 + t)) xr
    in
    Classical.forall_intro aux;
    tree_apx 4 s (wv * 32) g (wv * 32);
    assert (blk g 32 0 wv == rs g 32 (0 + 32 * wv) 1)

noextract
let grp_inv (sp sr : chest1 bf16 8192) (b : nat) (s : chest1 f32 256) (wv : nat) : prop =
  (forall (w:natlt 8). w < wv ==> gs_ok sp sr b w (acc1 s w)) /\
  (forall (j:natlt 256). wv * 32 <= j ==> pa_ok sp sr b j (acc1 s j))

noextract
let grp_step (sp sr : chest1 bf16 8192) (b : nat) (s : chest1 f32 256) (wv : nat{wv < 8})
  : Lemma (requires grp_inv sp sr b s wv)
          (ensures (let s1 = tree_spec s (wv * 32) 16 in
                    grp_inv sp sr b (upd s1 (wv, ()) (acc1 s1 (wv * 32))) (wv + 1)))
  = let s1 = tree_spec s (wv * 32) 16 in
    Classical.forall_intro (Classical.move_requires (grp_x sp sr b s wv));
    gs_i sp sr b wv (acc1 s1 (wv * 32));
    let fr (j:natlt 256) : Lemma (requires j < wv \/ j >= wv * 32 + 32) (ensures acc1 s1 j == acc1 s j) =
      tree_frame s (wv * 32) 16 j
    in
    Classical.forall_intro (Classical.move_requires fr)

noextract
let fin_x (sp sr : chest1 bf16 8192) (b : nat) (s : chest1 f32 256) (xr : S.seq real)
  : Lemma (requires (forall (w:natlt 8). gs_ok sp sr b w (acc1 s w)) /\ nx_apx sp sr xr b)
          (ensures acc1 (tree_spec s 0 4) 0 %~ sumsq 1024 xr b)
  = p16 ();
    let g = pq4 xr b 256 in
    let aux (t:nat{t < 8}) : Lemma (acc1 s (0 + t) %~ blk g 32 0 (0 + t)) =
      gs_e sp sr b t (acc1 s t) xr
    in
    Classical.forall_intro aux;
    tree_apx 2 s 0 (blk g 32 0) 0;
    rs_blk g 32 8 0;
    sumsq_pq4 xr b 256

(* ---- row inverse ---- *)
noextract
let inv_b (sp sr : chest1 bf16 8192) (b : nat) (v : f32) : prop =
  forall (xr : S.seq real). nx_apx sp sr xr b ==> v %~ rms_inv 1024 xr b

[@@"opaque_to_smt"] noextract
let inv_ok (sp sr : chest1 bf16 8192) (b : nat) (v : f32) : prop = inv_b sp sr b v

noextract
let inv_e (sp sr : chest1 bf16 8192) (b : nat) (v : f32) (xr : S.seq real)
  : Lemma (requires inv_ok sp sr b v /\ nx_apx sp sr xr b) (ensures v %~ rms_inv 1024 xr b)
  = reveal_opaque (`%inv_ok) (inv_ok sp sr b v)

noextract
let eps_apx () : Lemma (eps %~ rms_eps) =
  zo_apx ();
  oi 1000000L;
  let _ = approx_div #f32 in
  let c : f32 = of_int 1000000L in
  assert (c %~ 1000000.0R);
  assert (div one c %~ (1.0R /. 1000000.0R))

noextract
let rinv_apx (p0 ep : f32) (n : FStar.Int64.t) (sr : real)
  : Lemma (requires p0 %~ sr /\ sr >=. 0.0R /\ ep == eps /\ FStar.Int64.v n > 0)
          (ensures rsqrt (add (div p0 (of_int n)) ep) %~ Sqrt.rsqrt (sr /. of_int (FStar.Int64.v n) +. rms_eps))
  = eps_apx ();
    oi n;
    let _ = approx_div #f32 in
    let _ = approx_rsqrt #f32 in
    let d = FStar.Real.of_int (FStar.Int64.v n) in
    assert (d >. 0.0R);
    let q = div p0 (of_int n) in
    assert (q %~ (sr /. d));
    let a = add q ep in
    assert (a %~ (sr /. d +. rms_eps));
    assert (sr /. d +. rms_eps >. 0.0R)

noextract
let inv_x (sp sr : chest1 bf16 8192) (b : nat) (s : chest1 f32 256) (ep : f32) (xr : S.seq real)
  : Lemma (requires (forall (w:natlt 8). gs_ok sp sr b w (acc1 s w)) /\ ep == eps /\ nx_apx sp sr xr b)
          (ensures rsqrt (add (div (acc1 (tree_spec s 0 4) 0) (of_int 1024L)) ep) %~ rms_inv 1024 xr b)
  = fin_x sp sr b s xr;
    rinv_apx (acc1 (tree_spec s 0 4) 0) ep 1024L (sumsq 1024 xr b)

noextract
let inv_lem (sp sr : chest1 bf16 8192) (b : nat) (s : chest1 f32 256) (ep : f32)
  : Lemma (requires (forall (w:natlt 8). gs_ok sp sr b w (acc1 s w)) /\ ep == eps)
          (ensures inv_ok sp sr b (rsqrt (add (div (acc1 (tree_spec s 0 4) 0) (of_int 1024L)) ep)))
  = Classical.forall_intro (Classical.move_requires (inv_x sp sr b s ep));
    reveal_opaque (`%inv_ok) (inv_ok sp sr b (rsqrt (add (div (acc1 (tree_spec s 0 4) 0) (of_int 1024L)) ep)))

(* ---- normalized output ---- *)
noextract
let el_b (sp sr : chest1 bf16 8192) (sw : chest1 bf16 1024) (i : nat) (v : bf16) : prop =
  forall (xr wr : S.seq real). nx_apx sp sr xr ((i / 1024) * 1024) /\ capx sw wr ==>
    v %~ (at wr (i % 1024) *. (at xr i *. rms_inv 1024 xr ((i / 1024) * 1024)))

[@@"opaque_to_smt"] noextract
let el_ok (sp sr : chest1 bf16 8192) (sw : chest1 bf16 1024) (i : nat) (v : bf16) : prop = el_b sp sr sw i v

noextract
let el_e (sp sr : chest1 bf16 8192) (sw : chest1 bf16 1024) (i : nat) (v : bf16) (xr wr : S.seq real)
  : Lemma (requires el_ok sp sr sw i v /\ nx_apx sp sr xr ((i / 1024) * 1024) /\ capx sw wr)
          (ensures v %~ (at wr (i % 1024) *. (at xr i *. rms_inv 1024 xr ((i / 1024) * 1024))))
  = reveal_opaque (`%el_ok) (el_ok sp sr sw i v)

noextract
let el_x (sp sr : chest1 bf16 8192) (sw : chest1 bf16 1024) (r c : nat) (inv v wc : f32) (xr wr : S.seq real)
  : Lemma (requires r < 8 /\ c < 1024 /\ inv_ok sp sr (r * 1024) inv /\
                    v == tof (nres_f sp sr (r * 1024 + c)) /\ wc == tof (acc1 sw c) /\
                    nx_apx sp sr xr (((r * 1024 + c) / 1024) * 1024) /\ capx sw wr)
          (ensures tob (mul v (mul inv wc)) %~
                   (at wr ((r * 1024 + c) % 1024) *. (at xr (r * 1024 + c) *. rms_inv 1024 xr (((r * 1024 + c) / 1024) * 1024))))
  = ML.lemma_div_plus c r 1024;
    ML.small_division_lemma_1 c 1024;
    ML.lemma_mod_plus c r 1024;
    ML.small_mod c 1024;
    let b = r * 1024 in
    assert ((b + c) / 1024 == r /\ (b + c) % 1024 == c);
    inv_e sp sr b inv xr;
    assert (nres_f sp sr (b + c) %~ at xr (b + c));
    let x = at xr (b + c) in
    let wv = at wr c in
    let iv = rms_inv 1024 xr b in
    assert (v %~ x);
    assert (wc %~ wv);
    assert (mul inv wc %~ (iv *. wv));
    assert (mul v (mul inv wc) %~ (x *. (iv *. wv)));
    assert (x *. (iv *. wv) == wv *. (x *. iv))

noextract
let an_el_lem (sp sr : chest1 bf16 8192) (sw : chest1 bf16 1024) (r c : nat) (inv v wc : f32)
  : Lemma (requires r < 8 /\ c < 1024 /\ inv_ok sp sr (r * 1024) inv /\
                    v == tof (nres_f sp sr (r * 1024 + c)) /\ wc == tof (acc1 sw c))
          (ensures el_ok sp sr sw (r * 1024 + c) (tob (mul v (mul inv wc))))
  = Classical.forall_intro_2 (Classical.move_requires_2 (el_x sp sr sw r c inv v wc));
    reveal_opaque (`%el_ok) (el_ok sp sr sw (r * 1024 + c) (tob (mul v (mul inv wc))))

(* ---- Pulse ---- *)
inline_for_extraction noextract
fn an_res (pj res nres : array1 bf16 (l1_forward 8192)) (b : sz{SZ.v b + 1024 <= 8192})
  (#fp #fr : perm) (#sp #sr : erased (chest1 bf16 8192)) (#sn0 : erased (chest1 bf16 8192))
  preserves pj |-> Frac fp sp ** res |-> Frac fr sr
  requires nres |-> sn0 ** pure (forall (i:natlt 8192). i < SZ.v b ==> acc1 sn0 i == nres_f sp sr i)
  ensures exists* sn. nres |-> sn ** pure (forall (i:natlt 8192). i < SZ.v b + 1024 ==> acc1 sn i == nres_f sp sr i)
{
  let mut c : sz = 0sz;
  while (!c <^ 1024sz)
    invariant live c ** pure (SZ.v !c <= 1024)
    invariant exists* sn. nres |-> sn **
      pure (forall (i:natlt 8192). i < SZ.v b + SZ.v !c ==> acc1 sn i == nres_f sp sr i)
    decreases (1024 - SZ.v !c)
  {
    let cv = !c;
    let x = tof (rd pj (b +^ cv));
    let y = tof (rd res (b +^ cv));
    wr nres (b +^ cv) (tob (add x y));
    c := cv +^ 1sz;
  };
}

inline_for_extraction noextract
fn rdn (nres : array1 bf16 (l1_forward 8192)) (b : sz{SZ.v b + 1024 <= 8192}) (c : sz{SZ.v c < 1024})
  (sp sr : erased (chest1 bf16 8192)) (#sn : erased (chest1 bf16 8192))
  preserves nres |-> sn ** pure (row_ex sp sr sn (SZ.v b))
  returns x : f32
  ensures pure (x == tof (nres_f sp sr (SZ.v b + SZ.v c)))
{
  tof (rd nres (b +^ c))
}

inline_for_extraction noextract
fn pa_one (nres : array1 bf16 (l1_forward 8192)) (b : sz{SZ.v b + 1024 <= 8192}) (tv : sz{SZ.v tv < 256})
  (sp sr : erased (chest1 bf16 8192)) (#sn : erased (chest1 bf16 8192))
  preserves gpu ** nres |-> sn ** pure (row_ex sp sr sn (SZ.v b))
  returns a3 : f32
  ensures pure (pa_ok sp sr (SZ.v b) (SZ.v tv) a3)
{
  let v0 = rdn nres b tv sp sr;
  let a0 = Fast.fma_rn v0 v0 zero;
  let v1 = rdn nres b (tv +^ 256sz) sp sr;
  let a1 = Fast.fma_rn v1 v1 a0;
  let v2 = rdn nres b (tv +^ 512sz) sp sr;
  let a2 = Fast.fma_rn v2 v2 a1;
  let v3 = rdn nres b (tv +^ 768sz) sp sr;
  let a3 = Fast.fma_rn v3 v3 a2;
  pa_lem sp sr (SZ.v b) (SZ.v tv) v0 v1 v2 v3 a0 a1 a2 a3;
  a3
}

inline_for_extraction noextract
fn an_part (nres : array1 bf16 (l1_forward 8192)) (part : array1 f32 (l1_forward 256)) (b : sz{SZ.v b + 1024 <= 8192})
  (sp sr : erased (chest1 bf16 8192)) (#sn : erased (chest1 bf16 8192)) (#sq0 : erased (chest1 f32 256))
  preserves gpu ** nres |-> sn ** pure (row_ex sp sr sn (SZ.v b))
  requires part |-> sq0
  ensures exists* sq. part |-> sq ** pure (grp_inv sp sr (SZ.v b) sq 0)
{
  let mut t : sz = 0sz;
  while (!t <^ 256sz)
    invariant live t ** pure (SZ.v !t <= 256)
    invariant exists* sq. part |-> sq **
      pure (forall (j:natlt 256). j < SZ.v !t ==> pa_ok sp sr (SZ.v b) j (acc1 sq j))
    decreases (256 - SZ.v !t)
  {
    let tv = !t;
    let a3 = pa_one nres b tv sp sr;
    wr part tv a3;
    t := tv +^ 1sz;
  };
}

inline_for_extraction noextract
fn an_grp (part : array1 f32 (l1_forward 256)) (b : erased nat)
  (sp sr : erased (chest1 bf16 8192)) (#sq0 : erased (chest1 f32 256))
  requires part |-> sq0 ** pure (grp_inv sp sr b sq0 0)
  ensures exists* sq. part |-> sq ** pure (grp_inv sp sr b sq 8)
{
  let mut wi : sz = 0sz;
  while (!wi <^ 8sz)
    invariant live wi ** pure (SZ.v !wi <= 8)
    invariant exists* sq. part |-> sq ** pure (grp_inv sp sr b sq (SZ.v !wi))
    decreases (8 - SZ.v !wi)
  {
    let wv = !wi;
    with s1. assert (part |-> s1);
    grp_step sp sr b s1 (SZ.v wv);
    tree_red part (wv *^ 32sz) 32sz;
    let x = rd part (wv *^ 32sz);
    wr part wv x;
    wi := wv +^ 1sz;
  };
}

inline_for_extraction noextract
fn an_inv (part : array1 f32 (l1_forward 256)) (b : erased nat)
  (sp sr : erased (chest1 bf16 8192)) (#sq0 : erased (chest1 f32 256))
  preserves gpu
  requires part |-> sq0 ** pure (grp_inv sp sr b sq0 8)
  returns inv : f32
  ensures (exists* sq. part |-> sq) ** pure (inv_ok sp sr b inv)
{
  tree_red part 0sz 8sz;
  let p0 = rd part 0sz;
  let ep = mk_eps ();
  inv_lem sp sr b sq0 ep;
  rsqrt (add (div p0 (of_int 1024L)) ep)
}

inline_for_extraction noextract
fn an_nrm (w : array1 bf16 (l1_forward 1024)) (nres nrm : array1 bf16 (l1_forward 8192))
  (rv : sz{SZ.v rv < 8}) (inv : f32)
  (sp sr : erased (chest1 bf16 8192)) (#fw : perm) (#sw : erased (chest1 bf16 1024))
  (#sn #sm0 : erased (chest1 bf16 8192))
  preserves w |-> Frac fw sw ** nres |-> sn ** pure (row_ex sp sr sn (SZ.v rv * 1024))
  requires nrm |-> sm0 ** pure (inv_ok sp sr (SZ.v rv * 1024) inv) **
    pure (forall (i:natlt 8192). i < SZ.v rv * 1024 ==> el_ok sp sr sw i (acc1 sm0 i))
  ensures exists* sm. nrm |-> sm **
    pure (forall (i:natlt 8192). i < SZ.v rv * 1024 + 1024 ==> el_ok sp sr sw i (acc1 sm i))
{
  let b = rv *^ 1024sz;
  let mut c2 : sz = 0sz;
  while (!c2 <^ 1024sz)
    invariant live c2 ** pure (SZ.v !c2 <= 1024)
    invariant exists* sm. nrm |-> sm **
      pure (forall (i:natlt 8192). i < SZ.v b + SZ.v !c2 ==> el_ok sp sr sw i (acc1 sm i))
    decreases (1024 - SZ.v !c2)
  {
    let cv = !c2;
    let v = rdn nres b cv sp sr;
    let wc = tof (rd w cv);
    an_el_lem sp sr sw (SZ.v rv) (SZ.v cv) inv v wc;
    wr nrm (b +^ cv) (tob (mul v (mul inv wc)));
    c2 := cv +^ 1sz;
  };
}

inline_for_extraction noextract
fn k_addnorm (w : array1 bf16 (l1_forward 1024)) (pj : array1 bf16 (l1_forward 8192))
  (res : array1 bf16 (l1_forward 8192)) (nres : array1 bf16 (l1_forward 8192))
  (nrm : array1 bf16 (l1_forward 8192)) (part : array1 f32 (l1_forward 256))
  (#fw #fp #fr : perm) (#sw : erased (chest1 bf16 1024)) (#sp #sr : erased (chest1 bf16 8192))
  (#sn0 #sm0 : erased (chest1 bf16 8192)) (#sq0 : erased (chest1 f32 256))
  preserves gpu ** w |-> Frac fw sw ** pj |-> Frac fp sp ** res |-> Frac fr sr
  requires nres |-> sn0 ** nrm |-> sm0 ** part |-> sq0
  ensures exists* sn sm sq. nres |-> sn ** nrm |-> sm ** part |-> sq **
    pure ((forall (i:natlt 8192). acc1 sn i == nres_f sp sr i) /\
          (forall (i:natlt 8192). el_ok sp sr sw i (acc1 sm i)))
{
  let mut r : sz = 0sz;
  while (!r <^ 8sz)
    invariant live r ** pure (SZ.v !r <= 8)
    invariant exists* sn sm sq. nres |-> sn ** nrm |-> sm ** part |-> sq **
      pure ((forall (i:natlt 8192). i < SZ.v !r * 1024 ==> acc1 sn i == nres_f sp sr i) /\
            (forall (i:natlt 8192). i < SZ.v !r * 1024 ==> el_ok sp sr sw i (acc1 sm i)))
    decreases (8 - SZ.v !r)
  {
    let rv = !r;
    let b = rv *^ 1024sz;
    an_res pj res nres b;
    with sn. assert (nres |-> sn);
    assert pure (row_ex sp sr sn (SZ.v b));
    an_part nres part b sp sr;
    an_grp part (SZ.v b) sp sr;
    let inv = an_inv part (SZ.v b) sp sr;
    an_nrm w nres nrm rv inv sp sr;
    r := rv +^ 1sz;
  };
}

(* ===== K5 ===== *)

(* masked ideal of the qkv buffer: token row 7 is zero *)
noextract
let mk_m (qkr : S.seq real) : S.seq real =
  S.init 32768 (fun (j:nat{j < 32768}) -> if j / 4096 < 7 then at qkr j else 0.0R)

noextract
let mk_m_at (qkr : S.seq real) (j : nat{j < 32768})
  : Lemma (at (mk_m qkr) j == (if j / 4096 < 7 then at qkr j else 0.0R)) = ()

noextract
let mval (sq : chest1 bf16 32768) (tok : nat) (j : natlt 32768) : GTot f32 =
  if tok < 7 then tof (acc1 sq j) else fz

noextract
let vb_ok (vb tok h pt : nat) : prop = vb == tok * 4096 + h * 512 + pt * 128 /\ tok < 8 /\ h < 8 /\ pt < 4

noextract
let vb_div (vb tok h pt c : nat)
  : Lemma (requires vb_ok vb tok h pt /\ c < 128) (ensures vb + c < 32768 /\ (vb + c) / 4096 == tok)
  = ML.lemma_div_plus (h * 512 + pt * 128 + c) tok 4096;
    ML.small_div (h * 512 + pt * 128 + c) 4096

noextract
let mval_apx (sq : chest1 bf16 32768) (vb tok h pt c : nat) (qkr : S.seq real)
  : Lemma (requires vb_ok vb tok h pt /\ c < 128 /\ capx sq qkr)
          (ensures vb + c < 32768 /\ mval sq tok (vb + c) %~ at (mk_m qkr) (vb + c))
  = vb_div vb tok h pt c;
    mk_m_at qkr (vb + c);
    zo_apx ()

(* ---- partial sums ---- *)
noextract
let pq_b (sq : chest1 bf16 32768) (vb t : nat) (v : f32) : prop =
  forall (qkr : S.seq real). capx sq qkr ==> v %~ pq4 (mk_m qkr) vb 32 t

[@@"opaque_to_smt"] noextract
let pq_ok (sq : chest1 bf16 32768) (vb t : nat) (v : f32) : prop = pq_b sq vb t v

noextract
let pq_e (sq : chest1 bf16 32768) (vb t : nat) (v : f32) (qkr : S.seq real)
  : Lemma (requires pq_ok sq vb t v /\ capx sq qkr) (ensures v %~ pq4 (mk_m qkr) vb 32 t)
  = reveal_opaque (`%pq_ok) (pq_ok sq vb t v)

noextract
let pq_x (sq : chest1 bf16 32768) (vb tok h pt t : nat) (v0 v1 v2 v3 a0 a1 a2 a3 : f32) (qkr : S.seq real)
  : Lemma (requires vb_ok vb tok h pt /\ t < 32 /\
             vb + t + 96 < 32768 /\
             v0 == mval sq tok (vb + t) /\ v1 == mval sq tok (vb + t + 32) /\
             v2 == mval sq tok (vb + t + 64) /\ v3 == mval sq tok (vb + t + 96) /\
             (forall (xr yr zr : real). v_approximates v0 xr /\ v_approximates v0 yr /\ v_approximates (zero #f32) zr ==>
                v_approximates a0 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v1 xr /\ v_approximates v1 yr /\ v_approximates a0 zr ==>
                v_approximates a1 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v2 xr /\ v_approximates v2 yr /\ v_approximates a1 zr ==>
                v_approximates a2 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v3 xr /\ v_approximates v3 yr /\ v_approximates a2 zr ==>
                v_approximates a3 (r_fma xr yr zr)) /\
             capx sq qkr)
          (ensures a3 %~ pq4 (mk_m qkr) vb 32 t)
  = zo_apx ();
    let xm = mk_m qkr in
    mval_apx sq vb tok h pt t qkr;
    mval_apx sq vb tok h pt (t + 32) qkr;
    mval_apx sq vb tok h pt (t + 64) qkr;
    mval_apx sq vb tok h pt (t + 96) qkr;
    let x0 = at xm (vb + t) in
    let x1 = at xm (vb + t + 32) in
    let x2 = at xm (vb + t + 2 * 32) in
    let x3 = at xm (vb + t + 3 * 32) in
    assert (v0 %~ x0); assert (v1 %~ x1); assert (v2 %~ x2); assert (v3 %~ x3);
    fma_i v0 zero a0 x0 0.0R;
    fma_i v1 a0 a1 x1 (x0 *. x0 +. 0.0R);
    fma_i v2 a1 a2 x2 (x1 *. x1 +. (x0 *. x0 +. 0.0R));
    fma_i v3 a2 a3 x3 (x2 *. x2 +. (x1 *. x1 +. (x0 *. x0 +. 0.0R)))

noextract
let pq_lem (sq : chest1 bf16 32768) (vb tok h pt t : nat) (v0 v1 v2 v3 a0 a1 a2 a3 : f32)
  : Lemma (requires vb_ok vb tok h pt /\ t < 32 /\
             vb + t + 96 < 32768 /\
             v0 == mval sq tok (vb + t) /\ v1 == mval sq tok (vb + t + 32) /\
             v2 == mval sq tok (vb + t + 64) /\ v3 == mval sq tok (vb + t + 96) /\
             (forall (xr yr zr : real). v_approximates v0 xr /\ v_approximates v0 yr /\ v_approximates (zero #f32) zr ==>
                v_approximates a0 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v1 xr /\ v_approximates v1 yr /\ v_approximates a0 zr ==>
                v_approximates a1 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v2 xr /\ v_approximates v2 yr /\ v_approximates a1 zr ==>
                v_approximates a2 (r_fma xr yr zr)) /\
             (forall (xr yr zr : real). v_approximates v3 xr /\ v_approximates v3 yr /\ v_approximates a2 zr ==>
                v_approximates a3 (r_fma xr yr zr)))
          (ensures pq_ok sq vb t a3)
  = Classical.forall_intro (Classical.move_requires (pq_x sq vb tok h pt t v0 v1 v2 v3 a0 a1 a2 a3));
    reveal_opaque (`%pq_ok) (pq_ok sq vb t a3)

(* ---- row inverse ---- *)
noextract
let qi_b (sq : chest1 bf16 32768) (vb : nat) (v : f32) : prop =
  forall (qkr : S.seq real). capx sq qkr ==> v %~ rms_inv 128 (mk_m qkr) vb

[@@"opaque_to_smt"] noextract
let qi_ok (sq : chest1 bf16 32768) (vb : nat) (v : f32) : prop = qi_b sq vb v

noextract
let qi_e (sq : chest1 bf16 32768) (vb : nat) (v : f32) (qkr : S.seq real)
  : Lemma (requires qi_ok sq vb v /\ capx sq qkr) (ensures v %~ rms_inv 128 (mk_m qkr) vb)
  = reveal_opaque (`%qi_ok) (qi_ok sq vb v)

noextract
let qi_x (sq : chest1 bf16 32768) (vb : nat) (s : chest1 f32 32) (ep : f32) (qkr : S.seq real)
  : Lemma (requires (forall (t:natlt 32). pq_ok sq vb t (acc1 s t)) /\ ep == eps /\ capx sq qkr)
          (ensures rsqrt (add (div (acc1 (tree_spec s 0 16) 0) (of_int 128L)) ep) %~ rms_inv 128 (mk_m qkr) vb)
  = p16 ();
    let g = pq4 (mk_m qkr) vb 32 in
    let aux (t:nat{t < 32}) : Lemma (acc1 s (0 + t) %~ g (0 + t)) = pq_e sq vb t (acc1 s t) qkr in
    Classical.forall_intro aux;
    tree_apx 4 s 0 g 0;
    sumsq_pq4 (mk_m qkr) vb 32;
    rinv_apx (acc1 (tree_spec s 0 16) 0) ep 128L (sumsq 128 (mk_m qkr) vb)

noextract
let qi_lem (sq : chest1 bf16 32768) (vb : nat) (s : chest1 f32 32) (ep : f32)
  : Lemma (requires (forall (t:natlt 32). pq_ok sq vb t (acc1 s t)) /\ ep == eps)
          (ensures qi_ok sq vb (rsqrt (add (div (acc1 (tree_spec s 0 16) 0) (of_int 128L)) ep)))
  = Classical.forall_intro (Classical.move_requires (qi_x sq vb s ep));
    reveal_opaque (`%qi_ok) (qi_ok sq vb (rsqrt (add (div (acc1 (tree_spec s 0 16) 0) (of_int 128L)) ep)))

(* ---- normalized, weighted head row ---- *)
noextract
let unr (qkr wr : S.seq real) (vb wb c : nat) : real =
  (at (mk_m qkr) (vb + c) *. rms_inv 128 (mk_m qkr) vb) *. at wr (wb + c)

noextract
let un_b (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (vb wb c : nat) (v : bf16) : prop =
  forall (qkr wr : S.seq real). capx sq qkr /\ capx sw wr ==> v %~ unr qkr wr vb wb c

[@@"opaque_to_smt"] noextract
let un_ok (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (vb wb c : nat) (v : bf16) : prop = un_b sq sw vb wb c v

noextract
let un_e (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (vb wb c : nat) (v : bf16) (qkr wr : S.seq real)
  : Lemma (requires un_ok sq sw vb wb c v /\ capx sq qkr /\ capx sw wr) (ensures v %~ unr qkr wr vb wb c)
  = reveal_opaque (`%un_ok) (un_ok sq sw vb wb c v)

noextract
let un_x (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (vb tok h pt wb c : nat) (inv v wc : f32) (qkr wr : S.seq real)
  : Lemma (requires vb_ok vb tok h pt /\ c < 128 /\ wb + 128 <= 256 /\ qi_ok sq vb inv /\
                    vb + c < 32768 /\ v == mval sq tok (vb + c) /\ wc == tof (acc1 sw (wb + c)) /\
                    capx sq qkr /\ capx sw wr)
          (ensures tob (mul (tof (tob (mul v inv))) wc) %~ unr qkr wr vb wb c)
  = mval_apx sq vb tok h pt c qkr;
    qi_e sq vb inv qkr;
    let x = at (mk_m qkr) (vb + c) in
    let iv = rms_inv 128 (mk_m qkr) vb in
    let w = at wr (wb + c) in
    assert (wc %~ w);
    assert (mul v inv %~ (x *. iv));
    assert (tof (tob (mul v inv)) %~ (x *. iv));
    assert (mul (tof (tob (mul v inv))) wc %~ ((x *. iv) *. w))

noextract
let un_lem (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (vb tok h pt wb c : nat) (inv v wc : f32)
  : Lemma (requires vb_ok vb tok h pt /\ c < 128 /\ wb + 128 <= 256 /\ qi_ok sq vb inv /\
                    vb + c < 32768 /\ v == mval sq tok (vb + c) /\ wc == tof (acc1 sw (wb + c)))
          (ensures un_ok sq sw vb wb c (tob (mul (tof (tob (mul v inv))) wc)))
  = Classical.forall_intro_2 (Classical.move_requires_2 (un_x sq sw vb tok h pt wb c inv v wc));
    reveal_opaque (`%un_ok) (un_ok sq sw vb wb c (tob (mul (tof (tob (mul v inv))) wc)))

(* ---- rotary ---- *)
noextract
let rotr (qkr wr cr sr : S.seq real) (vb wb tb c : nat) : real =
  let oth = if c < 64 then 0.0R -. unr qkr wr vb wb (c + 64) else unr qkr wr vb wb (c - 64) in
  unr qkr wr vb wb c *. at cr (tb + c) +. oth *. at sr (tb + c)

noextract
let rot_b (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (vb wb tb c : nat) (v : bf16) : prop =
  forall (qkr wr cr sr : S.seq real). capx sq qkr /\ capx sw wr /\ capx scs cr /\ capx ssn sr ==>
    v %~ rotr qkr wr cr sr vb wb tb c

[@@"opaque_to_smt"] noextract
let rot_ok (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (vb wb tb c : nat) (v : bf16) : prop =
  rot_b sq sw scs ssn vb wb tb c v

noextract
let rot_e (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (vb wb tb c : nat) (v : bf16)
  (qkr wr cr sr : S.seq real)
  : Lemma (requires rot_ok sq sw scs ssn vb wb tb c v /\ capx sq qkr /\ capx sw wr /\ capx scs cr /\ capx ssn sr)
          (ensures v %~ rotr qkr wr cr sr vb wb tb c)
  = reveal_opaque (`%rot_ok) (rot_ok sq sw scs ssn vb wb tb c v)

noextract
let un_all (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (vb wb : nat) (su : chest1 bf16 128) : prop =
  forall (c:natlt 128). un_ok sq sw vb wb c (acc1 su c)

noextract
let rot_x (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (vb wb tb c : nat)
  (su : chest1 bf16 128) (u ot other co si : f32) (qkr wr cr sr : S.seq real)
  : Lemma (requires un_all sq sw vb wb su /\ c < 128 /\ tb + 128 <= 524288 /\
                    u == tof (acc1 su c) /\ ot == tof (acc1 su ((c + 64) % 128)) /\
                    other == (if c < 64 then neg ot else ot) /\
                    co == tof (acc1 scs (tb + c)) /\ si == tof (acc1 ssn (tb + c)) /\
                    capx sq qkr /\ capx sw wr /\ capx scs cr /\ capx ssn sr)
          (ensures tob (add (tof (tob (mul u co))) (tof (tob (mul other si)))) %~ rotr qkr wr cr sr vb wb tb c)
  = let _ = approx_neg #f32 in
    let c2 = (c + 64) % 128 in
    assert (c2 == (if c < 64 then c + 64 else c - 64));
    un_e sq sw vb wb c (acc1 su c) qkr wr;
    un_e sq sw vb wb c2 (acc1 su c2) qkr wr;
    let uc = unr qkr wr vb wb c in
    let uo = unr qkr wr vb wb c2 in
    let oth = if c < 64 then 0.0R -. uo else uo in
    assert (u %~ uc);
    assert (ot %~ uo);
    assert (other %~ oth);
    assert (co %~ at cr (tb + c));
    assert (si %~ at sr (tb + c));
    assert (tof (tob (mul u co)) %~ (uc *. at cr (tb + c)));
    assert (tof (tob (mul other si)) %~ (oth *. at sr (tb + c)))

noextract
let rot_lem (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (vb wb tb c : nat)
  (su : chest1 bf16 128) (u ot other co si : f32)
  : Lemma (requires un_all sq sw vb wb su /\ c < 128 /\ tb + 128 <= 524288 /\
                    u == tof (acc1 su c) /\ ot == tof (acc1 su ((c + 64) % 128)) /\
                    other == (if c < 64 then neg ot else ot) /\
                    co == tof (acc1 scs (tb + c)) /\ si == tof (acc1 ssn (tb + c)))
          (ensures rot_ok sq sw scs ssn vb wb tb c (tob (add (tof (tob (mul u co))) (tof (tob (mul other si))))))
  = let aux (qkr wr cr sr : S.seq real)
      : Lemma (requires capx sq qkr /\ capx sw wr /\ capx scs cr /\ capx ssn sr)
              (ensures tob (add (tof (tob (mul u co))) (tof (tob (mul other si)))) %~ rotr qkr wr cr sr vb wb tb c)
      = rot_x sq sw scs ssn vb wb tb c su u ot other co si qkr wr cr sr
    in
    let aux2 (qkr wr cr sr : S.seq real)
      : Lemma (capx sq qkr /\ capx sw wr /\ capx scs cr /\ capx ssn sr ==>
               tob (add (tof (tob (mul u co))) (tof (tob (mul other si)))) %~ rotr qkr wr cr sr vb wb tb c)
      = Classical.move_requires (aux qkr wr cr) sr
    in
    Classical.forall_intro_4 aux2;
    reveal_opaque (`%rot_ok) (rot_ok sq sw scs ssn vb wb tb c (tob (add (tof (tob (mul u co))) (tof (tob (mul other si))))))

(* ---- output index maps ---- *)
noextract
let qvb (j : nat) : nat = let rr = j / 128 in ((rr % 16) / 2) * 4096 + (rr / 16) * 512 + ((rr % 16) % 2) * 128
noextract
let qtb (off j : nat) : nat = (off + ((j / 128) % 16) / 2) * 128
noextract
let kvb (j : nat) : nat = let rr = j / 128 in (rr % 8) * 4096 + (rr / 8) * 512 + 256
noextract
let ktb (off j : nat) : nat = (off + (j / 128) % 8) * 128

noextract
let qb_ok (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (off j : nat) (v : bf16) : prop =
  rot_ok sq sw scs ssn (qvb j) 0 (qtb off j) (j % 128) v
noextract
let kb_ok (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (off j : nat) (v : bf16) : prop =
  rot_ok sq sw scs ssn (kvb j) 128 (ktb off j) (j % 128) v

(* row iv = h*24 + r of the kernel loop *)
noextract
let r_tok (iv : nat) : nat = if iv % 24 < 16 then (iv % 24) / 2 else iv % 24 - 16
noextract
let r_pt (iv : nat) : nat = if iv % 24 < 16 then (iv % 24) % 2 else 2
noextract
let r_vb (iv : nat) : nat = r_tok iv * 4096 + (iv / 24) * 512 + r_pt iv * 128
noextract
let r_wb (iv : nat) : nat = if iv % 24 < 16 then 0 else 128
noextract
let r_tb (off iv : nat) : nat = (off + r_tok iv) * 128

noextract
let qdone (iv j : nat) : bool = ((j / 128) / 16) * 24 + (j / 128) % 16 < iv
noextract
let kdone (iv j : nat) : bool = ((j / 128) / 8) * 24 + 16 + (j / 128) % 8 < iv
noextract
let qcur (iv c j : nat) : bool = iv % 24 < 16 && j / 128 = (iv / 24) * 16 + iv % 24 && j % 128 < c
noextract
let kcur (iv c j : nat) : bool = iv % 24 >= 16 && j / 128 = (iv / 24) * 8 + (iv % 24 - 16) && j % 128 < c

noextract
let qinv (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (off iv c : nat)
  (sqb : chest1 bf16 16384) : prop =
  forall (j:natlt 16384). (qdone iv j || qcur iv c j) ==> qb_ok sq sw scs ssn off j (acc1 sqb j)
noextract
let kinv (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (off iv c : nat)
  (skb : chest1 bf16 8192) : prop =
  forall (j:natlt 8192). (kdone iv j || kcur iv c j) ==> kb_ok sq sw scs ssn off j (acc1 skb j)

noextract
let dm128 (j : nat) : Lemma (j == (j / 128) * 128 + j % 128) = ML.euclidean_division_definition j 128

noextract
let rowq_map (off iv c : nat)
  : Lemma (requires iv < 192 /\ iv % 24 < 16 /\ c < 128)
          (ensures (let j = ((iv / 24) * 16 + iv % 24) * 128 + c in
                    j < 16384 /\ j / 128 == (iv / 24) * 16 + iv % 24 /\ j % 128 == c /\
                    qvb j == r_vb iv /\ qtb off j == r_tb off iv))
  = let rr = (iv / 24) * 16 + iv % 24 in
    ML.lemma_div_plus c rr 128; ML.small_div c 128;
    ML.lemma_mod_plus c rr 128; ML.small_mod c 128;
    ML.lemma_div_plus (iv % 24) (iv / 24) 16; ML.small_div (iv % 24) 16;
    ML.lemma_mod_plus (iv % 24) (iv / 24) 16; ML.small_mod (iv % 24) 16

noextract
let rowk_map (off iv c : nat)
  : Lemma (requires iv < 192 /\ iv % 24 >= 16 /\ c < 128)
          (ensures (let j = ((iv / 24) * 8 + (iv % 24 - 16)) * 128 + c in
                    j < 8192 /\ j / 128 == (iv / 24) * 8 + (iv % 24 - 16) /\ j % 128 == c /\
                    kvb j == r_vb iv /\ ktb off j == r_tb off iv))
  = let rr = (iv / 24) * 8 + (iv % 24 - 16) in
    ML.lemma_div_plus c rr 128; ML.small_div c 128;
    ML.lemma_mod_plus c rr 128; ML.small_mod c 128;
    ML.lemma_div_plus (iv % 24 - 16) (iv / 24) 8; ML.small_div (iv % 24 - 16) 8;
    ML.lemma_mod_plus (iv % 24 - 16) (iv / 24) 8; ML.small_mod (iv % 24 - 16) 8

noextract
let q_step (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (off iv c : nat)
  (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192) (v : bf16)
  : Lemma (requires iv < 192 /\ iv % 24 < 16 /\ c < 128 /\ qinv sq sw scs ssn off iv c sqb /\ kinv sq sw scs ssn off iv c skb /\
                    rot_ok sq sw scs ssn (r_vb iv) (r_wb iv) (r_tb off iv) c v)
          (ensures (let j = ((iv / 24) * 16 + iv % 24) * 128 + c in
                    j < 16384 /\
                    qinv sq sw scs ssn off iv (c + 1) (upd sqb (j, ()) v) /\ kinv sq sw scs ssn off iv (c + 1) skb))
  = rowq_map off iv c;
    let j0 = ((iv / 24) * 16 + iv % 24) * 128 + c in
    let aux (j:natlt 16384) : Lemma (requires qcur iv (c + 1) j /\ j <> j0) (ensures qcur iv c j) =
      dm128 j; dm128 j0
    in
    Classical.forall_intro (Classical.move_requires aux)

noextract
let k_step (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (off iv c : nat)
  (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192) (v : bf16)
  : Lemma (requires iv < 192 /\ iv % 24 >= 16 /\ c < 128 /\ qinv sq sw scs ssn off iv c sqb /\ kinv sq sw scs ssn off iv c skb /\
                    rot_ok sq sw scs ssn (r_vb iv) (r_wb iv) (r_tb off iv) c v)
          (ensures (let j = ((iv / 24) * 8 + (iv % 24 - 16)) * 128 + c in
                    j < 8192 /\
                    qinv sq sw scs ssn off iv (c + 1) sqb /\ kinv sq sw scs ssn off iv (c + 1) (upd skb (j, ()) v)))
  = rowk_map off iv c;
    let j0 = ((iv / 24) * 8 + (iv % 24 - 16)) * 128 + c in
    let aux (j:natlt 8192) : Lemma (requires kcur iv (c + 1) j /\ j <> j0) (ensures kcur iv c j) =
      dm128 j; dm128 j0
    in
    Classical.forall_intro (Classical.move_requires aux)

noextract
let row_next (sq : chest1 bf16 32768) (sw : chest1 bf16 256) (scs ssn : chest1 bf16 524288) (off iv : nat)
  (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192)
  : Lemma (requires iv < 192 /\ qinv sq sw scs ssn off iv 128 sqb /\ kinv sq sw scs ssn off iv 128 skb)
          (ensures qinv sq sw scs ssn off (iv + 1) 0 sqb /\ kinv sq sw scs ssn off (iv + 1) 0 skb)
  = let auxq (j:natlt 16384) : Lemma (requires qdone (iv + 1) j) (ensures qdone iv j || qcur iv 128 j) =
      let rr = j / 128 in
      if (rr / 16) * 24 + rr % 16 = iv then begin
        ML.lemma_div_plus (rr % 16) (rr / 16) 24; ML.small_div (rr % 16) 24;
        ML.lemma_mod_plus (rr % 16) (rr / 16) 24; ML.small_mod (rr % 16) 24;
        ML.euclidean_division_definition rr 16
      end
    in
    let auxk (j:natlt 8192) : Lemma (requires kdone (iv + 1) j) (ensures kdone iv j || kcur iv 128 j) =
      let rr = j / 128 in
      if (rr / 8) * 24 + 16 + rr % 8 = iv then begin
        ML.lemma_div_plus (16 + rr % 8) (rr / 8) 24; ML.small_div (16 + rr % 8) 24;
        ML.lemma_mod_plus (16 + rr % 8) (rr / 8) 24; ML.small_mod (16 + rr % 8) 24;
        ML.euclidean_division_definition rr 8
      end
    in
    Classical.forall_intro (Classical.move_requires auxq);
    Classical.forall_intro (Classical.move_requires auxk)

noextract
let all_done (j : nat) : Lemma ((j < 16384 ==> qdone 192 j) /\ (j < 8192 ==> kdone 192 j)) = ()

noextract
let row_vb (iv : nat) : Lemma (requires iv < 192) (ensures vb_ok (r_vb iv) (r_tok iv) (iv / 24) (r_pt iv)) = ()

(* ---- Pulse ---- *)
inline_for_extraction noextract
fn rdm (qkv : array1 bf16 (l1_forward 32768)) (j : sz{SZ.v j < 32768}) (tok : sz) (#fq : perm) (#sq : erased (chest1 bf16 32768))
  preserves qkv |-> Frac fq sq
  returns v : f32
  ensures pure (v == mval sq (SZ.v tok) (SZ.v j))
{
  let x = rd qkv j;
  (if tok <^ 7sz then tof x else fz)
}

inline_for_extraction noextract
fn pq_one (qkv : array1 bf16 (l1_forward 32768)) (vb tok tv : sz) (h pt : erased nat)
  (#fq : perm) (#sq : erased (chest1 bf16 32768))
  preserves gpu ** qkv |-> Frac fq sq
  requires pure (vb_ok (SZ.v vb) (SZ.v tok) h pt /\ SZ.v tv < 32)
  returns a3 : f32
  ensures pure (pq_ok sq (SZ.v vb) (SZ.v tv) a3)
{
  let v0 = rdm qkv (vb +^ tv) tok;
  let a0 = Fast.fma_rn v0 v0 zero;
  let v1 = rdm qkv (vb +^ tv +^ 32sz) tok;
  let a1 = Fast.fma_rn v1 v1 a0;
  let v2 = rdm qkv (vb +^ tv +^ 64sz) tok;
  let a2 = Fast.fma_rn v2 v2 a1;
  let v3 = rdm qkv (vb +^ tv +^ 96sz) tok;
  let a3 = Fast.fma_rn v3 v3 a2;
  pq_lem sq (SZ.v vb) (SZ.v tok) h pt (SZ.v tv) v0 v1 v2 v3 a0 a1 a2 a3;
  a3
}

inline_for_extraction noextract
fn qk_inv (qkv : array1 bf16 (l1_forward 32768)) (part : array1 f32 (l1_forward 32)) (vb tok : sz) (h pt : erased nat)
  (#fq : perm) (#sq : erased (chest1 bf16 32768)) (#sp0 : erased (chest1 f32 32))
  preserves gpu ** qkv |-> Frac fq sq
  requires part |-> sp0 ** pure (vb_ok (SZ.v vb) (SZ.v tok) h pt)
  returns inv : f32
  ensures (exists* sp. part |-> sp) ** pure (qi_ok sq (SZ.v vb) inv)
{
  let mut t : sz = 0sz;
  while (!t <^ 32sz)
    invariant live t ** pure (SZ.v !t <= 32)
    invariant exists* sp. part |-> sp **
      pure (forall (j:natlt 32). j < SZ.v !t ==> pq_ok sq (SZ.v vb) j (acc1 sp j))
    decreases (32 - SZ.v !t)
  {
    let tv = !t;
    let a3 = pq_one qkv vb tok tv h pt;
    wr part tv a3;
    t := tv +^ 1sz;
  };
  with s1. assert (part |-> s1);
  tree_red part 0sz 32sz;
  let p0 = rd part 0sz;
  let ep = mk_eps ();
  qi_lem sq (SZ.v vb) s1 ep;
  rsqrt (add (div p0 (of_int 128L)) ep)
}

inline_for_extraction noextract
fn qk_un (w : array1 bf16 (l1_forward 256)) (qkv : array1 bf16 (l1_forward 32768)) (un : array1 bf16 (l1_forward 128))
  (vb tok wb : sz) (h pt : erased nat) (inv : f32)
  (#fw #fq : perm) (#sw : erased (chest1 bf16 256)) (#sq : erased (chest1 bf16 32768)) (#su0 : erased (chest1 bf16 128))
  preserves w |-> Frac fw sw ** qkv |-> Frac fq sq
  requires un |-> su0 ** pure (vb_ok (SZ.v vb) (SZ.v tok) h pt /\ SZ.v wb + 128 <= 256 /\ qi_ok sq (SZ.v vb) inv)
  ensures exists* su. un |-> su ** pure (un_all sq sw (SZ.v vb) (SZ.v wb) su)
{
  let mut c : sz = 0sz;
  while (!c <^ 128sz)
    invariant live c ** pure (SZ.v !c <= 128)
    invariant exists* su. un |-> su **
      pure (forall (j:natlt 128). j < SZ.v !c ==> un_ok sq sw (SZ.v vb) (SZ.v wb) j (acc1 su j))
    decreases (128 - SZ.v !c)
  {
    let cv = !c;
    vb_div (SZ.v vb) (SZ.v tok) h pt (SZ.v cv);
    let v = rdm qkv (vb +^ cv) tok;
    let scd = tob (mul v inv);
    let wc = tof (rd w (wb +^ cv));
    un_lem sq sw (SZ.v vb) (SZ.v tok) h pt (SZ.v wb) (SZ.v cv) inv v wc;
    wr un cv (tob (mul (tof scd) wc));
    c := cv +^ 1sz;
  };
}

inline_for_extraction noextract
fn rot_wr (qb : array1 bf16 (l1_forward 16384)) (kb : array1 bf16 (l1_forward 8192))
  (iv : erased nat{iv < 192}) (h r cv : sz) (rot : bf16)
  (sq : erased (chest1 bf16 32768)) (sw : erased (chest1 bf16 256)) (scs ssn : erased (chest1 bf16 524288)) (off : erased nat)
  (#sqb : erased (chest1 bf16 16384)) (#skb : erased (chest1 bf16 8192))
  requires qb |-> sqb ** kb |-> skb **
    pure (SZ.v h == iv / 24 /\ SZ.v r == iv % 24 /\ SZ.v cv < 128 /\
          rot_ok sq sw scs ssn (r_vb iv) (r_wb iv) (r_tb off iv) (SZ.v cv) rot /\
          qinv sq sw scs ssn off iv (SZ.v cv) sqb /\ kinv sq sw scs ssn off iv (SZ.v cv) skb)
  ensures exists* sqb' skb'. qb |-> sqb' ** kb |-> skb' **
    pure (qinv sq sw scs ssn off iv (SZ.v cv + 1) sqb' /\ kinv sq sw scs ssn off iv (SZ.v cv + 1) skb')
{
  if (r <^ 16sz) {
    q_step sq sw scs ssn off iv (SZ.v cv) sqb skb rot;
    wr qb ((h *^ 16sz +^ r) *^ 128sz +^ cv) rot;
  } else {
    k_step sq sw scs ssn off iv (SZ.v cv) sqb skb rot;
    wr kb ((h *^ 8sz +^ (r -^ 16sz)) *^ 128sz +^ cv) rot;
  }
}

inline_for_extraction noextract
fn qk_rope (un : array1 bf16 (l1_forward 128)) (cs sn : array1 bf16 (l1_forward 524288))
  (qb : array1 bf16 (l1_forward 16384)) (kb : array1 bf16 (l1_forward 8192))
  (iv : erased nat{iv < 192}) (h r tb : sz)
  (sq : erased (chest1 bf16 32768)) (sw : erased (chest1 bf16 256)) (off : erased nat)
  (#fc #fs : perm) (#scs #ssn : erased (chest1 bf16 524288)) (#su : erased (chest1 bf16 128))
  (#sqb0 : erased (chest1 bf16 16384)) (#skb0 : erased (chest1 bf16 8192))
  preserves un |-> su ** cs |-> Frac fc scs ** sn |-> Frac fs ssn
  requires qb |-> sqb0 ** kb |-> skb0 **
    pure (SZ.v h == iv / 24 /\ SZ.v r == iv % 24 /\ SZ.v tb == r_tb off iv /\ SZ.v tb + 128 <= 524288 /\
          un_all sq sw (r_vb iv) (r_wb iv) su /\
          qinv sq sw scs ssn off iv 0 sqb0 /\ kinv sq sw scs ssn off iv 0 skb0)
  ensures exists* sqb skb. qb |-> sqb ** kb |-> skb **
    pure (qinv sq sw scs ssn off iv 128 sqb /\ kinv sq sw scs ssn off iv 128 skb)
{
  let mut c3 : sz = 0sz;
  while (!c3 <^ 128sz)
    invariant live c3 ** pure (SZ.v !c3 <= 128)
    invariant exists* sqb skb. qb |-> sqb ** kb |-> skb **
      pure (qinv sq sw scs ssn off iv (SZ.v !c3) sqb /\ kinv sq sw scs ssn off iv (SZ.v !c3) skb)
    decreases (128 - SZ.v !c3)
  {
    let cv = !c3;
    let ot = tof (rd un ((cv +^ 64sz) %% 128sz));
    let other = (if cv <^ 64sz then neg ot else ot);
    let u = tof (rd un cv);
    let co = tof (rd cs (tb +^ cv));
    let si = tof (rd sn (tb +^ cv));
    let a = tob (mul u co);
    let bb = tob (mul other si);
    rot_lem sq sw scs ssn (r_vb iv) (r_wb iv) (SZ.v tb) (SZ.v cv) su u ot other co si;
    let rot = tob (add (tof a) (tof bb));
    rot_wr qb kb iv h r cv rot sq sw scs ssn off;
    c3 := cv +^ 1sz;
  };
}

inline_for_extraction noextract
fn k_qknorm (w : array1 bf16 (l1_forward 256)) (qkv : array1 bf16 (l1_forward 32768))
  (cs sn : array1 bf16 (l1_forward 524288)) (off : sz{SZ.v off <= 4088})
  (part : array1 f32 (l1_forward 32)) (un : array1 bf16 (l1_forward 128))
  (qb : array1 bf16 (l1_forward 16384)) (kb : array1 bf16 (l1_forward 8192))
  (#fw #fq #fc #fs : perm) (#sw : erased (chest1 bf16 256)) (#sq : erased (chest1 bf16 32768))
  (#scs #ssn : erased (chest1 bf16 524288))
  (#sp0 : erased (chest1 f32 32)) (#su0 : erased (chest1 bf16 128))
  (#sqb0 : erased (chest1 bf16 16384)) (#skb0 : erased (chest1 bf16 8192))
  preserves gpu ** w |-> Frac fw sw ** qkv |-> Frac fq sq ** cs |-> Frac fc scs ** sn |-> Frac fs ssn
  requires part |-> sp0 ** un |-> su0 ** qb |-> sqb0 ** kb |-> skb0
  ensures exists* sp su sqb skb. part |-> sp ** un |-> su ** qb |-> sqb ** kb |-> skb **
    pure ((forall (j:natlt 16384). qb_ok sq sw scs ssn (SZ.v off) j (acc1 sqb j)) /\
          (forall (j:natlt 8192). kb_ok sq sw scs ssn (SZ.v off) j (acc1 skb j)))
{
  let mut i : sz = 0sz;
  while (!i <^ 192sz)
    invariant live i ** pure (SZ.v !i <= 192)
    invariant exists* sp su sqb skb. part |-> sp ** un |-> su ** qb |-> sqb ** kb |-> skb **
      pure (qinv sq sw scs ssn (SZ.v off) (SZ.v !i) 0 sqb /\ kinv sq sw scs ssn (SZ.v off) (SZ.v !i) 0 skb)
    decreases (192 - SZ.v !i)
  {
    let iv = !i;
    let h = iv /^ 24sz;
    let r = iv %% 24sz;
    let tok = (if r <^ 16sz then r /^ 2sz else r -^ 16sz);
    let pt = (if r <^ 16sz then r %% 2sz else 2sz);
    let vb = tok *^ 4096sz +^ h *^ 512sz +^ pt *^ 128sz;
    row_vb (SZ.v iv);
    assert pure (SZ.v tok == r_tok (SZ.v iv) /\ SZ.v pt == r_pt (SZ.v iv) /\ SZ.v vb == r_vb (SZ.v iv));
    let inv = qk_inv qkv part vb tok (SZ.v h) (SZ.v pt);
    let wb = (if r <^ 16sz then 0sz else 128sz);
    qk_un w qkv un vb tok wb (SZ.v h) (SZ.v pt) inv;
    let tb = (off +^ tok) *^ 128sz;
    qk_rope un cs sn qb kb (SZ.v iv) h r tb sq sw (SZ.v off);
    with sqb skb. assert (qb |-> sqb ** kb |-> skb);
    row_next sq sw scs ssn (SZ.v off) (SZ.v iv) sqb skb;
    i := iv +^ 1sz;
  };
  with sqb skb. assert (qb |-> sqb ** kb |-> skb);
  Classical.forall_intro all_done;
}

(* ===== BR0 ===== *)

(* ---- index arithmetic ---- *)
noextract
let dm (a b : nat) (m : pos)
  : Lemma (requires b < m) (ensures (a * m + b) / m == a /\ (a * m + b) % m == b)
  = ML.lemma_div_plus b a m;
    ML.small_div b m;
    ML.lemma_mod_plus b a m;
    ML.small_mod b m

noextract
let ed (j : nat) (m : pos) : Lemma (j == (j / m) * m + j % m /\ j % m < m) = ML.euclidean_division_definition j m

noextract
let at_win (s : S.seq real) (o n i : int)
  : Lemma (requires 0 <= o /\ 0 <= n /\ o + n <= S.length s /\ 0 <= i /\ i < n)
          (ensures at (window s o n) i == at s (o + i))
  = ()

(* ---- extensionality ---- *)
noextract
let rec sumsq_ext (n : nat) (a : S.seq real) (oa : int) (b : S.seq real) (ob : int)
  : Lemma (requires forall (k:nat). k < n ==> at a (oa + k) == at b (ob + k))
          (ensures sumsq n a oa == sumsq n b ob)
          (decreases n)
  = if n = 0 then () else sumsq_ext (n - 1) a oa b ob

noextract
let rms_ext (d : pos) (a : S.seq real) (oa : int) (b : S.seq real) (ob : int)
  : Lemma (requires forall (k:nat). k < d ==> at a (oa + k) == at b (ob + k))
          (ensures rms_inv d a oa == rms_inv d b ob)
  = sumsq_ext d a oa b ob

noextract
let rec sumexp_ext (n : nat) (a : S.seq real) (oa : int) (b : S.seq real) (ob : int)
  : Lemma (requires forall (k:nat). k < n ==> at a (oa + k) == at b (ob + k))
          (ensures sumexp n a oa == sumexp n b ob)
          (decreases n)
  = if n = 0 then () else sumexp_ext (n - 1) a oa b ob

(* ---- chests vs sequences ---- *)
noextract
let capx_ext (#n : nat) (s : chest1 bf16 n) (r r' : S.seq real)
  : Lemma (requires capx s r /\ (forall (i:nat). i < n ==> at r i == at r' i)) (ensures capx s r')
  = ()

noextract
let seq_capx (#n : nat) (s : chest1 bf16 n) (r : S.seq real)
  : Lemma (requires seq_approximates #bf16 (chest1_to_seq s) r) (ensures capx s r /\ S.length r == n)
  = let aux (i : natlt n) : Lemma (acc1 s i %~ at r i) =
      assert (S.index (chest1_to_seq s) i == acc1 s i)
    in
    Classical.forall_intro aux

noextract
let win_apx (#n : nat) (s : chest1 bf16 n) (o m : nat) (r : S.seq real)
  : Lemma (requires o + m <= n /\ S.length r == m /\ (forall (i:nat). i < m ==> acc1 s (o + i) %~ at r i))
          (ensures seq_approximates #bf16 (window (chest1_to_seq s) o m) r)
  = let w = window (chest1_to_seq s) o m in
    let aux (i : nat{i < m}) : Lemma (S.index w i %~ S.index r i) =
      assert (S.index w i == acc1 s (o + i))
    in
    Classical.forall_intro aux

(* ---- silu ---- *)
noextract
let silu_apx (g : f32) (gr : real)
  : Lemma (requires g %~ gr) (ensures div g (add one (fexp (neg g))) %~ silu gr)
  = zo_apx ();
    let _ = approx_neg #f32 in
    let _ = approx_fexp #f32 in
    let _ = approx_div #f32 in
    assert (neg g %~ (0.0R -. gr));
    assert (fexp (neg g) %~ Exp.exp (0.0R -. gr));
    Exp.exp_positive (0.0R -. gr);
    assert (add one (fexp (neg g)) %~ (1.0R +. Exp.exp (0.0R -. gr)));
    assert (div g (add one (fexp (neg g))) %~ (gr /. (1.0R +. Exp.exp (0.0R -. gr))))

(* ===== BR1 ===== *)

noextract
let lin_bnd (a n : nat) (m : pos) : Lemma (requires a < n) (ensures a * m + m <= n * m) =
  ML.lemma_mult_le_right m (a + 1) n;
  ML.distributivity_add_left a 1 m

noextract
let sdot_win (n : nat) (a : S.seq real) (oa la ao : nat) (b : S.seq real) (ob lb bo : nat)
  : Lemma (requires oa + la <= S.length a /\ ob + lb <= S.length b /\ ao + n <= la /\ bo + n <= lb)
          (ensures sdot n (window a oa la) ao 1 (window b ob lb) bo 1 == sdot n a (oa + ao) 1 b (ob + bo) 1)
  = let aux (k : nat) : Lemma (k < n ==>
        at (window a oa la) (ao + k * 1) *. at (window b ob lb) (bo + k * 1) ==
        at a ((oa + ao) + k * 1) *. at b ((ob + bo) + k * 1)) = ()
    in
    Classical.forall_intro aux;
    sdot_ext n (window a oa la) ao 1 (window b ob lb) bo 1 a (oa + ao) 1 b (ob + bo) 1

(* ---- the qkv GEMM ideal: layout-1 weight row of feature q ---- *)
noextract
let wr1 (q : nat) : nat =
  if q % 512 < 256 then (q / 512) * 256 + q % 512
  else 2048 + ((q % 512 - 256) / 128) * 1024 + (q / 512) * 128 + q % 128

noextract
let qkr (r0 r8 : S.seq real) : S.seq real =
  S.init 32768 (fun (j:nat{j < 32768}) -> sdot 1024 r0 (wr1 (j % 4096) * 1024) 1 r8 ((j / 4096) * 1024) 1)

noextract
let lens (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : prop =
  S.length r0 == 4194304 /\ S.length r1 == 256 /\ S.length r8 == 8192 /\
  S.length r9 == 524288 /\ S.length r10 == 524288 /\ off <= 4088

(* spec intermediates *)
noextract
let lq (r0 r8 : S.seq real) = linear 7 2048 1024 (window r0 0 2097152) (window r8 0 7168)
noextract
let lk (r0 r8 : S.seq real) = linear 7 1024 1024 (window r0 2097152 1048576) (window r8 0 7168)
noextract
let lv (r0 r8 : S.seq real) = linear 7 1024 1024 (window r0 3145728 1048576) (window r8 0 7168)
noextract
let xq (r0 r1 r8 : S.seq real) = rmsnorm_rows 112 128 (lq r0 r8) (window r1 0 128)
noextract
let xk (r0 r1 r8 : S.seq real) = rmsnorm_rows 56 128 (lk r0 r8) (window r1 128 128)
noextract
let cw (r9 : S.seq real) (off : nat) = window r9 (off * 128) 896
noextract
let qq (r0 r1 r8 r9 r10 : S.seq real) (off : nat) = rope 16 (xq r0 r1 r8) (cw r9 off) (cw r10 off)
noextract
let kq (r0 r1 r8 r9 r10 : S.seq real) (off : nat) = rope 8 (xk r0 r1 r8) (cw r9 off) (cw r10 off)

noextract
let qkr_at (r0 r8 : S.seq real) (tk q : nat)
  : Lemma (requires tk < 8 /\ q < 4096)
          (ensures at (qkr r0 r8) (tk * 4096 + q) == sdot 1024 r0 (wr1 q * 1024) 1 r8 (tk * 1024) 1)
  = dm tk q 4096

(* E1: q features *)
noextract
let e1 (r0 r8 : S.seq real) (t h p c : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r8 == 8192 /\ t < 7 /\ h < 8 /\ p < 2 /\ c < 128)
          (ensures at (mk_m (qkr r0 r8)) (t * 4096 + h * 512 + p * 128 + c) ==
                   at (lq r0 r8) ((t * 16 + (2 * h + p)) * 128 + c))
  = let q = h * 512 + p * 128 + c in
    let j = t * 4096 + q in
    dm t q 4096;
    mk_m_at (qkr r0 r8) j;
    qkr_at r0 r8 t q;
    dm h (p * 128 + c) 512;
    assert (wr1 q == (2 * h + p) * 128 + c);
    let o = (2 * h + p) * 128 + c in
    let i = (t * 16 + (2 * h + p)) * 128 + c in
    assert (i == t * 2048 + o);
    dm t o 2048;
    reveal_opaque (`%linear) (linear 7 2048 1024 (window r0 0 2097152) (window r8 0 7168));
    assert (at (lq r0 r8) i == sdot 1024 (window r0 0 2097152) (o * 1024) 1 (window r8 0 7168) (t * 1024) 1);
    lin_bnd o 2048 1024;
    lin_bnd t 7 1024;
    sdot_win 1024 r0 0 2097152 (o * 1024) r8 0 7168 (t * 1024)

(* E2: k features (pt = 2) *)
noextract
let e2 (r0 r8 : S.seq real) (k h c : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r8 == 8192 /\ k < 7 /\ h < 8 /\ c < 128)
          (ensures at (mk_m (qkr r0 r8)) (k * 4096 + h * 512 + 256 + c) ==
                   at (lk r0 r8) ((k * 8 + h) * 128 + c))
  = let q = h * 512 + 256 + c in
    let j = k * 4096 + q in
    dm k q 4096;
    mk_m_at (qkr r0 r8) j;
    qkr_at r0 r8 k q;
    dm h (256 + c) 512;
    dm h c 128;
    assert (wr1 q == 2048 + h * 128 + c);
    let o = h * 128 + c in
    let i = (k * 8 + h) * 128 + c in
    assert (i == k * 1024 + o);
    dm k o 1024;
    reveal_opaque (`%linear) (linear 7 1024 1024 (window r0 2097152 1048576) (window r8 0 7168));
    assert (at (lk r0 r8) i == sdot 1024 (window r0 2097152 1048576) (o * 1024) 1 (window r8 0 7168) (k * 1024) 1);
    lin_bnd o 1024 1024;
    lin_bnd k 7 1024;
    sdot_win 1024 r0 2097152 1048576 (o * 1024) r8 0 7168 (k * 1024)

(* E3: v features (pt = 3) *)
noextract
let e3 (r0 r8 : S.seq real) (k h c : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r8 == 8192 /\ k < 7 /\ h < 8 /\ c < 128)
          (ensures at (qkr r0 r8) (k * 4096 + h * 512 + 384 + c) ==
                   at (lv r0 r8) (k * 1024 + h * 128 + c))
  = let q = h * 512 + 384 + c in
    dm k q 4096;
    qkr_at r0 r8 k q;
    dm h (384 + c) 512;
    dm 1 c 128;
    dm h c 128;
    assert (wr1 q == 3072 + h * 128 + c);
    let o = h * 128 + c in
    dm k o 1024;
    reveal_opaque (`%linear) (linear 7 1024 1024 (window r0 3145728 1048576) (window r8 0 7168));
    lin_bnd o 1024 1024;
    lin_bnd k 7 1024;
    sdot_win 1024 r0 3145728 1048576 (o * 1024) r8 0 7168 (k * 1024)

noextract
let rcomm (x y : real) : Lemma (x *. y == y *. x) = ()

noextract
let rms_at (rows : nat) (d : pos) (x w : S.seq real) (i : nat)
  : Lemma (requires i < rows * d)
          (ensures at (rmsnorm_rows rows d x w) i == at w (i % d) *. (at x i *. rms_inv d x ((i / d) * d)))
  = reveal_opaque (`%rmsnorm_rows) (rmsnorm_rows rows d x w)

(* E4: normalized q / k rows *)
noextract
let e4qr (r0 r8 : S.seq real) (t h p : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r8 == 8192 /\ t < 7 /\ h < 8 /\ p < 2)
          (ensures rms_inv 128 (mk_m (qkr r0 r8)) (t * 4096 + h * 512 + p * 128) ==
                   rms_inv 128 (lq r0 r8) ((t * 16 + (2 * h + p)) * 128))
  = let vb = t * 4096 + h * 512 + p * 128 in
    let oi = (t * 16 + (2 * h + p)) * 128 in
    let aux (c' : nat) : Lemma (c' < 128 ==> at (mk_m (qkr r0 r8)) (vb + c') == at (lq r0 r8) (oi + c')) =
      if c' < 128 then e1 r0 r8 t h p c' else ()
    in
    Classical.forall_intro aux;
    rms_ext 128 (mk_m (qkr r0 r8)) vb (lq r0 r8) oi

noextract
let e4q (r0 r1 r8 : S.seq real) (t h p c : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r1 == 256 /\ S.length r8 == 8192 /\
                    t < 7 /\ h < 8 /\ p < 2 /\ c < 128)
          (ensures unr (qkr r0 r8) r1 (t * 4096 + h * 512 + p * 128) 0 c ==
                   at (xq r0 r1 r8) ((t * 16 + (2 * h + p)) * 128 + c))
  = let vb = t * 4096 + h * 512 + p * 128 in
    let rw = t * 16 + (2 * h + p) in
    let oi = rw * 128 in
    assert (rw < 112);
    dm rw c 128;
    assert (oi + c < 112 * 128);
    e4qr r0 r8 t h p;
    rms_at 112 128 (lq r0 r8) (window r1 0 128) (oi + c);
    at_win r1 0 128 c;
    e1 r0 r8 t h p c;
    let a = at (mk_m (qkr r0 r8)) (vb + c) in
    let ri = rms_inv 128 (lq r0 r8) oi in
    assert (unr (qkr r0 r8) r1 vb 0 c == (a *. ri) *. at r1 c);
    assert (at (xq r0 r1 r8) (oi + c) == at r1 c *. (a *. ri));
    rcomm (a *. ri) (at r1 c)

noextract
let e4kr (r0 r8 : S.seq real) (k h : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r8 == 8192 /\ k < 7 /\ h < 8)
          (ensures rms_inv 128 (mk_m (qkr r0 r8)) (k * 4096 + h * 512 + 256) ==
                   rms_inv 128 (lk r0 r8) ((k * 8 + h) * 128))
  = let vb = k * 4096 + h * 512 + 256 in
    let oi = (k * 8 + h) * 128 in
    let aux (c' : nat) : Lemma (c' < 128 ==> at (mk_m (qkr r0 r8)) (vb + c') == at (lk r0 r8) (oi + c')) =
      if c' < 128 then e2 r0 r8 k h c' else ()
    in
    Classical.forall_intro aux;
    rms_ext 128 (mk_m (qkr r0 r8)) vb (lk r0 r8) oi

noextract
let e4k (r0 r1 r8 : S.seq real) (k h c : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r1 == 256 /\ S.length r8 == 8192 /\
                    k < 7 /\ h < 8 /\ c < 128)
          (ensures unr (qkr r0 r8) r1 (k * 4096 + h * 512 + 256) 128 c ==
                   at (xk r0 r1 r8) ((k * 8 + h) * 128 + c))
  = let vb = k * 4096 + h * 512 + 256 in
    let rw = k * 8 + h in
    let oi = rw * 128 in
    assert (rw < 56);
    dm rw c 128;
    assert (oi + c < 56 * 128);
    e4kr r0 r8 k h;
    rms_at 56 128 (lk r0 r8) (window r1 128 128) (oi + c);
    at_win r1 128 128 c;
    e2 r0 r8 k h c;
    let a = at (mk_m (qkr r0 r8)) (vb + c) in
    let ri = rms_inv 128 (lk r0 r8) oi in
    assert (unr (qkr r0 r8) r1 vb 128 c == (a *. ri) *. at r1 (128 + c));
    assert (at (xk r0 r1 r8) (oi + c) == at r1 (128 + c) *. (a *. ri));
    rcomm (a *. ri) (at r1 (128 + c))

(* E5: rotated rows *)
noextract
let e5q (r0 r1 r8 r9 r10 : S.seq real) (off t h p c : nat)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\ t < 7 /\ h < 8 /\ p < 2 /\ c < 128)
          (ensures rotr (qkr r0 r8) r1 r9 r10 (t * 4096 + h * 512 + p * 128) 0 ((off + t) * 128) c ==
                   at (qq r0 r1 r8 r9 r10 off) ((t * 16 + (2 * h + p)) * 128 + c))
  = let i = (t * 16 + (2 * h + p)) * 128 + c in
    assert (i == t * 2048 + ((2 * h + p) * 128 + c));
    dm t ((2 * h + p) * 128 + c) 2048;
    dm (t * 16 + (2 * h + p)) c 128;
    reveal_opaque (`%rope) (rope 16 (xq r0 r1 r8) (cw r9 off) (cw r10 off));
    e4q r0 r1 r8 t h p c;
    if c < 64 then e4q r0 r1 r8 t h p (c + 64) else e4q r0 r1 r8 t h p (c - 64);
    at_win r9 (off * 128) 896 (t * 128 + c);
    at_win r10 (off * 128) 896 (t * 128 + c)

noextract
let e5k (r0 r1 r8 r9 r10 : S.seq real) (off k h c : nat)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\ k < 7 /\ h < 8 /\ c < 128)
          (ensures rotr (qkr r0 r8) r1 r9 r10 (k * 4096 + h * 512 + 256) 128 ((off + k) * 128) c ==
                   at (kq r0 r1 r8 r9 r10 off) ((k * 8 + h) * 128 + c))
  = let i = (k * 8 + h) * 128 + c in
    assert (i == k * 1024 + (h * 128 + c));
    dm k (h * 128 + c) 1024;
    dm (k * 8 + h) c 128;
    reveal_opaque (`%rope) (rope 8 (xk r0 r1 r8) (cw r9 off) (cw r10 off));
    e4k r0 r1 r8 k h c;
    if c < 64 then e4k r0 r1 r8 k h (c + 64) else e4k r0 r1 r8 k h (c - 64);
    at_win r9 (off * 128) 896 (k * 128 + c);
    at_win r10 (off * 128) 896 (k * 128 + c)

(* padded token 7: everything is zero *)
noextract
let e5z (qkr0 wr cr sr : S.seq real) (vb wb tb c : nat)
  : Lemma (requires vb >= 7 * 4096 /\ vb + 128 <= 32768 /\ c < 128)
          (ensures rotr qkr0 wr cr sr vb wb tb c == 0.0R)
  = let aux (c' : nat) : Lemma (c' < 128 ==> unr qkr0 wr vb wb c' == 0.0R) =
      if c' < 128 then mk_m_at qkr0 (vb + c') else ()
    in
    Classical.forall_intro aux

(* ---- ideals of qb and kb ---- *)
noextract
let qi (j : nat) : nat =
  let rr = j / 128 in (((rr % 16) / 2) * 16 + 2 * (rr / 16) + (rr % 16) % 2) * 128 + j % 128
noextract
let ki (j : nat) : nat =
  let rr = j / 128 in ((rr % 8) * 8 + rr / 8) * 128 + j % 128

noextract
let qbr (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : S.seq real =
  S.init 16384 (fun (j:nat{j < 16384}) -> at (qq r0 r1 r8 r9 r10 off) (qi j))
noextract
let kbr (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : S.seq real =
  S.init 8192 (fun (j:nat{j < 8192}) -> at (kq r0 r1 r8 r9 r10 off) (ki j))

noextract
let qb_eq (r0 r1 r8 r9 r10 : S.seq real) (off : nat) (j : nat)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\ j < 16384)
          (ensures rotr (qkr r0 r8) r1 r9 r10 (qvb j) 0 (qtb off j) (j % 128) == at (qbr r0 r1 r8 r9 r10 off) j)
  = let rr = j / 128 in
    let c = j % 128 in
    ed j 128;
    ed rr 16;
    let h = rr / 16 in
    let r = rr % 16 in
    ed r 2;
    let t = r / 2 in
    let p = r % 2 in
    assert (qvb j == t * 4096 + h * 512 + p * 128);
    assert (qtb off j == (off + t) * 128);
    assert (qi j == (t * 16 + (2 * h + p)) * 128 + c);
    if t < 7 then e5q r0 r1 r8 r9 r10 off t h p c
    else e5z (qkr r0 r8) r1 r9 r10 (qvb j) 0 (qtb off j) c

noextract
let kb_eq (r0 r1 r8 r9 r10 : S.seq real) (off : nat) (j : nat)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\ j < 8192)
          (ensures rotr (qkr r0 r8) r1 r9 r10 (kvb j) 128 (ktb off j) (j % 128) == at (kbr r0 r1 r8 r9 r10 off) j)
  = let rr = j / 128 in
    let c = j % 128 in
    ed j 128;
    ed rr 8;
    let h = rr / 8 in
    let k = rr % 8 in
    assert (kvb j == k * 4096 + h * 512 + 256);
    assert (ktb off j == (off + k) * 128);
    assert (ki j == (k * 8 + h) * 128 + c);
    if k < 7 then e5k r0 r1 r8 r9 r10 off k h c
    else e5z (qkr r0 r8) r1 r9 r10 (kvb j) 128 (ktb off j) c

(* ===== BR2 ===== *)

(* ---- scores GEMM (mode 3) ideal ---- *)
noextract
let cqkr (qb kb : S.seq real) : S.seq real =
  S.init 1024 (fun (p:nat{p < 1024}) ->
    sdot 128 qb (((p / 128) * 16 + (p / 8) % 16) * 128) 1 kb (((p / 128) * 8 + p % 8) * 128) 1)

noextract
let cqr (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : S.seq real =
  cqkr (qbr r0 r1 r8 r9 r10 off) (kbr r0 r1 r8 r9 r10 off)

noextract
let asc (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : S.seq real =
  attn_scores (qq r0 r1 r8 r9 r10 off) (kq r0 r1 r8 r9 r10 off)

noextract
let qbr_at (r0 r1 r8 r9 r10 : S.seq real) (off h r c : nat)
  : Lemma (requires h < 8 /\ r < 16 /\ c < 128)
          (ensures at (qbr r0 r1 r8 r9 r10 off) ((h * 16 + r) * 128 + c) ==
                   at (qq r0 r1 r8 r9 r10 off) (((r / 2) * 16 + (2 * h + r % 2)) * 128 + c))
  = let rr = h * 16 + r in
    dm rr c 128;
    dm h r 16

noextract
let kbr_at (r0 r1 r8 r9 r10 : S.seq real) (off h k c : nat)
  : Lemma (requires h < 8 /\ k < 8 /\ c < 128)
          (ensures at (kbr r0 r1 r8 r9 r10 off) ((h * 8 + k) * 128 + c) ==
                   at (kq r0 r1 r8 r9 r10 off) ((k * 8 + h) * 128 + c))
  = let rr = h * 8 + k in
    dm rr c 128;
    dm h k 8

noextract
let s1_dot (r0 r1 r8 r9 r10 : S.seq real) (off h r k : nat)
  : Lemma (requires h < 8 /\ r < 14 /\ k < 7)
          (ensures sdot 128 (qbr r0 r1 r8 r9 r10 off) ((h * 16 + r) * 128) 1 (kbr r0 r1 r8 r9 r10 off) ((h * 8 + k) * 128) 1 ==
                   sdot 128 (qq r0 r1 r8 r9 r10 off) (((r / 2) * 16 + (2 * h + r % 2)) * 128) 1
                            (kq r0 r1 r8 r9 r10 off) ((k * 8 + (2 * h + r % 2) / 2) * 128) 1)
  = ed r 2;
    dm h (r % 2) 2;
    assert ((2 * h + r % 2) / 2 == h);
    let aux (c : nat) : Lemma (c < 128 ==>
        at (qbr r0 r1 r8 r9 r10 off) ((h * 16 + r) * 128 + c * 1) *. at (kbr r0 r1 r8 r9 r10 off) ((h * 8 + k) * 128 + c * 1) ==
        at (qq r0 r1 r8 r9 r10 off) (((r / 2) * 16 + (2 * h + r % 2)) * 128 + c * 1) *.
        at (kq r0 r1 r8 r9 r10 off) ((k * 8 + (2 * h + r % 2) / 2) * 128 + c * 1)) =
      if c < 128 then (qbr_at r0 r1 r8 r9 r10 off h r c; kbr_at r0 r1 r8 r9 r10 off h k c) else ()
    in
    Classical.forall_intro aux;
    sdot_ext 128 (qbr r0 r1 r8 r9 r10 off) ((h * 16 + r) * 128) 1 (kbr r0 r1 r8 r9 r10 off) ((h * 8 + k) * 128) 1
                 (qq r0 r1 r8 r9 r10 off) (((r / 2) * 16 + (2 * h + r % 2)) * 128) 1
                 (kq r0 r1 r8 r9 r10 off) ((k * 8 + (2 * h + r % 2) / 2) * 128) 1

noextract
let asi (h r k : nat) : nat = ((r / 2) * 16 + (2 * h + r % 2)) * 7 + k

(* S1: one masked scaled score *)
noextract
let s1 (r0 r1 r8 r9 r10 : S.seq real) (off h r k : nat)
  : Lemma (requires h < 8 /\ r < 14 /\ k < 7)
          (ensures sc_r (cqr r0 r1 r8 r9 r10 off) (h * 16 + r) k == at (asc r0 r1 r8 r9 r10 off) (asi h r k))
  = let rr = h * 16 + r in
    let p = rr * 8 + k in
    assert (p == h * 128 + (r * 8 + k));
    dm h (r * 8 + k) 128;
    dm rr k 8;
    dm h r 16;
    assert (at (cqr r0 r1 r8 r9 r10 off) p ==
            sdot 128 (qbr r0 r1 r8 r9 r10 off) (rr * 128) 1 (kbr r0 r1 r8 r9 r10 off) ((h * 8 + k) * 128) 1);
    s1_dot r0 r1 r8 r9 r10 off h r k;
    let t = r / 2 in
    let hq = 2 * h + r % 2 in
    let i = asi h r k in
    assert (i == t * 112 + (hq * 7 + k));
    dm t (hq * 7 + k) 112;
    dm (t * 16 + hq) k 7;
    dm t hq 16;
    reveal_opaque (`%attn_scores) (attn_scores (qq r0 r1 r8 r9 r10 off) (kq r0 r1 r8 r9 r10 off))

noextract
let s2 (r0 r1 r8 r9 r10 : S.seq real) (off h r : nat)
  : Lemma (requires h < 8 /\ r < 14)
          (ensures sumexp 7 (scs (cqr r0 r1 r8 r9 r10 off) (h * 16 + r)) 0 ==
                   sumexp 7 (asc r0 r1 r8 r9 r10 off) (asi h r 0))
  = let aux (k : nat) : Lemma (k < 7 ==> at (scs (cqr r0 r1 r8 r9 r10 off) (h * 16 + r)) (0 + k) ==
                                         at (asc r0 r1 r8 r9 r10 off) (asi h r 0 + k)) =
      if k < 7 then s1 r0 r1 r8 r9 r10 off h r k else ()
    in
    Classical.forall_intro aux;
    sumexp_ext 7 (scs (cqr r0 r1 r8 r9 r10 off) (h * 16 + r)) 0 (asc r0 r1 r8 r9 r10 off) (asi h r 0)

noextract
let smv (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : S.seq real = softmax7 (asc r0 r1 r8 r9 r10 off)

(* S3: softmax probability *)
noextract
let s3 (r0 r1 r8 r9 r10 : S.seq real) (off h r k : nat)
  : Lemma (requires h < 8 /\ r < 14 /\ k < 7)
          (ensures sm_r (cqr r0 r1 r8 r9 r10 off) (h * 16 + r) k == at (smv r0 r1 r8 r9 r10 off) (asi h r k))
  = s1 r0 r1 r8 r9 r10 off h r k;
    s2 r0 r1 r8 r9 r10 off h r;
    dm (16 * 0 + (h * 16 + r)) 0 1;
    dm h r 16;
    let i = asi h r k in
    dm ((r / 2) * 16 + (2 * h + r % 2)) k 7;
    reveal_opaque (`%softmax7) (softmax7 (asc r0 r1 r8 r9 r10 off))

(* ---- V^T ideal ---- *)
noextract
let vtr (qk : S.seq real) : S.seq real =
  S.init 65536 (fun (i:nat{i < 65536}) -> if i % 64 < 7 then at qk (vt_src i) else 0.0R)

(* ---- attention values (mode 4 GEMM + pvs) ---- *)
noextract
let avr (pr vt : S.seq real) (pp : nat) : real =
  sdot 64 pr (((pp / 128) / 16 * 16 + (pp / 8) % 16) * 64) 1 vt (((pp / 128) * 8 + pp % 8) * 64) 1

noextract
let avl (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : S.seq real =
  attn_values (smv r0 r1 r8 r9 r10 off) (lv r0 r8)

noextract
let pvs_dec (i : nat)
  : Lemma (requires i < 14336)
          (ensures (let t = i / 2048 in let h = (i % 2048) / 256 in let p = (i % 256) / 128 in let c = i % 128 in
                    let pp = pvs_i i in
                    t < 7 /\ h < 8 /\ p < 2 /\ c < 128 /\
                    i == t * 2048 + (2 * h + p) * 128 + c /\
                    (pp / 128) / 16 == h /\ (pp / 8) % 16 == 2 * t + p /\
                    (pp / 128) * 8 + pp % 8 == h * 128 + c))
  = let t = i / 2048 in
    let h = (i % 2048) / 256 in
    let p = (i % 256) / 128 in
    let c = i % 128 in
    ed i 2048; ed (i % 2048) 256; ed (i % 256) 128; ed i 256; ed i 128;
    ML.modulo_modulo_lemma i 256 8;
    ML.modulo_modulo_lemma i 128 2;
    assert (i % 2048 == h * 256 + p * 128 + c);
    let pp = pvs_i i in
    let hi = h * 16 + c / 8 in
    let lo = (t * 2 + p) * 8 + c % 8 in
    ed c 8;
    assert (pp == hi * 128 + lo);
    dm hi lo 128;
    dm (hi * 16 + (t * 2 + p)) (c % 8) 8;
    dm hi (t * 2 + p) 16;
    dm h (c / 8) 16

noextract
let pr_el (r0 r1 r8 r9 r10 : S.seq real) (off h r k : nat)
  : Lemma (requires h < 8 /\ r < 14 /\ k < 7)
          (ensures at (sm_seq (cqr r0 r1 r8 r9 r10 off)) ((h * 16 + r) * 64 + k) == at (smv r0 r1 r8 r9 r10 off) (asi h r k))
  = dm (h * 16 + r) k 64;
    lin_bnd (h * 16 + r) 128 64;
    dm h r 16;
    s3 r0 r1 r8 r9 r10 off h r k

noextract
let vt_el (r0 r8 : S.seq real) (h c k : nat)
  : Lemma (requires S.length r0 == 4194304 /\ S.length r8 == 8192 /\ h < 8 /\ c < 128 /\ k < 7)
          (ensures at (vtr (qkr r0 r8)) ((h * 128 + c) * 64 + k) == at (lv r0 r8) (k * 1024 + h * 128 + c))
  = dm (h * 128 + c) k 64;
    lin_bnd (h * 128 + c) 1024 64;
    dm h c 128;
    assert (vt_src ((h * 128 + c) * 64 + k) == k * 4096 + h * 512 + 384 + c);
    e3 r0 r8 k h c

noextract
let v1 (r0 r1 r8 r9 r10 : S.seq real) (off : nat) (i : nat)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\ i < 14336)
          (ensures avr (sm_seq (cqr r0 r1 r8 r9 r10 off)) (vtr (qkr r0 r8)) (pvs_i i) == at (avl r0 r1 r8 r9 r10 off) i)
  = pvs_dec i;
    let t = i / 2048 in
    let h = (i % 2048) / 256 in
    let p = (i % 256) / 128 in
    let c = i % 128 in
    let hq = 2 * h + p in
    let r = 2 * t + p in
    let pr = sm_seq (cqr r0 r1 r8 r9 r10 off) in
    let vt = vtr (qkr r0 r8) in
    let a0 = (h * 16 + r) * 64 in
    let b0 = (h * 128 + c) * 64 in
    assert (avr pr vt (pvs_i i) == sdot 64 pr a0 1 vt b0 1);
    let pad (k : nat) : Lemma (7 <= k /\ k < 64 ==> at pr (a0 + k * 1) *. at vt (b0 + k * 1) == 0.0R) =
      if 7 <= k && k < 64 then (dm (h * 128 + c) k 64; lin_bnd (h * 128 + c) 1024 64) else ()
    in
    Classical.forall_intro pad;
    sdot_pad 7 64 pr a0 1 vt b0 1;
    let sv = smv r0 r1 r8 r9 r10 off in
    let el (k : nat) : Lemma (k < 7 ==>
        at pr (a0 + k * 1) *. at vt (b0 + k * 1) ==
        at sv ((t * 16 + hq) * 7 + k * 1) *. at (lv r0 r8) ((hq / 2) * 128 + c + k * 1024)) =
      if k < 7 then begin
        dm t p 2;
        dm h p 2;
        assert (asi h r k == (t * 16 + hq) * 7 + k);
        pr_el r0 r1 r8 r9 r10 off h r k;
        vt_el r0 r8 h c k
      end else ()
    in
    Classical.forall_intro el;
    sdot_ext 7 pr a0 1 vt b0 1 sv ((t * 16 + hq) * 7) 1 (lv r0 r8) ((hq / 2) * 128 + c) 1024;
    dm t (hq * 128 + c) 2048;
    dm hq c 128;
    reveal_opaque (`%attn_values) (attn_values sv (lv r0 r8))

noextract
let attr (r0 r1 r8 r9 r10 : S.seq real) (off : nat) : S.seq real =
  S.init 16384 (fun (i:nat{i < 16384}) -> if i < 14336 then at (avl r0 r1 r8 r9 r10 off) i else 0.0R)

(* ===== BR3 ===== *)

(* ---- generic linear access ---- *)
noextract
let lin_at (rows : nat) (outs inner : pos) (w x w' n : S.seq real) (wo : nat) (i : nat)
  : Lemma (requires i < rows * outs /\ S.length x == rows * inner /\
                    (forall (k:nat). k < outs * inner ==> at w k == at w' (wo + k)) /\
                    (forall (k:nat). k < rows * inner ==> at x k == at n k))
          (ensures at (linear rows outs inner w x) i ==
                   sdot inner w' (wo + (i % outs) * inner) 1 n ((i / outs) * inner) 1)
  = reveal_opaque (`%linear) (linear rows outs inner w x);
    let o = i % outs in
    let t = i / outs in
    ed i outs;
    if t >= rows then ML.lemma_mult_le_right outs rows t;
    lin_bnd o outs inner;
    lin_bnd t rows inner;
    let aux (k : nat) : Lemma (k < inner ==>
        at w (o * inner + k * 1) *. at x (t * inner + k * 1) ==
        at w' ((wo + o * inner) + k * 1) *. at n (t * inner + k * 1)) =
      if k < inner then begin
        assert (o * inner + k < outs * inner);
        assert (t * inner + k < rows * inner)
      end else ()
    in
    Classical.forall_intro aux;
    sdot_ext inner w (o * inner) 1 x (t * inner) 1 w' (wo + o * inner) 1 n (t * inner) 1

noextract
let win_all (w : S.seq real) (wo wl : nat)
  : Lemma (requires wo + wl <= S.length w)
          (ensures forall (k:nat). k < wl ==> at (window w wo wl) k == at w (wo + k))
  = ()

(* ---- generic rmsnorm access ---- *)
noextract
let nr (h w : S.seq real) : S.seq real =
  S.init 8192 (fun (i:nat{i < 8192}) -> at w (i % 1024) *. (at h i *. rms_inv 1024 h ((i / 1024) * 1024)))

noextract
let nr_a (h x w : S.seq real) (i : nat)
  : Lemma (requires i < 7168 /\ S.length x == 7168 /\ (forall (k:nat). k < 7168 ==> at h k == at x k))
          (ensures at (nr h w) i == at (rmsnorm_rows 7 1024 x w) i)
  = reveal_opaque (`%rmsnorm_rows) (rmsnorm_rows 7 1024 x w);
    let b = (i / 1024) * 1024 in
    ed i 1024;
    lin_bnd (i / 1024) 7 1024;
    rms_ext 1024 h b x b

noextract
let nr_z (h p w : S.seq real) (c : nat)
  : Lemma (requires c < 1024 /\ S.length p == 1024 /\ (forall (k:nat). k < 1024 ==> at h (7168 + k) == at p k))
          (ensures at (nr h w) (7168 + c) == at (rmsnorm_rows 1 1024 p w) c)
  = reveal_opaque (`%rmsnorm_rows) (rmsnorm_rows 1 1024 p w);
    dm 7 c 1024;
    ML.small_div c 1024;
    ML.small_mod c 1024;
    rms_ext 1024 h 7168 p 0

(* ---- ideals ---- *)
noeq type rv = {
  v0 : S.seq real; v1 : S.seq real; v2 : S.seq real; v3 : S.seq real; v4 : S.seq real;
  v5 : S.seq real; v6 : S.seq real; v7 : S.seq real; v8 : S.seq real; v9 : S.seq real;
  v10 : S.seq real; voff : nat
}

noextract
let vlens (v : rv) : prop =
  lens v.v0 v.v1 v.v8 v.v9 v.v10 v.voff /\ S.length v.v2 == 2097152 /\ S.length v.v3 == 1024 /\
  S.length v.v4 == 6291456 /\ S.length v.v5 == 3145728 /\ S.length v.v6 == 1024 /\ S.length v.v7 == 8192

noextract
let pjr (w a : S.seq real) : S.seq real =
  S.init 8192 (fun (j:nat{j < 8192}) -> sdot 2048 w ((j % 1024) * 2048) 1 a ((j / 1024) * 2048) 1)

noextract
let pj2r (w a : S.seq real) : S.seq real =
  S.init 8192 (fun (j:nat{j < 8192}) -> sdot 3072 w ((j % 1024) * 3072) 1 a ((j / 1024) * 3072) 1)

noextract
let hr (p r : S.seq real) : S.seq real =
  S.init 8192 (fun (i:nat{i < 8192}) -> at p i +. at r i)

noextract
let wr2 (q : nat) : nat = (q / 128) * 64 + q % 64 + ((q % 128) / 64) * 3072

noextract
let gur (w n : S.seq real) : S.seq real =
  S.init 49152 (fun (j:nat{j < 49152}) -> sdot 1024 w (wr2 (j % 6144) * 1024) 1 n ((j / 6144) * 1024) 1)

noextract
let actr (g : S.seq real) : S.seq real =
  S.init 24576 (fun (q:nat{q < 24576}) ->
    if q / 3072 < 7 then silu (at g (silu_off q)) *. at g (silu_off q + 64) else 0.0R)

noextract
let atv (v : rv) : S.seq real = attr v.v0 v.v1 v.v8 v.v9 v.v10 v.voff
noextract
let h1v (v : rv) : S.seq real = hr (pjr v.v2 (atv v)) v.v7
noextract
let n1v (v : rv) : S.seq real = nr (h1v v) v.v3
noextract
let guv (v : rv) : S.seq real = gur v.v4 (n1v v)
noextract
let acv (v : rv) : S.seq real = actr (guv v)
noextract
let h2v (v : rv) : S.seq real = hr (pj2r v.v5 (acv v)) (h1v v)
noextract
let o1v (v : rv) : S.seq real = nr (h2v v) v.v6

(* spec side *)
noextract
let lwv (v : rv) : layer_weights = split_layer v.v0 v.v1 v.v2 v.v3 v.v4 v.v5
noextract
let attv (v : rv) : S.seq real =
  attention (lwv v) (cw v.v9 v.voff) (cw v.v10 v.voff) (window v.v8 0 7168)
noextract
let hH (v : rv) : S.seq real = vadd (window v.v7 0 7168) (attv v)
noextract
let xX (v : rv) : S.seq real = rmsnorm_rows 7 1024 (hH v) v.v3
noextract
let gG (v : rv) : S.seq real = linear 7 3072 1024 (window v.v4 0 3145728) (xX v)
noextract
let uU (v : rv) : S.seq real = linear 7 3072 1024 (window v.v4 3145728 3145728) (xX v)
noextract
let resv (v : rv) : S.seq real = vadd (hH v) (mlp (lwv v) (xX v))

noextract
let res_eq (v : rv)
  : Lemma (layer_residual (lwv v) (cw v.v9 v.voff) (cw v.v10 v.voff) (window v.v7 0 7168) (window v.v8 0 7168) == resv v)
  = ()

(* ---- attention block ---- *)
noextract
let att_eq (v : rv)
  : Lemma (attv v == linear 7 1024 2048 v.v2 (avl v.v0 v.v1 v.v8 v.v9 v.v10 v.voff))
  = assert (queries (lwv v) (cw v.v9 v.voff) (cw v.v10 v.voff) (window v.v8 0 7168) == qq v.v0 v.v1 v.v8 v.v9 v.v10 v.voff);
    assert (keys (lwv v) (cw v.v9 v.voff) (cw v.v10 v.voff) (window v.v8 0 7168) == kq v.v0 v.v1 v.v8 v.v9 v.v10 v.voff)

noextract
let attr_all (v : rv)
  : Lemma (forall (k:nat). k < 7 * 2048 ==> at (avl v.v0 v.v1 v.v8 v.v9 v.v10 v.voff) k == at (atv v) k)
  = ()

noextract
let pj_a (v : rv) (j : nat)
  : Lemma (requires j < 7168)
          (ensures at (pjr v.v2 (atv v)) j == at (attv v) j)
  = att_eq v;
    attr_all v;
    lin_at 7 1024 2048 v.v2 (avl v.v0 v.v1 v.v8 v.v9 v.v10 v.voff) v.v2 (atv v) 0 j

noextract
let pj_z (v : rv) (j : nat)
  : Lemma (requires 7168 <= j /\ j < 8192) (ensures at (pjr v.v2 (atv v)) j == 0.0R)
  = dm 7 (j - 7168) 1024;
    let aux (k : nat) : Lemma (k < 2048 ==> at v.v2 ((j % 1024) * 2048 + k * 1) *. at (atv v) (7 * 2048 + k * 1) == 0.0R) =
      ()
    in
    Classical.forall_intro aux;
    sdot_zero 2048 v.v2 ((j % 1024) * 2048) 1 (atv v) (7 * 2048) 1

noextract
let h1_a (v : rv) (i : nat)
  : Lemma (requires vlens v /\ i < 7168) (ensures at (h1v v) i == at (hH v) i)
  = pj_a v i

noextract
let h1_z (v : rv) (i : nat)
  : Lemma (requires vlens v /\ 7168 <= i /\ i < 8192) (ensures at (h1v v) i == at v.v7 i)
  = pj_z v i

noextract
let h1_all (v : rv)
  : Lemma (requires vlens v)
          (ensures S.length (hH v) == 7168 /\ (forall (k:nat). k < 7168 ==> at (h1v v) k == at (hH v) k))
  = Classical.forall_intro (Classical.move_requires (h1_a v))

noextract
let n1_all (v : rv)
  : Lemma (requires vlens v)
          (ensures S.length (xX v) == 7168 /\ (forall (k:nat). k < 7168 ==> at (xX v) k == at (n1v v) k))
  = h1_all v;
    Classical.forall_intro (Classical.move_requires (nr_a (h1v v) (hH v) v.v3))

(* ---- gate / up ---- *)
noextract
let gidx (t o : nat)
  : Lemma (requires t < 7 /\ o < 3072)
          (ensures (let q = (o / 64) * 128 + o % 64 in
                    q + 64 < 6144 /\ t * 3072 + o < 21504 /\
                    (t * 6144 + q) % 6144 == q /\ (t * 6144 + q) / 6144 == t /\
                    (t * 6144 + q + 64) % 6144 == q + 64 /\ (t * 6144 + q + 64) / 6144 == t /\
                    wr2 q == o /\ wr2 (q + 64) == o + 3072 /\
                    (t * 3072 + o) % 3072 == o /\ (t * 3072 + o) / 3072 == t))
  = let q = (o / 64) * 128 + o % 64 in
    ed o 64;
    lin_bnd (o / 64) 48 128;
    dm t q 6144;
    dm t (q + 64) 6144;
    dm (o / 64) (o % 64) 128;
    dm ((o / 64) * 2) (o % 64) 64;
    dm (o / 64) (64 + o % 64) 128;
    dm ((o / 64) * 2 + 1) (o % 64) 64;
    dm t o 3072

noextract
let lin_g (v : rv) (i : nat)
  : Lemma (requires vlens v /\ i < 21504)
          (ensures at (gG v) i == sdot 1024 v.v4 (0 + (i % 3072) * 1024) 1 (n1v v) ((i / 3072) * 1024) 1)
  = n1_all v;
    win_all v.v4 0 3145728;
    lin_at 7 3072 1024 (window v.v4 0 3145728) (xX v) v.v4 (n1v v) 0 i

noextract
let lin_u (v : rv) (i : nat)
  : Lemma (requires vlens v /\ i < 21504)
          (ensures at (uU v) i == sdot 1024 v.v4 (3145728 + (i % 3072) * 1024) 1 (n1v v) ((i / 3072) * 1024) 1)
  = n1_all v;
    win_all v.v4 3145728 3145728;
    lin_at 7 3072 1024 (window v.v4 3145728 3145728) (xX v) v.v4 (n1v v) 3145728 i

noextract
let gu_g (v : rv) (t o : nat)
  : Lemma (requires vlens v /\ t < 7 /\ o < 3072)
          (ensures at (guv v) (t * 6144 + ((o / 64) * 128 + o % 64)) == at (gG v) (t * 3072 + o))
  = gidx t o;
    let j = t * 6144 + ((o / 64) * 128 + o % 64) in
    assert (at (guv v) j == sdot 1024 v.v4 (o * 1024) 1 (n1v v) (t * 1024) 1);
    lin_g v (t * 3072 + o)

noextract
let gu_u (v : rv) (t o : nat)
  : Lemma (requires vlens v /\ t < 7 /\ o < 3072)
          (ensures at (guv v) (t * 6144 + ((o / 64) * 128 + o % 64) + 64) == at (uU v) (t * 3072 + o))
  = gidx t o;
    let j = t * 6144 + ((o / 64) * 128 + o % 64) + 64 in
    assert (at (guv v) j == sdot 1024 v.v4 (3145728 + o * 1024) 1 (n1v v) (t * 1024) 1);
    lin_u v (t * 3072 + o)

noextract
let act_a (v : rv) (q : nat)
  : Lemma (requires vlens v /\ q < 21504)
          (ensures at (silu_product (gG v) (uU v)) q == at (acv v) q)
  = reveal_opaque (`%silu_product) (silu_product (gG v) (uU v));
    let t = q / 3072 in
    let o = q % 3072 in
    ed q 3072;
    assert (silu_off q == t * 6144 + ((o / 64) * 128 + o % 64));
    gu_g v t o;
    gu_u v t o

noextract
let act_z (v : rv) (q : nat)
  : Lemma (requires 21504 <= q /\ q < 24576) (ensures at (acv v) q == 0.0R)
  = dm 7 (q - 21504) 3072

noextract
let act_all (v : rv)
  : Lemma (requires vlens v)
          (ensures S.length (silu_product (gG v) (uU v)) == 7 * 3072 /\
                   (forall (k:nat). k < 7 * 3072 ==> at (silu_product (gG v) (uU v)) k == at (acv v) k))
  = Classical.forall_intro (Classical.move_requires (act_a v))

noextract
let pj2_a (v : rv) (j : nat)
  : Lemma (requires vlens v /\ j < 7168)
          (ensures at (pj2r v.v5 (acv v)) j == at (mlp (lwv v) (xX v)) j)
  = act_all v;
    assert (mlp (lwv v) (xX v) == linear 7 1024 3072 v.v5 (silu_product (gG v) (uU v)));
    lin_at 7 1024 3072 v.v5 (silu_product (gG v) (uU v)) v.v5 (acv v) 0 j

noextract
let pj2_z (v : rv) (j : nat)
  : Lemma (requires 7168 <= j /\ j < 8192) (ensures at (pj2r v.v5 (acv v)) j == 0.0R)
  = dm 7 (j - 7168) 1024;
    let aux (k : nat) : Lemma (k < 3072 ==> at v.v5 ((j % 1024) * 3072 + k * 1) *. at (acv v) (7 * 3072 + k * 1) == 0.0R) =
      if k < 3072 then act_z v (7 * 3072 + k) else ()
    in
    Classical.forall_intro aux;
    sdot_zero 3072 v.v5 ((j % 1024) * 3072) 1 (acv v) (7 * 3072) 1

(* ---- second residual and final norm ---- *)
noextract
let h2_a (v : rv) (i : nat)
  : Lemma (requires vlens v /\ i < 7168) (ensures at (h2v v) i == at (resv v) i)
  = pj2_a v i;
    h1_a v i

noextract
let h2_z (v : rv) (c : nat)
  : Lemma (requires vlens v /\ c < 1024) (ensures at (h2v v) (7168 + c) == at (window v.v7 7168 1024) c)
  = pj2_z v (7168 + c);
    h1_z v (7168 + c)

noextract
let fin (v : rv)
  : Lemma (requires vlens v)
          (ensures S.length (resv v) == 7168 /\
                   (forall (i:nat). i < 7168 ==> at (h2v v) i == at (resv v) i) /\
                   (forall (c:nat). c < 1024 ==> at (h2v v) (7168 + c) == at (window v.v7 7168 1024) c) /\
                   (forall (i:nat). i < 7168 ==> at (o1v v) i == at (rmsnorm_rows 7 1024 (resv v) v.v6) i) /\
                   (forall (c:nat). c < 1024 ==> at (o1v v) (7168 + c) == at (rmsnorm_rows 1 1024 (window v.v7 7168 1024) v.v6) c))
  = Classical.forall_intro (Classical.move_requires (h2_a v));
    Classical.forall_intro (Classical.move_requires (h2_z v));
    Classical.forall_intro (Classical.move_requires (nr_a (h2v v) (resv v) v.v6));
    Classical.forall_intro (Classical.move_requires (nr_z (h2v v) (window v.v7 7168 1024) v.v6))

(* ===== GL ===== *)

(* ---- index arithmetic of the GEMM row selectors ---- *)
noextract
let sidx (w : pos) (j : natlt (w * 8))
  : Lemma (let p = scat_i w j in p / 128 == (j % w) / 16 /\ (p / 8) % 16 == (j % w) % 16 /\ p % 8 == j / w)
  = scat_bound w j;
    let p = scat_i w j in
    dm (j % w) (j / w) 8;
    ML.division_multiplication_lemma p 8 16

noextract
let wr1_eq (q : nat)
  : Lemma (requires q < 4096) (ensures ra_s 1 (q / 16) * 16 + q % 16 == wr1 q)
  = let a = q / 512 in
    let m = q % 512 in
    let b = m / 16 in
    let c = m % 16 in
    ed q 512; ed m 16;
    dm (a * 32 + b) c 16;
    dm a b 32;
    if b < 16 then ()
    else begin
      let b' = b - 16 in
      ed b' 8;
      dm (a * 4 + 2 + b' / 8) (b' % 8) 8;
      dm (b' / 8) ((b' % 8) * 16 + c) 128;
      dm (a * 4 + 2 + b' / 8) ((b' % 8) * 16 + c) 128
    end

noextract
let wr2_eq (q : nat)
  : Lemma (requires q < 6144) (ensures ra_s 2 (q / 16) * 16 + q % 16 == wr2 q)
  = let a = q / 128 in
    let m = q % 128 in
    let b = m / 64 in
    let m2 = m % 64 in
    let bb = m2 / 16 in
    let c = m2 % 16 in
    ed q 128; ed m 64; ed m2 16;
    dm (a * 8 + b * 4 + bb) c 16;
    dm a (b * 4 + bb) 8;
    dm b bb 4;
    dm (a * 2 + b) bb 4;
    dm (a * 2 + b) m2 64

(* ---- GEMM stages ---- *)
noextract
let cong1 (a b c d : nat) (e : nat)
  : Lemma (requires a == b /\ c == d /\ ra_s 1 b * 16 + d == e) (ensures ra_s 1 a * 16 + c == e) = ()

noextract
let cong2 (t c d : nat)
  : Lemma (requires c == d) (ensures cb_s 1 t * 8 + c == d) = ()

noextract
let qkv_el (#pf : squash (gemm_ok 4096 1024 8 256 1)) (s0 : chest1 bf16 4194304) (s8 : chest1 bf16 8192)
  (z : chest1 f32 32768) (r0 r8 : S.seq real) (j : natlt 32768)
  : Lemma (requires capx s0 r0 /\ capx s8 r8)
          (ensures scat_f 4096 (gemm_out 4096 1024 8 256 1 #pf s0 s8 z) j %~ at (qkr r0 r8) j)
  = let p = scat_i 4096 j in
    let c = gemm_out 4096 1024 8 256 1 #pf s0 s8 z in
    sidx 4096 j;
    wr1_eq (j % 4096);
    cong1 (p / 128) ((j % 4096) / 16) ((p / 8) % 16) ((j % 4096) % 16) (wr1 (j % 4096));
    let o1 = ra_s 1 (p / 128) * 16 + (p / 8) % 16 in
    cong2 (p / 128) (p % 8) (j / 4096);
    let o2 = cb_s 1 (p / 128) * 8 + p % 8 in
    assert (o1 == wr1 (j % 4096));
    assert (o2 == j / 4096);
    gemm_apx 4096 1024 8 256 1 #pf s0 s8 z r0 r8 p;
    assert (acc1 c p %~ sdot 1024 r0 (o1 * 1024) 1 r8 (o2 * 1024) 1);
    assert (at (qkr r0 r8) j == sdot 1024 r0 (wr1 (j % 4096) * 1024) 1 r8 ((j / 4096) * 1024) 1)

noextract
let qkv_l (#pf : squash (gemm_ok 4096 1024 8 256 1)) (s0 : chest1 bf16 4194304) (s8 : chest1 bf16 8192)
  (z : chest1 f32 32768) (q : chest1 bf16 32768) (r0 r8 : S.seq real)
  : Lemma (requires capx s0 r0 /\ capx s8 r8 /\
                    (forall (x:natlt 32768). acc1 q x == scat_f 4096 (gemm_out 4096 1024 8 256 1 #pf s0 s8 z) x))
          (ensures capx q (qkr r0 r8))
  = let aux (j : natlt 32768) : Lemma (acc1 q j %~ at (qkr r0 r8) j) =
      qkv_el #pf s0 s8 z r0 r8 j
    in
    Classical.forall_intro aux

noextract
let sc_l (#pf : squash (gemm_ok 128 128 64 8 3)) (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192)
  (z : chest1 f32 1024) (qb kb : S.seq real)
  : Lemma (requires capx sqb qb /\ capx skb kb)
          (ensures capx (gemm_out 128 128 64 8 3 #pf sqb skb z) (cqkr qb kb))
  = let aux (p : natlt 1024) : Lemma (acc1 (gemm_out 128 128 64 8 3 #pf sqb skb z) p %~ at (cqkr qb kb) p) =
      gemm_apx 128 128 64 8 3 #pf sqb skb z qb kb p
    in
    Classical.forall_intro aux

noextract
let pv_l (#pf : squash (gemm_ok 128 64 1024 128 4)) (pr : chest1 bf16 8192) (vt : chest1 bf16 65536)
  (z : chest1 f32 16384) (att : chest1 bf16 16384) (prr vtr' : S.seq real) (i : natlt 16384)
  : Lemma (requires capx pr prr /\ capx vt vtr' /\
                    (forall (x:natlt 16384). acc1 att x == pvs_f (gemm_out 128 64 1024 128 4 #pf pr vt z) x))
          (ensures acc1 att i %~ (if i < 14336 then avr prr vtr' (pvs_i i) else 0.0R))
  = if i < 14336 then gemm_apx 128 64 1024 128 4 #pf pr vt z prr vtr' (pvs_i i)
    else tobz_apx ()

noextract
let pj_l (#pf : squash (gemm_ok 1024 2048 8 64 0)) (s2 : chest1 bf16 2097152) (att : chest1 bf16 16384)
  (z : chest1 f32 8192) (pj : chest1 bf16 8192) (r2 a : S.seq real)
  : Lemma (requires capx s2 r2 /\ capx att a /\
                    (forall (x:natlt 8192). acc1 pj x == scat_f 1024 (gemm_out 1024 2048 8 64 0 #pf s2 att z) x))
          (ensures capx pj (pjr r2 a))
  = let aux (j : natlt 8192) : Lemma (acc1 pj j %~ at (pjr r2 a) j) =
      let p = scat_i 1024 j in
      sidx 1024 j;
      ed (j % 1024) 16;
      gemm_apx 1024 2048 8 64 0 #pf s2 att z r2 a p
    in
    Classical.forall_intro aux

noextract
let cong3 (a b c d : nat) (e : nat)
  : Lemma (requires a == b /\ c == d /\ ra_s 2 b * 16 + d == e) (ensures ra_s 2 a * 16 + c == e) = ()

noextract
let cong4 (t c d : nat)
  : Lemma (requires c == d) (ensures cb_s 2 t * 8 + c == d) = ()

#push-options "--z3rlimit 80 --fuel 0 --ifuel 0"
noextract
let gu_el (#pf : squash (gemm_ok 6144 1024 8 384 2)) (s4 : chest1 bf16 6291456) (n1 : chest1 bf16 8192)
  (z : chest1 f32 49152) (r4 n : S.seq real) (j : natlt 49152)
  : Lemma (requires capx s4 r4 /\ capx n1 n)
          (ensures scat_f 6144 (gemm_out 6144 1024 8 384 2 #pf s4 n1 z) j %~ at (gur r4 n) j)
  = let p = scat_i 6144 j in
    let c = gemm_out 6144 1024 8 384 2 #pf s4 n1 z in
    sidx 6144 j;
    wr2_eq (j % 6144);
    cong3 (p / 128) ((j % 6144) / 16) ((p / 8) % 16) ((j % 6144) % 16) (wr2 (j % 6144));
    cong4 (p / 128) (p % 8) (j / 6144);
    let o1 = ra_s 2 (p / 128) * 16 + (p / 8) % 16 in
    let o2 = cb_s 2 (p / 128) * 8 + p % 8 in
    assert (o1 == wr2 (j % 6144));
    assert (o2 == j / 6144);
    gemm_apx 6144 1024 8 384 2 #pf s4 n1 z r4 n p;
    assert (acc1 c p %~ sdot 1024 r4 (o1 * 1024) 1 n (o2 * 1024) 1);
    assert (at (gur r4 n) j == sdot 1024 r4 (wr2 (j % 6144) * 1024) 1 n ((j / 6144) * 1024) 1)
#pop-options

noextract
let gu_l (#pf : squash (gemm_ok 6144 1024 8 384 2)) (s4 : chest1 bf16 6291456) (n1 : chest1 bf16 8192)
  (z : chest1 f32 49152) (gu : chest1 bf16 49152) (r4 n : S.seq real)
  : Lemma (requires capx s4 r4 /\ capx n1 n /\
                    (forall (x:natlt 49152). acc1 gu x == scat_f 6144 (gemm_out 6144 1024 8 384 2 #pf s4 n1 z) x))
          (ensures capx gu (gur r4 n))
  = let aux (j : natlt 49152) : Lemma (acc1 gu j %~ at (gur r4 n) j) =
      gu_el #pf s4 n1 z r4 n j
    in
    Classical.forall_intro aux

noextract
let pj2_l (#pf : squash (gemm_ok 1024 3072 8 64 0)) (s5 : chest1 bf16 3145728) (act : chest1 bf16 24576)
  (z : chest1 f32 8192) (pj : chest1 bf16 8192) (r5 a : S.seq real)
  : Lemma (requires capx s5 r5 /\ capx act a /\
                    (forall (x:natlt 8192). acc1 pj x == scat_f 1024 (gemm_out 1024 3072 8 64 0 #pf s5 act z) x))
          (ensures capx pj (pj2r r5 a))
  = let aux (j : natlt 8192) : Lemma (acc1 pj j %~ at (pj2r r5 a) j) =
      let p = scat_i 1024 j in
      sidx 1024 j;
      ed (j % 1024) 16;
      gemm_apx 1024 3072 8 64 0 #pf s5 act z r5 a p
    in
    Classical.forall_intro aux

(* ---- exact kernels ---- *)
noextract
let vt_l (q : chest1 bf16 32768) (vt : chest1 bf16 65536) (qk : S.seq real)
  : Lemma (requires capx q qk /\ (forall (x:natlt 65536). acc1 vt x == vt_f q x))
          (ensures capx vt (vtr qk))
  = tobz_apx ()

noextract
let act_el (gu : chest1 bf16 49152) (g : S.seq real) (q : natlt 24576)
  : Lemma (requires capx gu g) (ensures silu_f gu q %~ at (actr g) q)
  = silu_bound q;
    if q / 3072 < 7 then begin
      let gv = tof (acc1 gu (silu_off q)) in
      let uv = tof (acc1 gu (silu_off q + 64)) in
      let gr = at g (silu_off q) in
      let ur = at g (silu_off q + 64) in
      assert (gv %~ gr);
      assert (uv %~ ur);
      silu_apx gv gr;
      let a = tob (div gv (add one (fexp (neg gv)))) in
      assert (tof a %~ silu gr);
      assert (mul (tof a) uv %~ (silu gr *. ur));
      assert (silu_f gu q == tob (mul (tof a) uv));
      assert (at (actr g) q == silu gr *. ur)
    end else tobz_apx ()

noextract
let act_l (gu : chest1 bf16 49152) (act : chest1 bf16 24576) (g : S.seq real)
  : Lemma (requires capx gu g /\ (forall (x:natlt 24576). acc1 act x == silu_f gu x))
          (ensures capx act (actr g))
  = Classical.forall_intro (Classical.move_requires (act_el gu g))

noextract
let nres_el (sp sr : chest1 bf16 8192) (p r : S.seq real) (i : nat)
  : Lemma (requires capx sp p /\ capx sr r) (ensures nres_f sp sr i %~ at (hr p r) i)
  = if i < 8192 then begin
      let a = tof (acc1 sp i) in
      let b = tof (acc1 sr i) in
      assert (a %~ at p i);
      assert (b %~ at r i);
      assert (add a b %~ (at p i +. at r i));
      assert (at (hr p r) i == at p i +. at r i)
    end else tobz_apx ()

noextract
let an_l (sw : chest1 bf16 1024) (sp sr sn sm : chest1 bf16 8192) (wr p r : S.seq real)
  : Lemma (requires capx sp p /\ capx sr r /\ capx sw wr /\
                    (forall (i:natlt 8192). acc1 sn i == nres_f sp sr i) /\
                    (forall (i:natlt 8192). el_ok sp sr sw i (acc1 sm i)))
          (ensures capx sn (hr p r) /\ capx sm (nr (hr p r) wr))
  = Classical.forall_intro (Classical.move_requires (nres_el sp sr p r));
    let aux (i : natlt 8192) : Lemma (acc1 sm i %~ at (nr (hr p r) wr) i) =
      el_e sp sr sw i (acc1 sm i) (hr p r) wr
    in
    Classical.forall_intro aux

(* ---- q/k after the norm + rope kernel ---- *)
noextract
let qb_l (q : chest1 bf16 32768) (s1 : chest1 bf16 256) (s9 s10 : chest1 bf16 524288) (off : nat)
  (sqb : chest1 bf16 16384) (r0 r1 r8 r9 r10 : S.seq real)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\ capx q (qkr r0 r8) /\ capx s1 r1 /\ capx s9 r9 /\ capx s10 r10 /\
                    (forall (j:natlt 16384). qb_ok q s1 s9 s10 off j (acc1 sqb j)))
          (ensures capx sqb (qbr r0 r1 r8 r9 r10 off))
  = let aux (j : natlt 16384) : Lemma (acc1 sqb j %~ at (qbr r0 r1 r8 r9 r10 off) j) =
      rot_e q s1 s9 s10 (qvb j) 0 (qtb off j) (j % 128) (acc1 sqb j) (qkr r0 r8) r1 r9 r10;
      qb_eq r0 r1 r8 r9 r10 off j
    in
    Classical.forall_intro aux

noextract
let kb_l (q : chest1 bf16 32768) (s1 : chest1 bf16 256) (s9 s10 : chest1 bf16 524288) (off : nat)
  (skb : chest1 bf16 8192) (r0 r1 r8 r9 r10 : S.seq real)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\ capx q (qkr r0 r8) /\ capx s1 r1 /\ capx s9 r9 /\ capx s10 r10 /\
                    (forall (j:natlt 8192). kb_ok q s1 s9 s10 off j (acc1 skb j)))
          (ensures capx skb (kbr r0 r1 r8 r9 r10 off))
  = let aux (j : natlt 8192) : Lemma (acc1 skb j %~ at (kbr r0 r1 r8 r9 r10 off) j) =
      rot_e q s1 s9 s10 (kvb j) 128 (ktb off j) (j % 128) (acc1 skb j) (qkr r0 r8) r1 r9 r10;
      kb_eq r0 r1 r8 r9 r10 off j
    in
    Classical.forall_intro aux

noextract
let att_l (#pf : squash (gemm_ok 128 64 1024 128 4)) (pr : chest1 bf16 8192) (vt : chest1 bf16 65536)
  (z : chest1 f32 16384) (att : chest1 bf16 16384) (r0 r1 r8 r9 r10 : S.seq real) (off : nat)
  : Lemma (requires lens r0 r1 r8 r9 r10 off /\
                    capx pr (sm_seq (cqr r0 r1 r8 r9 r10 off)) /\ capx vt (vtr (qkr r0 r8)) /\
                    (forall (x:natlt 16384). acc1 att x == pvs_f (gemm_out 128 64 1024 128 4 #pf pr vt z) x))
          (ensures capx att (attr r0 r1 r8 r9 r10 off))
  = let aux (i : natlt 16384) : Lemma (acc1 att i %~ at (attr r0 r1 r8 r9 r10 off) i) =
      pv_l #pf pr vt z att (sm_seq (cqr r0 r1 r8 r9 r10 off)) (vtr (qkr r0 r8)) i;
      if i < 14336 then v1 r0 r1 r8 r9 r10 off i else ()
    in
    Classical.forall_intro aux

(* ===== H0 ===== *)

noextract
let ok1 : squash (gemm_ok 4096 1024 8 256 1) = ()
noextract
let ok2 : squash (gemm_ok 128 128 64 8 3) = ()
noextract
let ok3 : squash (gemm_ok 128 64 1024 128 4) = ()
noextract
let ok4 : squash (gemm_ok 1024 2048 8 64 0) = ()
noextract
let ok5 : squash (gemm_ok 6144 1024 8 384 2) = ()
noextract
let ok6 : squash (gemm_ok 1024 3072 8 64 0) = ()
noextract
let wlen (r : S.seq real) : Lemma (requires S.length r == 8192) (ensures S.length (window r 7168 1024) == 1024) = ()

(* boundary facts of the kernel calls, all exact (no reals) *)
noextract
let dec_pre
  (s0 : chest1 bf16 4194304) (s1 : chest1 bf16 256) (s2 : chest1 bf16 2097152)
  (s3 : chest1 bf16 1024) (s4 : chest1 bf16 6291456) (s5 : chest1 bf16 3145728)
  (s6 : chest1 bf16 1024) (s7 : chest1 bf16 8192) (s8 : chest1 bf16 8192)
  (s9 : chest1 bf16 524288) (s10 : chest1 bf16 524288) (off : nat)
  (z1 : chest1 f32 32768) (q : chest1 bf16 32768)
  (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192) (vt : chest1 bf16 65536)
  (z2 : chest1 f32 1024) (pr : chest1 bf16 8192)
  (z3 : chest1 f32 16384) (att : chest1 bf16 16384)
  (z4 : chest1 f32 8192) (pj1 res1 n1 : chest1 bf16 8192)
  (z5 : chest1 f32 49152) (gu : chest1 bf16 49152) (act : chest1 bf16 24576)
  (z6 : chest1 f32 8192) (pj2 o0 o1 : chest1 bf16 8192) : prop =
  (forall (x:natlt 32768). acc1 q x == scat_f 4096 (gemm_out 4096 1024 8 256 1 #ok1 s0 s8 z1) x) /\
  (forall (j:natlt 16384). qb_ok q s1 s9 s10 off j (acc1 sqb j)) /\
  (forall (j:natlt 8192). kb_ok q s1 s9 s10 off j (acc1 skb j)) /\
  (forall (x:natlt 65536). acc1 vt x == vt_f q x) /\
  sm_post (gemm_out 128 128 64 8 3 #ok2 sqb skb z2) pr /\
  (forall (x:natlt 16384). acc1 att x == pvs_f (gemm_out 128 64 1024 128 4 #ok3 pr vt z3) x) /\
  (forall (x:natlt 8192). acc1 pj1 x == scat_f 1024 (gemm_out 1024 2048 8 64 0 #ok4 s2 att z4) x) /\
  (forall (i:natlt 8192). acc1 res1 i == nres_f pj1 s7 i) /\
  (forall (i:natlt 8192). el_ok pj1 s7 s3 i (acc1 n1 i)) /\
  (forall (x:natlt 49152). acc1 gu x == scat_f 6144 (gemm_out 6144 1024 8 384 2 #ok5 s4 n1 z5) x) /\
  (forall (x:natlt 24576). acc1 act x == silu_f gu x) /\
  (forall (x:natlt 8192). acc1 pj2 x == scat_f 1024 (gemm_out 1024 3072 8 64 0 #ok6 s5 act z6) x) /\
  (forall (i:natlt 8192). acc1 o0 i == nres_f pj2 res1 i) /\
  (forall (i:natlt 8192). el_ok pj2 res1 s6 i (acc1 o1 i))

noextract
let wa (o : chest1 bf16 8192) (h r : S.seq real) (b m : nat)
  : Lemma (requires b + m <= 8192 /\ S.length r == m /\ capx o h /\
                    (forall (i:nat). i < m ==> at h (b + i) == at r i))
          (ensures seq_approximates #bf16 (window (chest1_to_seq o) b m) r)
  = let aux (i : nat) : Lemma (i < m ==> acc1 o (b + i) %~ at r i) =
      if i < m then assert (acc1 o (b + i) %~ at h (b + i)) else ()
    in
    Classical.forall_intro aux;
    win_apx o b m r

noextract
let dec_real
  (s0 : chest1 bf16 4194304) (s1 : chest1 bf16 256) (s2 : chest1 bf16 2097152)
  (s3 : chest1 bf16 1024) (s4 : chest1 bf16 6291456) (s5 : chest1 bf16 3145728)
  (s6 : chest1 bf16 1024) (s7 : chest1 bf16 8192) (s8 : chest1 bf16 8192)
  (s9 : chest1 bf16 524288) (s10 : chest1 bf16 524288) (off : nat)
  (z1 : chest1 f32 32768) (q : chest1 bf16 32768)
  (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192) (vt : chest1 bf16 65536)
  (z2 : chest1 f32 1024) (pr : chest1 bf16 8192)
  (z3 : chest1 f32 16384) (att : chest1 bf16 16384)
  (z4 : chest1 f32 8192) (pj1 res1 n1 : chest1 bf16 8192)
  (z5 : chest1 f32 49152) (gu : chest1 bf16 49152) (act : chest1 bf16 24576)
  (z6 : chest1 f32 8192) (pj2 o0 o1 : chest1 bf16 8192)
  (r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 : S.seq real)
  : Lemma (requires off <= 4088 /\
                    dec_pre s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 off z1 q sqb skb vt z2 pr z3 att z4 pj1 res1 n1
                            z5 gu act z6 pj2 o0 o1 /\
                    seq_approximates #bf16 (chest1_to_seq s0) r0 /\ seq_approximates #bf16 (chest1_to_seq s1) r1 /\
                    seq_approximates #bf16 (chest1_to_seq s2) r2 /\ seq_approximates #bf16 (chest1_to_seq s3) r3 /\
                    seq_approximates #bf16 (chest1_to_seq s4) r4 /\ seq_approximates #bf16 (chest1_to_seq s5) r5 /\
                    seq_approximates #bf16 (chest1_to_seq s6) r6 /\ seq_approximates #bf16 (chest1_to_seq s7) r7 /\
                    seq_approximates #bf16 (chest1_to_seq s8) r8 /\ seq_approximates #bf16 (chest1_to_seq s9) r9 /\
                    seq_approximates #bf16 (chest1_to_seq s10) r10)
          (ensures (let res = layer_residual (split_layer r0 r1 r2 r3 r4 r5)
                                (window r9 (off * 128) 896) (window r10 (off * 128) 896)
                                (window r7 0 7168) (window r8 0 7168) in
                    let pad = window r7 7168 1024 in
                    seq_approximates #bf16 (window (chest1_to_seq o0) 0 7168) res /\
                    seq_approximates #bf16 (window (chest1_to_seq o1) 0 7168) (rmsnorm_rows 7 1024 res r6) /\
                    seq_approximates #bf16 (window (chest1_to_seq o0) 7168 1024) pad /\
                    seq_approximates #bf16 (window (chest1_to_seq o1) 7168 1024) (rmsnorm_rows 1 1024 pad r6)))
  = seq_capx s0 r0; seq_capx s1 r1; seq_capx s2 r2; seq_capx s3 r3; seq_capx s4 r4; seq_capx s5 r5;
    seq_capx s6 r6; seq_capx s7 r7; seq_capx s8 r8; seq_capx s9 r9; seq_capx s10 r10;
    let v : rv = { v0 = r0; v1 = r1; v2 = r2; v3 = r3; v4 = r4; v5 = r5; v6 = r6; v7 = r7;
                   v8 = r8; v9 = r9; v10 = r10; voff = off } in
    assert (vlens v);
    qkv_l #ok1 s0 s8 z1 q r0 r8;
    qb_l q s1 s9 s10 off sqb r0 r1 r8 r9 r10;
    kb_l q s1 s9 s10 off skb r0 r1 r8 r9 r10;
    vt_l q vt (qkr r0 r8);
    sc_l #ok2 sqb skb z2 (qbr r0 r1 r8 r9 r10 off) (kbr r0 r1 r8 r9 r10 off);
    assert (cqr r0 r1 r8 r9 r10 off == cqkr (qbr r0 r1 r8 r9 r10 off) (kbr r0 r1 r8 r9 r10 off));
    assert (capx pr (sm_seq (cqr r0 r1 r8 r9 r10 off)));
    att_l #ok3 pr vt z3 att r0 r1 r8 r9 r10 off;
    assert (atv v == attr r0 r1 r8 r9 r10 off);
    pj_l #ok4 s2 att z4 pj1 r2 (atv v);
    an_l s3 pj1 s7 res1 n1 r3 (pjr r2 (atv v)) r7;
    assert (capx res1 (h1v v));
    assert (capx n1 (n1v v));
    gu_l #ok5 s4 n1 z5 gu r4 (n1v v);
    act_l gu act (guv v);
    pj2_l #ok6 s5 act z6 pj2 r5 (acv v);
    an_l s6 pj2 res1 o0 o1 r6 (pj2r r5 (acv v)) (h1v v);
    assert (capx o0 (h2v v));
    assert (capx o1 (o1v v));
    fin v;
    wlen r7;
    res_eq v;
    wa o0 (h2v v) (resv v) 0 7168;
    wa o1 (o1v v) (rmsnorm_rows 7 1024 (resv v) r6) 0 7168;
    wa o0 (h2v v) (window r7 7168 1024) 7168 1024;
    wa o1 (o1v v) (rmsnorm_rows 1 1024 (window r7 7168 1024) r6) 7168 1024

noextract
let dec_all
  (s0 : chest1 bf16 4194304) (s1 : chest1 bf16 256) (s2 : chest1 bf16 2097152)
  (s3 : chest1 bf16 1024) (s4 : chest1 bf16 6291456) (s5 : chest1 bf16 3145728)
  (s6 : chest1 bf16 1024) (s7 : chest1 bf16 8192) (s8 : chest1 bf16 8192)
  (s9 : chest1 bf16 524288) (s10 : chest1 bf16 524288) (off : nat)
  (z1 : chest1 f32 32768) (q : chest1 bf16 32768)
  (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192) (vt : chest1 bf16 65536)
  (z2 : chest1 f32 1024) (pr : chest1 bf16 8192)
  (z3 : chest1 f32 16384) (att : chest1 bf16 16384)
  (z4 : chest1 f32 8192) (pj1 res1 n1 : chest1 bf16 8192)
  (z5 : chest1 f32 49152) (gu : chest1 bf16 49152) (act : chest1 bf16 24576)
  (z6 : chest1 f32 8192) (pj2 o0 o1 : chest1 bf16 8192)
  : Lemma (requires off <= 4088 /\
                    dec_pre s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 off z1 q sqb skb vt z2 pr z3 att z4 pj1 res1 n1
                            z5 gu act z6 pj2 o0 o1)
          (ensures (forall (r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 : S.seq real).
      ((chest1_to_seq s0 <: S.seq BF.t) %~ r0 /\ (chest1_to_seq s1 <: S.seq BF.t) %~ r1 /\ (chest1_to_seq s2 <: S.seq BF.t) %~ r2 /\
       (chest1_to_seq s3 <: S.seq BF.t) %~ r3 /\ (chest1_to_seq s4 <: S.seq BF.t) %~ r4 /\ (chest1_to_seq s5 <: S.seq BF.t) %~ r5 /\
       (chest1_to_seq s6 <: S.seq BF.t) %~ r6 /\ (chest1_to_seq s7 <: S.seq BF.t) %~ r7 /\ (chest1_to_seq s8 <: S.seq BF.t) %~ r8 /\
       (chest1_to_seq s9 <: S.seq BF.t) %~ r9 /\ (chest1_to_seq s10 <: S.seq BF.t) %~ r10) ==>
      (let res = layer_residual (split_layer r0 r1 r2 r3 r4 r5)
                   (window r9 (off * 128) 896) (window r10 (off * 128) 896)
                   (window r7 0 7168) (window r8 0 7168) in
       let pad = window r7 7168 1024 in
       window (chest1_to_seq o0) 0 7168 %~ res /\
       window (chest1_to_seq o1) 0 7168 %~ rmsnorm_rows 7 1024 res r6 /\
       window (chest1_to_seq o0) 7168 1024 %~ pad /\
       window (chest1_to_seq o1) 7168 1024 %~ rmsnorm_rows 1 1024 pad r6)))
  = introduce forall (r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 : S.seq real).
      ((chest1_to_seq s0 <: S.seq BF.t) %~ r0 /\ (chest1_to_seq s1 <: S.seq BF.t) %~ r1 /\ (chest1_to_seq s2 <: S.seq BF.t) %~ r2 /\
       (chest1_to_seq s3 <: S.seq BF.t) %~ r3 /\ (chest1_to_seq s4 <: S.seq BF.t) %~ r4 /\ (chest1_to_seq s5 <: S.seq BF.t) %~ r5 /\
       (chest1_to_seq s6 <: S.seq BF.t) %~ r6 /\ (chest1_to_seq s7 <: S.seq BF.t) %~ r7 /\ (chest1_to_seq s8 <: S.seq BF.t) %~ r8 /\
       (chest1_to_seq s9 <: S.seq BF.t) %~ r9 /\ (chest1_to_seq s10 <: S.seq BF.t) %~ r10) ==>
      (let res = layer_residual (split_layer r0 r1 r2 r3 r4 r5)
                   (window r9 (off * 128) 896) (window r10 (off * 128) 896)
                   (window r7 0 7168) (window r8 0 7168) in
       let pad = window r7 7168 1024 in
       window (chest1_to_seq o0) 0 7168 %~ res /\
       window (chest1_to_seq o1) 0 7168 %~ rmsnorm_rows 7 1024 res r6 /\
       window (chest1_to_seq o0) 7168 1024 %~ pad /\
       window (chest1_to_seq o1) 7168 1024 %~ rmsnorm_rows 1 1024 pad r6)
    with introduce _ ==> _ with
      dec_real s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 off z1 q sqb skb vt z2 pr z3 att z4 pj1 res1 n1
               z5 gu act z6 pj2 o0 o1 r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10

noextract
let dec_rel
  (s0 : chest1 bf16 4194304) (s1 : chest1 bf16 256) (s2 : chest1 bf16 2097152)
  (s3 : chest1 bf16 1024) (s4 : chest1 bf16 6291456) (s5 : chest1 bf16 3145728)
  (s6 : chest1 bf16 1024) (s7 : chest1 bf16 8192) (s8 : chest1 bf16 8192)
  (s9 : chest1 bf16 524288) (s10 : chest1 bf16 524288) (off : nat)
  (z1 : chest1 f32 32768) (q : chest1 bf16 32768)
  (sqb : chest1 bf16 16384) (skb : chest1 bf16 8192) (vt : chest1 bf16 65536)
  (z2 : chest1 f32 1024) (pr : chest1 bf16 8192)
  (z3 : chest1 f32 16384) (att : chest1 bf16 16384)
  (z4 : chest1 f32 8192) (pj1 res1 n1 : chest1 bf16 8192)
  (z5 : chest1 f32 49152) (gu : chest1 bf16 49152) (act : chest1 bf16 24576)
  (z6 : chest1 f32 8192) (pj2 o0 o1 : chest1 bf16 8192)
  : Lemma (requires off <= 4088 /\
                    dec_pre s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 off z1 q sqb skb vt z2 pr z3 att z4 pj1 res1 n1
                            z5 gu act z6 pj2 o0 o1)
          (ensures optimized_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
                     (chest1_to_seq s3) (chest1_to_seq s4) (chest1_to_seq s5) (chest1_to_seq s6)
                     (chest1_to_seq s7) (chest1_to_seq s8) (chest1_to_seq s9) (chest1_to_seq s10)
                     off (chest1_to_seq o0) (chest1_to_seq o1))
  = dec_all s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 off z1 q sqb skb vt z2 pr z3 att z4 pj1 res1 n1
            z5 gu act z6 pj2 o0 o1
    ;assert (optimized_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
                     (chest1_to_seq s3) (chest1_to_seq s4) (chest1_to_seq s5) (chest1_to_seq s6)
                     (chest1_to_seq s7) (chest1_to_seq s8) (chest1_to_seq s9) (chest1_to_seq s10)
                     off (chest1_to_seq o0) (chest1_to_seq o1)) by
      (FStar.Tactics.V2.norm [delta_only [`%optimized_decoder_relation]; zeta])

noextract
let rel_vac (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : S.seq BF.t) (a11 : int) (o0 o1 : S.seq BF.t)
  : Lemma (requires ~(0 <= a11 /\ a11 <= 4088))
          (ensures optimized_decoder_relation a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 o0 o1)
  = assert (optimized_decoder_relation a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 o0 o1) by
      (FStar.Tactics.V2.norm [delta_only [`%optimized_decoder_relation]; zeta])

(* ===== H ===== *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn decoder_body
  (input0 : array1 bf16 (l1_forward 4194304) { is_global input0 })
  (input1 : array1 bf16 (l1_forward 256) { is_global input1 })
  (input2 : array1 bf16 (l1_forward 2097152) { is_global input2 })
  (input3 : array1 bf16 (l1_forward 1024) { is_global input3 })
  (input4 : array1 bf16 (l1_forward 6291456) { is_global input4 })
  (input5 : array1 bf16 (l1_forward 3145728) { is_global input5 })
  (input6 : array1 bf16 (l1_forward 1024) { is_global input6 })
  (input7 : array1 bf16 (l1_forward 8192) { is_global input7 })
  (input8 : array1 bf16 (l1_forward 8192) { is_global input8 })
  (input9 : array1 bf16 (l1_forward 524288) { is_global input9 })
  (input10 : array1 bf16 (l1_forward 524288) { is_global input10 })
  (off : sz{SZ.v off <= 4088})
  (output0 : array1 bf16 (l1_forward 8192) { is_global output0 })
  (output1 : array1 bf16 (l1_forward 8192) { is_global output1 })
  (#f0 #f1 #f2 #f3 #f4 #f5 #f6 #f7 #f8 #f9 #f10 : perm)
  (#s0 : erased (chest1 bf16 4194304)) (#s1 : erased (chest1 bf16 256)) (#s2 : erased (chest1 bf16 2097152))
  (#s3 : erased (chest1 bf16 1024)) (#s4 : erased (chest1 bf16 6291456)) (#s5 : erased (chest1 bf16 3145728))
  (#s6 : erased (chest1 bf16 1024)) (#s7 : erased (chest1 bf16 8192)) (#s8 : erased (chest1 bf16 8192))
  (#s9 : erased (chest1 bf16 524288)) (#s10 : erased (chest1 bf16 524288))
  (#o0 #o1 : erased (chest1 bf16 8192))
  preserves cpu **
    on gpu_loc (input0 |-> Frac f0 s0) ** on gpu_loc (input1 |-> Frac f1 s1) **
    on gpu_loc (input2 |-> Frac f2 s2) ** on gpu_loc (input3 |-> Frac f3 s3) **
    on gpu_loc (input4 |-> Frac f4 s4) ** on gpu_loc (input5 |-> Frac f5 s5) **
    on gpu_loc (input6 |-> Frac f6 s6) ** on gpu_loc (input7 |-> Frac f7 s7) **
    on gpu_loc (input8 |-> Frac f8 s8) ** on gpu_loc (input9 |-> Frac f9 s9) **
    on gpu_loc (input10 |-> Frac f10 s10)
  requires on gpu_loc (output0 |-> o0) ** on gpu_loc (output1 |-> o1)
  ensures exists* o0' o1'. on gpu_loc (output0 |-> o0') ** on gpu_loc (output1 |-> o1') **
    pure (optimized_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
            (chest1_to_seq s3) (chest1_to_seq s4) (chest1_to_seq s5) (chest1_to_seq s6)
            (chest1_to_seq s7) (chest1_to_seq s8) (chest1_to_seq s9) (chest1_to_seq s10)
            (SZ.v off) (chest1_to_seq o0') (chest1_to_seq o1'))
{
  let qkv = alloc0 #bf16 32768sz (l1_forward 32768);
  let cqkv = alloc0 #f32 32768sz (l1_forward 32768);
  with z1. assert (on gpu_loc (cqkv |-> z1));
  gemm_launch 4096sz 1024sz 8sz 256sz 1sz #ok1 input0 input8 cqkv;
  with a2. assert (on gpu_loc (qkv |-> a2));
  launch_kernel_1 (fun () -> k_scatter 4096sz cqkv qkv #1.0R #(gemm_out 4096 1024 8 256 1 #ok1 s0 s8 z1) #a2);
  with q. assert (on gpu_loc (qkv |-> q));
  free cqkv;
  let part32 = alloc0 #f32 32sz (l1_forward 32);
  let un = alloc0 #bf16 128sz (l1_forward 128);
  let qb = alloc0 #bf16 16384sz (l1_forward 16384);
  let kb = alloc0 #bf16 8192sz (l1_forward 8192);
  with a2. assert (on gpu_loc (part32 |-> a2));
  with a3. assert (on gpu_loc (un |-> a3));
  with a4. assert (on gpu_loc (qb |-> a4));
  with a5. assert (on gpu_loc (kb |-> a5));
  launch_kernel_1 (fun () -> k_qknorm input1 qkv input9 input10 off part32 un qb kb #f1 #1.0R #f9 #f10 #s1 #q #s9 #s10 #a2 #a3 #a4 #a5);
  with sqb. assert (on gpu_loc (qb |-> sqb));
  with skb. assert (on gpu_loc (kb |-> skb));
  free part32;
  free un;
  let vt = alloc0 #bf16 n64k (l1_forward 65536);
  with a2. assert (on gpu_loc (vt |-> a2));
  launch_kernel_1 (fun () -> k_vt qkv vt #1.0R #q #a2);
  with svt. assert (on gpu_loc (vt |-> svt));
  free qkv;
  let cqk = alloc0 #f32 1024sz (l1_forward 1024);
  with z2. assert (on gpu_loc (cqk |-> z2));
  gemm_launch 128sz 128sz 64sz 8sz 3sz #ok2 qb kb cqk;
  free qb;
  free kb;
  let pr = alloc0 #bf16 8192sz (l1_forward 8192);
  with a2. assert (on gpu_loc (pr |-> a2));
  launch_kernel_1 (fun () -> k_softmax cqk pr #1.0R #(gemm_out 128 128 64 8 3 #ok2 sqb skb z2) #a2);
  with spr. assert (on gpu_loc (pr |-> spr));
  free cqk;
  let cpv = alloc0 #f32 16384sz (l1_forward 16384);
  with z3. assert (on gpu_loc (cpv |-> z3));
  gemm_launch 128sz 64sz 1024sz 128sz 4sz #ok3 pr vt cpv;
  free pr;
  free vt;
  let att = alloc0 #bf16 16384sz (l1_forward 16384);
  with a2. assert (on gpu_loc (att |-> a2));
  launch_kernel_1 (fun () -> k_pvs cpv att #1.0R #(gemm_out 128 64 1024 128 4 #ok3 spr svt z3) #a2);
  with satt. assert (on gpu_loc (att |-> satt));
  free cpv;
  let cpj = alloc0 #f32 8192sz (l1_forward 8192);
  with z4. assert (on gpu_loc (cpj |-> z4));
  gemm_launch 1024sz 2048sz 8sz 64sz 0sz #ok4 input2 att cpj;
  free att;
  let pj = alloc0 #bf16 8192sz (l1_forward 8192);
  with a2. assert (on gpu_loc (pj |-> a2));
  launch_kernel_1 (fun () -> k_scatter 1024sz cpj pj #1.0R #(gemm_out 1024 2048 8 64 0 #ok4 s2 satt z4) #a2);
  with pj1. assert (on gpu_loc (pj |-> pj1));
  let res1 = alloc0 #bf16 8192sz (l1_forward 8192);
  let part256 = alloc0 #f32 256sz (l1_forward 256);
  with a2. assert (on gpu_loc (res1 |-> a2));
  with a3. assert (on gpu_loc (output1 |-> a3));
  with a4. assert (on gpu_loc (part256 |-> a4));
  launch_kernel_1 (fun () -> k_addnorm input3 pj input7 res1 output1 part256 #f3 #1.0R #f7 #s3 #pj1 #s7 #a2 #a3 #a4);
  with sres1. assert (on gpu_loc (res1 |-> sres1));
  with n1. assert (on gpu_loc (output1 |-> n1));
  let cgu = alloc0 #f32 49152sz (l1_forward 49152);
  with z5. assert (on gpu_loc (cgu |-> z5));
  gemm_launch 6144sz 1024sz 8sz 384sz 2sz #ok5 input4 output1 cgu;
  let gu = alloc0 #bf16 49152sz (l1_forward 49152);
  with a2. assert (on gpu_loc (gu |-> a2));
  launch_kernel_1 (fun () -> k_scatter 6144sz cgu gu #1.0R #(gemm_out 6144 1024 8 384 2 #ok5 s4 n1 z5) #a2);
  with sgu. assert (on gpu_loc (gu |-> sgu));
  free cgu;
  let act = alloc0 #bf16 24576sz (l1_forward 24576);
  with a2. assert (on gpu_loc (act |-> a2));
  launch_kernel_1 (fun () -> k_silu gu act #1.0R #sgu #a2);
  with sact. assert (on gpu_loc (act |-> sact));
  free gu;
  with z6. assert (on gpu_loc (cpj |-> z6));
  gemm_launch 1024sz 3072sz 8sz 64sz 0sz #ok6 input5 act cpj;
  free act;
  launch_kernel_1 (fun () -> k_scatter 1024sz cpj pj #1.0R #(gemm_out 1024 3072 8 64 0 #ok6 s5 sact z6) #pj1);
  with pj2. assert (on gpu_loc (pj |-> pj2));
  free cpj;
  with a3. assert (on gpu_loc (output0 |-> a3));
  with a5. assert (on gpu_loc (part256 |-> a5));
  launch_kernel_1 (fun () -> k_addnorm input6 pj res1 output0 output1 part256 #f6 #1.0R #1.0R #s6 #pj2 #sres1 #a3 #n1 #a5);
  with so0. assert (on gpu_loc (output0 |-> so0));
  with so1. assert (on gpu_loc (output1 |-> so1));
  free pj;
  free res1;
  free part256;
  assert (pure (dec_pre s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 (SZ.v off) z1 q sqb skb svt z2 spr z3 satt z4 pj1 sres1 n1
          z5 sgu sact z6 pj2 so0 so1));
  dec_rel s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 (SZ.v off) z1 q sqb skb svt z2 spr z3 satt z4 pj1 sres1 n1
          z5 sgu sact z6 pj2 so0 so1;
  assert (pure (optimized_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
            (chest1_to_seq s3) (chest1_to_seq s4) (chest1_to_seq s5) (chest1_to_seq s6)
            (chest1_to_seq s7) (chest1_to_seq s8) (chest1_to_seq s9) (chest1_to_seq s10)
            (SZ.v off) (chest1_to_seq so0) (chest1_to_seq so1)));
}
#pop-options

fn kplay_run
  (input0 : array1 bf16 (l1_forward 4194304) { is_global input0 })
  (input1 : array1 bf16 (l1_forward 256) { is_global input1 })
  (input2 : array1 bf16 (l1_forward 2097152) { is_global input2 })
  (input3 : array1 bf16 (l1_forward 1024) { is_global input3 })
  (input4 : array1 bf16 (l1_forward 6291456) { is_global input4 })
  (input5 : array1 bf16 (l1_forward 3145728) { is_global input5 })
  (input6 : array1 bf16 (l1_forward 1024) { is_global input6 })
  (input7 : array1 bf16 (l1_forward 8192) { is_global input7 })
  (input8 : array1 bf16 (l1_forward 8192) { is_global input8 })
  (input9 : array1 bf16 (l1_forward 524288) { is_global input9 })
  (input10 : array1 bf16 (l1_forward 524288) { is_global input10 })
  (input11 : FStar.Int64.t)
  (output0 : array1 bf16 (l1_forward 8192) { is_global output0 })
  (output1 : array1 bf16 (l1_forward 8192) { is_global output1 })
  (#f0 #f1 #f2 #f3 #f4 #f5 #f6 #f7 #f8 #f9 #f10 : perm)
  (#s0 : chest1 bf16 4194304) (#s1 : chest1 bf16 256) (#s2 : chest1 bf16 2097152)
  (#s3 : chest1 bf16 1024) (#s4 : chest1 bf16 6291456) (#s5 : chest1 bf16 3145728)
  (#s6 : chest1 bf16 1024) (#s7 : chest1 bf16 8192) (#s8 : chest1 bf16 8192)
  (#s9 : chest1 bf16 524288) (#s10 : chest1 bf16 524288)
  (#o0 #o1 : chest1 bf16 8192)
  preserves cpu **
    on gpu_loc (input0 |-> Frac f0 s0) ** on gpu_loc (input1 |-> Frac f1 s1) **
    on gpu_loc (input2 |-> Frac f2 s2) ** on gpu_loc (input3 |-> Frac f3 s3) **
    on gpu_loc (input4 |-> Frac f4 s4) ** on gpu_loc (input5 |-> Frac f5 s5) **
    on gpu_loc (input6 |-> Frac f6 s6) ** on gpu_loc (input7 |-> Frac f7 s7) **
    on gpu_loc (input8 |-> Frac f8 s8) ** on gpu_loc (input9 |-> Frac f9 s9) **
    on gpu_loc (input10 |-> Frac f10 s10)
  requires on gpu_loc (output0 |-> o0) ** on gpu_loc (output1 |-> o1)
  ensures exists* o0' o1'. on gpu_loc (output0 |-> o0') ** on gpu_loc (output1 |-> o1') **
    pure (optimized_decoder_relation (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
            (chest1_to_seq s3) (chest1_to_seq s4) (chest1_to_seq s5) (chest1_to_seq s6)
            (chest1_to_seq s7) (chest1_to_seq s8) (chest1_to_seq s9) (chest1_to_seq s10)
            (FStar.Int64.v input11) (chest1_to_seq o0') (chest1_to_seq o1'))
{
  if (FStar.Int64.lte 0L input11 && FStar.Int64.lte input11 4088L) {
    let off = SZ.uint32_to_sizet (FStar.Int.Cast.int64_to_uint32 input11);
    assert (pure (SZ.v off == FStar.Int64.v input11));
    decoder_body input0 input1 input2 input3 input4 input5 input6 input7 input8 input9 input10 off output0 output1;
  } else {
    assert (pure (~(0 <= FStar.Int64.v input11 /\ FStar.Int64.v input11 <= 4088)));
    rel_vac (chest1_to_seq s0) (chest1_to_seq s1) (chest1_to_seq s2)
            (chest1_to_seq s3) (chest1_to_seq s4) (chest1_to_seq s5) (chest1_to_seq s6)
            (chest1_to_seq s7) (chest1_to_seq s8) (chest1_to_seq s9) (chest1_to_seq s10)
            (FStar.Int64.v input11) (chest1_to_seq o0) (chest1_to_seq o1);
  }
}
