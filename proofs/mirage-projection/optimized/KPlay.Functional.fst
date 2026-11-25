module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Injection
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Spec.GEMM
open KPlay.SharedSpec
open Kuiper.Tensor.Tiling
open Kuiper.Array2.Strided
module S = FStar.Seq
module SZ = Kuiper.SizeT
module MMA = Kuiper.TensorCore.MMA
module TMap = Kuiper.Kernel.TMap

(* ---------------- layouts ---------------- *)

#push-options "--fuel 2 --ifuel 2 --z3rlimit 40"
let rm_imap (m n : nat) (i : natlt m) (j : natlt n)
  : Lemma ((l2_row_major m n).imap.f (idx2 i j) == i * n + j)
  = ()

let cm_imap (m n : nat) (i : natlt m) (j : natlt n)
  : Lemma ((l2_col_major m n).imap.f (idx2 i j) == j * m + i)
  = ()
#pop-options

(* ---------------- physical spec ---------------- *)
let atile (sW : chest2 bf16 151936 1024) (sZ : chest2 bf16 16 16) (b : natlt 9600) (k : natlt 64)
  : chest2 bf16 16 16
  = if b < 9496 then ematrix_subtile sW 16 16 b k else sZ

let btile (#et:Type) (sX : chest2 et 1024 8) (k : natlt 64) : chest2 et 16 8
  = ematrix_subtile sX 16 8 k 0

let rec frag_iter (sW : chest2 bf16 151936 1024) (sZ : chest2 bf16 16 16) (sX : chest2 bf16 1024 8)
  (b : natlt 9600) (k : nat{k <= 64})
  : Tot (chest2 float 16 8) (decreases k)
  = if k = 0 then const (16 @| 8 @| INil) (zero #float)
    else MMA.emma (frag_iter sW sZ sX b (k - 1)) (atile sW sZ b (k - 1)) (btile sX (k - 1))

(* ---------------- ideal spec ---------------- *)
let ratile (rW : chest2 real 151936 1024) (b : natlt 9600) (k : natlt 64) : chest2 real 16 16
  = if b < 9496 then ematrix_subtile rW 16 16 b k else const (16 @| 16 @| INil) 0.0R

let rec rfrag (rW : chest2 real 151936 1024) (rX : chest2 real 1024 8)
  (b : natlt 9600) (k : nat{k <= 64})
  : Tot (chest2 real 16 8) (decreases k)
  = if k = 0 then const (16 @| 8 @| INil) 0.0R
    else matplus (rfrag rW rX b (k - 1)) (matmul (ratile rW b (k - 1)) (btile rX (k - 1)))

let rec frag_approx (sW : chest2 bf16 151936 1024) (sZ : chest2 bf16 16 16) (sX : chest2 bf16 1024 8)
  (rW : chest2 real 151936 1024) (rX : chest2 real 1024 8)
  (b : natlt 9600) (k : nat{k <= 64})
  : Lemma (requires sW %~ rW /\ sX %~ rX /\ sZ %~ const (16 @| 16 @| INil) 0.0R)
          (ensures frag_iter sW sZ sX b k %~ rfrag rW rX b k)
          (decreases k)
  = if k = 0 then (
      assert (frag_iter sW sZ sX b k %~ rfrag rW rX b k)
    ) else (
      frag_approx sW sZ sX rW rX b (k - 1);
      assert (atile sW sZ b (k - 1) %~ ratile rW b (k - 1));
      assert (btile sX (k - 1) %~ btile rX (k - 1));
      MMA.emma_approx_lemma (frag_iter sW sZ sX b (k - 1)) (atile sW sZ b (k - 1)) (btile sX (k - 1))
        (rfrag rW rX b (k - 1)) (ratile rW b (k - 1)) (btile rX (k - 1))
    )

(* ---------------- sdot algebra ---------------- *)
let rec sdot_split (m p : nat) (a : S.seq real) (ao : int) (x : S.seq real)
  : Lemma (ensures sdot (m + p) a ao 1 x 0 1 == sdot m a ao 1 x 0 1 +. sdot p a (ao + m) 1 x m 1)
          (decreases p)
  = if p = 0 then () else sdot_split m (p - 1) a ao x

let rec msum_tile (rW : chest2 real 151936 1024) (rX : chest2 real 1024 8)
  (r0 x : S.seq real) (b : natlt 9496) (k : natlt 64) (i : natlt 16) (n : natlt 8) (t : nat{t <= 16})
  : Lemma (requires (forall (ii : natlt 151936) (jj : natlt 1024). acc2 rW ii jj == at r0 (ii * 1024 + jj)) /\
                    (forall (kk : natlt 1024). acc2 rX kk n == at x kk))
          (ensures __matmul_single (ratile rW b k) (btile rX k) i n t ==
                   sdot t r0 ((16 * b + i) * 1024 + 16 * k) 1 x (16 * k) 1)
          (decreases t)
  = if t = 0 then () else msum_tile rW rX r0 x b k i n (t - 1)

let rec rfrag_val (rW : chest2 real 151936 1024) (rX : chest2 real 1024 8)
  (r0 x : S.seq real) (b : natlt 9496) (k : nat{k <= 64}) (i : natlt 16) (n : natlt 8)
  : Lemma (requires (forall (ii : natlt 151936) (jj : natlt 1024). acc2 rW ii jj == at r0 (ii * 1024 + jj)) /\
                    (forall (kk : natlt 1024). acc2 rX kk n == at x kk))
          (ensures acc2 (rfrag rW rX b k) i n == sdot (16 * k) r0 ((16 * b + i) * 1024) 1 x 0 1)
          (decreases k)
  = if k = 0 then () else (
      rfrag_val rW rX r0 x b (k - 1) i n;
      lemma_matmul_index (ratile rW b (k - 1)) (btile rX (k - 1)) i n;
      msum_tile rW rX r0 x b (k - 1) i n 16;
      sdot_split (16 * (k - 1)) 16 r0 ((16 * b + i) * 1024) x
    )

let rec msum_zero (rW : chest2 real 151936 1024) (rX : chest2 real 1024 8)
  (b : natlt 9600{b >= 9496}) (k : natlt 64) (i : natlt 16) (n : natlt 8) (t : nat{t <= 16})
  : Lemma (ensures __matmul_single (ratile rW b k) (btile rX k) i n t == 0.0R)
          (decreases t)
  = if t = 0 then () else msum_zero rW rX b k i n (t - 1)

let rec rfrag_zero (rW : chest2 real 151936 1024) (rX : chest2 real 1024 8)
  (b : natlt 9600{b >= 9496}) (k : nat{k <= 64}) (i : natlt 16) (n : natlt 8)
  : Lemma (ensures acc2 (rfrag rW rX b k) i n == 0.0R)
          (decreases k)
  = if k = 0 then () else (
      rfrag_zero rW rX b (k - 1) i n;
      lemma_matmul_index (ratile rW b (k - 1)) (btile rX (k - 1)) i n;
      msum_zero rW rX b (k - 1) i n 16
    )

(* ---------------- layout index lemmas ---------------- *)
#push-options "--fuel 2 --ifuel 2 --z3rlimit 40"
let rm_to_seq (#et:Type) (m n : pos) (c : chest2 et m n) (p : natlt (m * n))
  : Lemma (S.length (to_seq (l2_row_major m n) c) == m * n /\
           S.index (to_seq (l2_row_major m n) c) p == acc2 c (p / n) (p % n))
  = let ix = Kuiper.Injection.inverse_f (l2_row_major m n).imap p in
    let (i, (j, ())) = ix in
    rm_imap m n i j;
    FStar.Math.Lemmas.lemma_div_plus j i n;
    FStar.Math.Lemmas.small_div j n;
    FStar.Math.Lemmas.lemma_mod_plus j i n;
    FStar.Math.Lemmas.small_mod j n

let cm_to_seq (#et:Type) (m n : pos) (c : chest2 et m n) (p : natlt (m * n))
  : Lemma (S.length (to_seq (l2_col_major m n) c) == m * n /\
           S.index (to_seq (l2_col_major m n) c) p == acc2 c (p % m) (p / m))
  = let ix = Kuiper.Injection.inverse_f (l2_col_major m n).imap p in
    let (i, (j, ())) = ix in
    cm_imap m n i j;
    FStar.Math.Lemmas.lemma_div_plus i j m;
    FStar.Math.Lemmas.small_div i m;
    FStar.Math.Lemmas.lemma_mod_plus i j m;
    FStar.Math.Lemmas.small_mod i m

let rm_from_seq (#et:Type) (m n : pos) (s : S.seq et{S.length s == m * n}) (i : natlt m) (j : natlt n)
  : Lemma (i * n + j < m * n /\ acc2 (from_seq (l2_row_major m n) s) i j == S.index s (i * n + j))
  = rm_imap m n i j

let cm_from_seq (#et:Type) (m n : pos) (s : S.seq et{S.length s == m * n}) (i : natlt m) (j : natlt n)
  : Lemma (j * m + i < m * n /\ acc2 (from_seq (l2_col_major m n) s) i j == S.index s (j * m + i))
  = cm_imap m n i j
#pop-options

(* ---------------- approximation bridges ---------------- *)
let z_approx () : Lemma (zero #bf16 %~ 0.0R) = ()

let from_seq_approx (#r : nat) (#d : shape r) (l : full_tlayout d)
  (s : S.seq bf16 {S.length s == sizeof d}) (rs : S.seq real {S.length rs == sizeof d})
  : Lemma (requires s %~ rs) (ensures from_seq l s %~ from_seq l rs)
  = ()

let zero_approx (sZ : chest2 bf16 16 16)
  : Lemma (requires forall (i : natlt 16) (j : natlt 16). acc2 sZ i j == zero)
          (ensures sZ %~ const (16 @| 16 @| INil) 0.0R)
  = z_approx ()

(* ---------------- final assembly ---------------- *)
let staging_ok (sW : chest2 bf16 151936 1024) (sZ : chest2 bf16 16 16) (sX : chest2 bf16 1024 8)
  (cS : chest2 float 153600 8) : prop
  = forall (m : natlt 153600) (n : natlt 8).
      acc2 cS m n == acc2 (frag_iter sW sZ sX (m / 16) 64) (m % 16) n

let out0_ok (cS : chest2 float 153600 8) (c0 : chest2 bf16 8 153600) : prop
  = forall (n : natlt 8) (m : natlt 153600).
      acc2 c0 n m == Kuiper.Float.Casts.c_32_bf16.fcast (acc2 cS m n)

let out1_ok (sX : chest2 bf16 1024 8) (c1 : chest2 bf16 1024 7) : prop
  = forall (kk : natlt 1024) (n : natlt 7). acc2 c1 kk n == acc2 sX kk n

#push-options "--fuel 1 --ifuel 1 --z3rlimit 40"
let elem_lemma (s0 s1 : S.seq bf16) (sZ : chest2 bf16 16 16)
  (cS : chest2 float 153600 8) (c0 : chest2 bf16 8 153600)
  (r0 r1 : S.seq real) (t : natlt 8) (v : natlt 153600)
  : Lemma (requires S.length s0 == 155582464 /\ S.length s1 == 8192 /\
                    s0 %~ r0 /\ s1 %~ r1 /\
                    (forall (i : natlt 16) (j : natlt 16). acc2 sZ i j == zero) /\
                    staging_ok (from_seq lw s0) sZ (from_seq lx s1) cS /\ out0_ok cS c0)
          (ensures acc2 c0 t v %~
                   (if v < 151936 then sdot 1024 r0 (v * 1024) 1 (window r1 (t * 1024) 1024) 0 1 else 0.0R))
  = let sW = from_seq lw s0 in
    let sX = from_seq lx s1 in
    let rW = from_seq lw r0 in
    let rX = from_seq lx r1 in
    let x = window r1 (t * 1024) 1024 in
    let b : natlt 9600 = v / 16 in
    let i : natlt 16 = v % 16 in
    from_seq_approx lw s0 r0;
    from_seq_approx lx s1 r1;
    zero_approx sZ;
    frag_approx sW sZ sX rW rX b 64;
    assert (acc2 (frag_iter sW sZ sX b 64) i t %~ acc2 (rfrag rW rX b 64) i t);
    if v < 151936 then (
      let aux1 (ii : natlt 151936) (jj : natlt 1024) : Lemma (acc2 rW ii jj == at r0 (ii * 1024 + jj))
        = rm_from_seq 151936 1024 r0 ii jj in
      let aux2 (kk : natlt 1024) : Lemma (acc2 rX kk t == at x kk)
        = cm_from_seq 1024 8 r1 kk t in
      Classical.forall_intro_2 aux1;
      Classical.forall_intro aux2;
      rfrag_val rW rX r0 x b 64 i t;
      assert (16 * b + i == v)
    ) else (
      rfrag_zero rW rX b 64 i t
    )
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let row_lemma (s0 s1 : S.seq bf16) (sZ : chest2 bf16 16 16)
  (cS : chest2 float 153600 8) (c0 : chest2 bf16 8 153600)
  (r0 r1 : S.seq real) (t : natlt 8)
  : Lemma (requires S.length s0 == 155582464 /\ S.length s1 == 8192 /\
                    s0 %~ r0 /\ s1 %~ r1 /\
                    (forall (i : natlt 16) (j : natlt 16). acc2 sZ i j == zero) /\
                    staging_ok (from_seq lw s0) sZ (from_seq lx s1) cS /\ out0_ok cS c0)
          (ensures window (to_seq (l2_row_major 8 153600) c0) (t * 153600) 153600 %~
                   logits_row 153600 r0 (window r1 (t * 1024) 1024))
  = let o0 = to_seq (l2_row_major 8 153600) c0 in
    let x = window r1 (t * 1024) 1024 in
    rm_to_seq 8 153600 c0 0;
    let w = window o0 (t * 153600) 153600 in
    let lr = logits_row 153600 r0 x in
    reveal_opaque (`%logits_row) (logits_row 153600 r0 x);
    let aux (v : nat{v < 153600}) : Lemma (S.index w v %~ S.index lr v)
      = rm_to_seq 8 153600 c0 (t * 153600 + v);
        FStar.Math.Lemmas.lemma_div_plus v t 153600;
        FStar.Math.Lemmas.small_div v 153600;
        FStar.Math.Lemmas.lemma_mod_plus v t 153600;
        FStar.Math.Lemmas.small_mod v 153600;
        elem_lemma s0 s1 sZ cS c0 r0 r1 t v
    in
    Classical.forall_intro aux
#pop-options

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let out1_lemma (s1 : S.seq bf16 {S.length s1 == 8192}) (c1 : chest2 bf16 1024 7)
  : Lemma (requires out1_ok (from_seq lx s1) c1)
          (ensures to_seq (l2_col_major 1024 7) c1 == window s1 0 7168)
  = let o1 = to_seq (l2_col_major 1024 7) c1 in
    cm_to_seq 1024 7 c1 0;
    let aux (p : nat{p < 7168}) : Lemma (S.index o1 p == S.index (window s1 0 7168) p)
      = cm_to_seq 1024 7 c1 p;
        cm_from_seq 1024 8 s1 (p % 1024) (p / 1024);
        FStar.Math.Lemmas.euclidean_division_definition p 1024
    in
    Classical.forall_intro aux;
    assert (S.equal o1 (window s1 0 7168))
#pop-options

let final_lemma (s0 s1 : S.seq bf16) (sZ : chest2 bf16 16 16)
  (cS : chest2 float 153600 8) (c0 : chest2 bf16 8 153600) (c1 : chest2 bf16 1024 7)
  : Lemma (requires S.length s0 == 155582464 /\ S.length s1 == 8192 /\
                    (forall (i : natlt 16) (j : natlt 16). acc2 sZ i j == zero) /\
                    staging_ok (from_seq lw s0) sZ (from_seq lx s1) cS /\ out0_ok cS c0 /\
                    out1_ok (from_seq lx s1) c1)
          (ensures optimized_projection_relation s0 s1
                     (to_seq (l2_row_major 8 153600) c0) (to_seq (l2_col_major 1024 7) c1))
  = rm_to_seq 8 153600 c0 0;
    out1_lemma s1 c1;
    let aux (r0 r1 : S.seq real)
      : Lemma (s0 %~ r0 /\ s1 %~ r1 ==>
               (forall (t:nat). t < 8 ==>
                 window (to_seq (l2_row_major 8 153600) c0) (t * 153600) 153600 %~
                 logits_row 153600 r0 (window r1 (t * 1024) 1024)))
      = let aux2 (t : nat) : Lemma (s0 %~ r0 /\ s1 %~ r1 /\ t < 8 ==>
                 window (to_seq (l2_row_major 8 153600) c0) (t * 153600) 153600 %~
                 logits_row 153600 r0 (window r1 (t * 1024) 1024))
          = if t < 8 then Classical.move_requires (row_lemma s0 s1 sZ cS c0 r0 r1) t in
        Classical.forall_intro aux2
    in
    Classical.forall_intro_2 aux

(* ---------------- staging spec ---------------- *)
let rec stage (sW : chest2 bf16 151936 1024) (sZ : chest2 bf16 16 16) (sX : chest2 bf16 1024 8)
  (c0 : chest2 float 153600 8) (b : nat{b <= 9600})
  : Tot (chest2 float 153600 8) (decreases b)
  = if b = 0 then c0
    else update_tile (stage sW sZ sX c0 (b - 1)) 16 8 (b - 1) 0 (frag_iter sW sZ sX (b - 1) 64)

(* ---------------- Pulse: MMA kernel ---------------- *)
inline_for_extraction noextract instance c1024 : concrete_sz 1024 = { x = 1024sz; }

let lz = l2_row_major 16 16
let ls = l2_row_major 153600 8

inline_for_extraction noextract
fn mma_tile
  (w : array2 bf16 lw) (x : array2 bf16 lx) (z : array2 bf16 lz)
  (fr : MMA.fragment) (bv : szlt 9600)
  (#fw #fx #fz : perm)
  (#sW : chest2 bf16 151936 1024) (#sX : chest2 bf16 1024 8) (#sZ : chest2 bf16 16 16)
  preserves gpu ** (w |-> Frac fw sW) ** (x |-> Frac fx sX) ** (z |-> Frac fz sZ)
  requires exists* v. fr |-> v
  ensures fr |-> frag_iter sW sZ sX (SZ.v bv) 64
{
  tensor_pts_to_ref w;
  tensor_pts_to_ref x;
  MMA.fill fr zero;
  let mut k : szle 64 = 0sz;
  while (!k <^ 64sz)
    invariant exists* (vk : szle 64). k |-> vk **
      (fr |-> frag_iter sW sZ sX (SZ.v bv) (SZ.v vk))
    decreases (64 - SZ.v !k)
  {
    let kv = !k;
    let xt = array2_extract_tile_ro' x 16 8 (SZ.v kv) 0;
    if (bv <^ 9496sz) {
      let wt = array2_extract_tile_ro' w 16 16 (SZ.v bv) (SZ.v kv);
      MMA.mma_sync wt xt fr;
      ambig_trade_elim ();
    } else {
      MMA.mma_sync z xt fr;
    };
    ambig_trade_elim ();
    k := kv +^ 1sz;
  };
}

unfold let qf : perm = 1.0R /. 32

inline_for_extraction noextract
fn store_tile
  (s : array2 float ls) (fr : MMA.fragment) (bv : szlt 9600)
  (#c : chest2 float 153600 8) (#v : chest2 float 16 8)
  preserves gpu ** (fr |-> v)
  requires s |-> Frac qf c
  ensures s |-> Frac qf (update_tile c 16 8 (SZ.v bv) 0 v)
{
  tensor_pts_to_ref s;
  array2_extract_tile s 16 8 (SZ.v bv) 0;
  rewrite (array2_subtile s 16 8 (SZ.v bv) 0 |-> Frac qf (ematrix_subtile c 16 8 (SZ.v bv) 0))
       as (array2_subtile s 16 8 (SZ.v bv) 0 |-> Frac (1.0R /. Kuiper.Barrier.Warp.warp_size) (ematrix_subtile c 16 8 (SZ.v bv) 0));
  MMA.store fr (array2_subtile s 16 8 (SZ.v bv) 0);
  rewrite (array2_subtile s 16 8 (SZ.v bv) 0 |-> Frac (1.0R /. Kuiper.Barrier.Warp.warp_size) v)
       as (array2_subtile s 16 8 (SZ.v bv) 0 |-> Frac qf v);
  Pulse.Lib.Forall.elim_forall v;
  Pulse.Lib.Trade.elim_trade _ _;
}

inline_for_extraction noextract
fn mma_thread
  (w : array2 bf16 lw) (x : array2 bf16 lx) (z : array2 bf16 lz) (s : array2 float ls)
  (#fw #fx #fz : perm)
  (#sW : chest2 bf16 151936 1024) (#sX : chest2 bf16 1024 8) (#sZ : chest2 bf16 16 16)
  (#c0 : chest2 float 153600 8)
  preserves gpu ** (w |-> Frac fw sW) ** (x |-> Frac fx sX) ** (z |-> Frac fz sZ)
  requires s |-> Frac qf c0
  ensures s |-> Frac qf (stage sW sZ sX c0 9600)
{
  let fr = MMA.alloc_fragment ();
  let mut b : szle 9600 = 0sz;
  while (!b <^ 9600sz)
    invariant exists* (vb : szle 9600). b |-> vb **
      (exists* v. fr |-> v) **
      (s |-> Frac qf (stage sW sZ sX c0 (SZ.v vb)))
    decreases (9600 - SZ.v !b)
  {
    let bv = !b;
    mma_tile w x z fr bv;
    store_tile s fr bv;
    b := bv +^ 1sz;
  };
  with v. assert (fr |-> v);
  drop_ (fr |-> v);
}

(* ---------------- kernel A descriptor: one warp, as in the CUDA ---------------- *)
let kpre_a (w : array2 bf16 lw) (x : array2 bf16 lx) (z : array2 bf16 lz) (s : array2 float ls)
  (fw fx fz : perm) (sW : chest2 bf16 151936 1024) (sX : chest2 bf16 1024 8) (sZ : chest2 bf16 16 16)
  (c : chest2 float 153600 8) (tid : natlt 32) : slprop
  = ((w |-> Frac (fw /. 32) sW) ** (x |-> Frac (fx /. 32) sX)) **
    ((z |-> Frac (fz /. 32) sZ) ** (s |-> Frac (1.0R /. 32) c))

ghost
fn setup_a
  (w : array2 bf16 lw) (x : array2 bf16 lx) (z : array2 bf16 lz) (s : array2 float ls)
  (fw fx fz : perm) (sW : chest2 bf16 151936 1024) (sX : chest2 bf16 1024 8) (sZ : chest2 bf16 16 16)
  (c0 : chest2 float 153600 8) ()
  norewrite
  requires ((w |-> Frac fw sW) ** (x |-> Frac fx sX)) ** ((z |-> Frac fz sZ) ** (s |-> c0))
  ensures (forall+ (tid : natlt 32). kpre_a w x z s fw fx fz sW sX sZ c0 tid) ** emp
{
  tensor_share_n w 32;
  tensor_share_n x 32;
  tensor_share_n z 32;
  tensor_share_n s 32;
  forevery_zip #(natlt 32) (fun _ -> w |-> Frac (fw /. 32) sW) (fun _ -> x |-> Frac (fx /. 32) sX);
  forevery_zip #(natlt 32) (fun _ -> z |-> Frac (fz /. 32) sZ) (fun _ -> s |-> Frac (1.0R /. 32) c0);
  forevery_zip #(natlt 32)
    (fun _ -> (w |-> Frac (fw /. 32) sW) ** (x |-> Frac (fx /. 32) sX))
    (fun _ -> (z |-> Frac (fz /. 32) sZ) ** (s |-> Frac (1.0R /. 32) c0));
  forevery_ext #(natlt 32)
    (fun _ -> ((w |-> Frac (fw /. 32) sW) ** (x |-> Frac (fx /. 32) sX)) **
              ((z |-> Frac (fz /. 32) sZ) ** (s |-> Frac (1.0R /. 32) c0)))
    (kpre_a w x z s fw fx fz sW sX sZ c0);
}

ghost
fn teardown_a
  (w : array2 bf16 lw) (x : array2 bf16 lx) (z : array2 bf16 lz) (s : array2 float ls)
  (fw fx fz : perm) (sW : chest2 bf16 151936 1024) (sX : chest2 bf16 1024 8) (sZ : chest2 bf16 16 16)
  (c1 : chest2 float 153600 8) ()
  norewrite
  requires (forall+ (tid : natlt 32). kpre_a w x z s fw fx fz sW sX sZ c1 tid) ** emp
  ensures ((w |-> Frac fw sW) ** (x |-> Frac fx sX)) ** ((z |-> Frac fz sZ) ** (s |-> c1))
{
  forevery_ext #(natlt 32)
    (kpre_a w x z s fw fx fz sW sX sZ c1)
    (fun _ -> ((w |-> Frac (fw /. 32) sW) ** (x |-> Frac (fx /. 32) sX)) **
              ((z |-> Frac (fz /. 32) sZ) ** (s |-> Frac (1.0R /. 32) c1)));
  forevery_unzip #(natlt 32)
    (fun _ -> (w |-> Frac (fw /. 32) sW) ** (x |-> Frac (fx /. 32) sX))
    (fun _ -> (z |-> Frac (fz /. 32) sZ) ** (s |-> Frac (1.0R /. 32) c1));
  forevery_unzip #(natlt 32) (fun _ -> w |-> Frac (fw /. 32) sW) (fun _ -> x |-> Frac (fx /. 32) sX);
  forevery_unzip #(natlt 32) (fun _ -> z |-> Frac (fz /. 32) sZ) (fun _ -> s |-> Frac (1.0R /. 32) c1);
  tensor_gather_n w 32;
  tensor_gather_n x 32;
  tensor_gather_n z 32;
  tensor_gather_n s 32;
}

inline_for_extraction noextract
fn body_a
  (w : array2 bf16 lw) (x : array2 bf16 lx) (z : array2 bf16 lz) (s : array2 float ls)
  (fw fx fz : perm) (sW : chest2 bf16 151936 1024) (sX : chest2 bf16 1024 8) (sZ : chest2 bf16 16 16)
  (c0 : chest2 float 153600 8)
  (tid : szlt 32sz) ()
  norewrite
  requires gpu ** kpre_a w x z s fw fx fz sW sX sZ c0 (SZ.v tid) ** thread_id 32sz tid
  ensures gpu ** kpre_a w x z s fw fx fz sW sX sZ (stage sW sZ sX c0 9600) (SZ.v tid) ** thread_id 32sz tid
{
  unfold kpre_a;
  mma_thread w x z s;
  fold kpre_a w x z s fw fx fz sW sX sZ (stage sW sZ sX c0 9600) (SZ.v tid);
}

inline_for_extraction noextract
let kdesc_a
  (w : array2 bf16 lw {is_global w}) (x : array2 bf16 lx {is_global x})
  (z : array2 bf16 lz {is_global z}) (s : array2 float ls {is_global s})
  (fw fx fz : perm) (sW : chest2 bf16 151936 1024) (sX : chest2 bf16 1024 8) (sZ : chest2 bf16 16 16)
  (c0 : chest2 float 153600 8)
  : kernel_desc_1_n
      (((w |-> Frac fw sW) ** (x |-> Frac fx sX)) ** ((z |-> Frac fz sZ) ** (s |-> c0)))
      (((w |-> Frac fw sW) ** (x |-> Frac fx sX)) ** ((z |-> Frac fz sZ) ** (s |-> stage sW sZ sX c0 9600)))
  = {
    nthr = 32sz;
    frame = emp;
    kpre = kpre_a w x z s fw fx fz sW sX sZ c0;
    kpost = kpre_a w x z s fw fx fz sW sX sZ (stage sW sZ sX c0 9600);
    block_setup = setup_a w x z s fw fx fz sW sX sZ c0;
    block_teardown = teardown_a w x z s fw fx fz sW sX sZ (stage sW sZ sX c0 9600);
    f = body_a w x z s fw fx fz sW sX sZ c0;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve;
  }

(* ---------------- Pulse: cast and copy kernels ---------------- *)
inline_for_extraction noextract
let sz1228800 : (x:sz{SZ.v x == 1228800}) = FStar.SizeT.uint32_to_sizet 1228800ul

inline_for_extraction noextract instance ct_ls : ctlayout ls = c_l2_row_major 153600 8sz
inline_for_extraction noextract instance ct_lo0 : ctlayout lo0 = c_l2_row_major 8 sz153600
inline_for_extraction noextract instance ct_lo1 : ctlayout lo1 = c_l2_col_major 1024sz 7
inline_for_extraction noextract instance ct_lx : ctlayout lx = c_l2_col_major 1024sz 8
inline_for_extraction noextract instance ct_lz : ctlayout lz = c_l2_row_major 16 16sz

let vf_b (cS : chest2 float 153600 8) (i : abs (8 @| 153600 @| INil)) (_ : bf16) (r : bf16) : prop
  = let (n, (m, ())) = i in r == Kuiper.Float.Casts.c_32_bf16.fcast (acc2 cS m n)

inline_for_extraction noextract
let idx_swap (#a #b : erased nat) (i : szlt a & (szlt b & unit)) : szlt b & (szlt a & unit)
  = match i with | (x, (y, u)) -> (y, (x, u))

inline_for_extraction noextract
fn cast_cell
  (s : array2 float ls) (#cS : chest2 float 153600 8) (#fr : perm)
  (i : conc (8 @| 153600 @| INil)) (x : bf16)
  norewrite
  preserves gpu ** (s |-> Frac fr cS)
  returns r : bf16
  ensures pure (vf_b cS (up i) x r)
{
  let v = read s (idx_swap #8 #153600 i);
  Kuiper.Float.Casts.c_32_bf16.fcast v
}

inline_for_extraction noextract
let kdesc_b
  (s : array2 float ls {is_global s}) (o : array2 bf16 lo0 {is_global o})
  (cS : chest2 float 153600 8) (#c0 : chest2 bf16 8 153600) (#fs : perm)
  = TMap.kmap (CCons 8sz (CCons #_ #153600 sz153600 CNil))
      (fun fr -> s |-> Frac fr cS)
      #(tensor_pts_to_shareable s cS)
      (vf_b cS)
      (cast_cell s #cS)
      #lo0 #ct_lo0 sz1228800 o #c0 #_ #fs

let vf_c (sX : chest2 bf16 1024 8) (i : abs (1024 @| 7 @| INil)) (_ : bf16) (r : bf16) : prop
  = let (k, (n, ())) = i in r == acc2 sX k n

inline_for_extraction noextract
let idx_widen (i : szlt 1024 & (szlt 7 & unit)) : szlt 1024 & (szlt 8 & unit)
  = match i with | (x, (y, u)) -> (x, ((y <: szlt 8), u))

inline_for_extraction noextract
fn copy_cell
  (x : array2 bf16 lx) (#sX : chest2 bf16 1024 8) (#fr : perm)
  (i : conc (1024 @| 7 @| INil)) (y : bf16)
  norewrite
  preserves gpu ** (x |-> Frac fr sX)
  returns r : bf16
  ensures pure (vf_c sX (up i) y r)
{
  read x (idx_widen i)
}

inline_for_extraction noextract
let kdesc_c
  (x : array2 bf16 lx {is_global x}) (o : array2 bf16 lo1 {is_global o})
  (sX : chest2 bf16 1024 8) (#c1 : chest2 bf16 1024 7) (#fx : perm)
  = TMap.kmap (CCons 1024sz (CCons 7sz CNil))
      (fun fr -> x |-> Frac fr sX)
      #(tensor_pts_to_shareable x sX)
      (vf_c sX)
      (copy_cell x #sX)
      #lo1 #ct_lo1 7168sz o #c1 #_ #fx

(* ---------------- glue lemmas ---------------- *)
let from_to_seq (#et:Type) (#r:nat) (#d:shape r) (l : full_tlayout d) (c : chest d et)
  : Lemma (from_seq l (to_seq l c) == c)
  = let aux (i : abs d) : Lemma (acc (from_seq l (to_seq l c)) i == acc c i) = () in
    Classical.forall_intro aux;
    assert (Kuiper.Chest.equal (from_seq l (to_seq l c)) c)

#push-options "--fuel 0 --ifuel 0 --z3rlimit 40"
let update_tile_acc (em : chest2 float 153600 8) (b : natlt 9600) (tm : chest2 float 16 8)
  (m : natlt 153600) (n : natlt 8)
  : Lemma (acc2 (update_tile em 16 8 b 0 tm) m n == (if m / 16 = b then acc2 tm (m % 16) n else acc2 em m n))
  = FStar.Math.Lemmas.small_div n 8;
    FStar.Math.Lemmas.small_mod n 8
#pop-options

#push-options "--fuel 1 --ifuel 0 --z3rlimit 40"
let rec stage_lemma (sW : chest2 bf16 151936 1024) (sZ : chest2 bf16 16 16) (sX : chest2 bf16 1024 8)
  (c0 : chest2 float 153600 8) (b : nat{b <= 9600}) (m : natlt 153600) (n : natlt 8)
  : Lemma (requires m < 16 * b)
          (ensures acc2 (stage sW sZ sX c0 b) m n == acc2 (frag_iter sW sZ sX (m / 16) 64) (m % 16) n)
          (decreases b)
  = update_tile_acc (stage sW sZ sX c0 (b - 1)) (b - 1) (frag_iter sW sZ sX (b - 1) 64) m n;
    if m / 16 = b - 1 then ()
    else (
      FStar.Math.Lemmas.euclidean_division_definition m 16;
      assert (m / 16 < b);
      assert (m / 16 <= b - 2);
      assert (m < 16 * (b - 1));
      stage_lemma sW sZ sX c0 (b - 1) m n
    )
#pop-options

let stage_ok (sW : chest2 bf16 151936 1024) (sZ : chest2 bf16 16 16) (sX : chest2 bf16 1024 8)
  (c0 : chest2 float 153600 8)
  : Lemma (staging_ok sW sZ sX (stage sW sZ sX c0 9600))
  = let aux (m : natlt 153600) (n : natlt 8)
      : Lemma (acc2 (stage sW sZ sX c0 9600) m n == acc2 (frag_iter sW sZ sX (m / 16) 64) (m % 16) n)
      = stage_lemma sW sZ sX c0 9600 m n in
    Classical.forall_intro_2 aux

let out0_from (cS : chest2 float 153600 8) (c0 c0' : chest2 bf16 8 153600)
  : Lemma (requires chest_foralli (fun i x -> vf_b cS i (acc c0 i) x) c0')
          (ensures out0_ok cS c0')
  = let aux (n : natlt 8) (m : natlt 153600)
      : Lemma (acc2 c0' n m == Kuiper.Float.Casts.c_32_bf16.fcast (acc2 cS m n))
      = assert (vf_b cS (n, (m, ())) (acc c0 (n, (m, ()))) (acc c0' (n, (m, ())))) in
    Classical.forall_intro_2 aux

let out1_from (sX : chest2 bf16 1024 8) (c1 c1' : chest2 bf16 1024 7)
  : Lemma (requires chest_foralli (fun i x -> vf_c sX i (acc c1 i) x) c1')
          (ensures out1_ok sX c1')
  = let aux (k : natlt 1024) (n : natlt 7)
      : Lemma (acc2 c1' k n == acc2 sX k n)
      = assert (vf_c sX (k, (n, ())) (acc c1 (k, (n, ()))) (acc c1' (k, (n, ())))) in
    Classical.forall_intro_2 aux

let zero_fn (_ : bf16) : bf16 = zero

let zero_ok (eZ : chest2 bf16 16 16)
  : Lemma (forall (i : natlt 16) (j : natlt 16). acc2 (chest_map zero_fn eZ) i j == zero)
  = ()

let finish_lemma (sW : chest2 bf16 151936 1024) (sX : chest2 bf16 1024 8) (sZ : chest2 bf16 16 16)
  (cS : chest2 float 153600 8) (c0 : chest2 bf16 8 153600) (c1 : chest2 bf16 1024 7)
  : Lemma (requires (forall (i : natlt 16) (j : natlt 16). acc2 sZ i j == zero) /\
                    staging_ok sW sZ sX cS /\ out0_ok cS c0 /\ out1_ok sX c1)
          (ensures optimized_projection_relation (to_seq lw sW) (to_seq lx sX) (to_seq lo0 c0) (to_seq lo1 c1))
  = from_to_seq lw sW;
    from_to_seq lx sX;
    final_lemma (to_seq lw sW) (to_seq lx sX) sZ cS c0 c1

(* ---------------- entry point ---------------- *)
fn kplay_run
  (input0 : array2 bf16 lw {is_global input0})
  (input1 : array2 bf16 lx {is_global input1})
  (output0 : array2 bf16 lo0 {is_global output0})
  (output1 : array2 bf16 lo1 {is_global output1})
  (#p0 #p1 : perm)
  (#sW : chest2 bf16 151936 1024) (#sX : chest2 bf16 1024 8)
  (#c0 : chest2 bf16 8 153600) (#c1 : chest2 bf16 1024 7)
  preserves cpu ** on gpu_loc (input0 |-> Frac p0 sW) ** on gpu_loc (input1 |-> Frac p1 sX)
  requires on gpu_loc (output0 |-> c0) ** on gpu_loc (output1 |-> c1)
  ensures exists* (c0' : chest2 bf16 8 153600) (c1' : chest2 bf16 1024 7).
    on gpu_loc (output0 |-> c0') ** on gpu_loc (output1 |-> c1') **
    pure (optimized_projection_relation (to_seq lw sW) (to_seq lx sX) (to_seq lo0 c0') (to_seq lo1 c1'))
{
  let z = alloc0 #bf16 256sz lz;
  let s = alloc0 #float sz1228800 ls;
  with eZ. assert on gpu_loc (z |-> eZ);
  with eS. assert on gpu_loc (s |-> eS);
  TMap.map_gpu (CCons 16sz (CCons 16sz CNil)) zero_fn #lz #ct_lz 256sz z;
  zero_ok eZ;
  launch_sync (kdesc_a input0 input1 z s p0 p1 1.0R sW sX (chest_map zero_fn eZ) eS);
  stage_ok sW (chest_map zero_fn eZ) sX eS;
  launch_sync (kdesc_b s output0 (stage sW (chest_map zero_fn eZ) sX eS 9600) #c0 #1.0R);
  with c0'. assert on gpu_loc (output0 |-> c0');
  out0_from (stage sW (chest_map zero_fn eZ) sX eS 9600) c0 c0';
  launch_sync (kdesc_c input1 output1 sX #c1 #p1);
  with c1'. assert on gpu_loc (output1 |-> c1');
  out1_from sX c1 c1';
  finish_lemma sW sX (chest_map zero_fn eZ) (stage sW (chest_map zero_fn eZ) sX eS 9600) c0' c1';
  free s;
  free z;
}
