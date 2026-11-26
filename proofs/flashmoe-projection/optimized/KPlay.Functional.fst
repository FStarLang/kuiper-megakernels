module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.EMatrix
open Kuiper.EMatrix.Tiling
open Kuiper.Tensor.Tiling
open Kuiper.Spec.GEMM
open Kuiper.TensorCore.MMA
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major, l2_col_major }
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I32 = FStar.Int32
module I64 = FStar.Int64

(* Pure math: MMA tiles approximate the real dot products of the spec. *)
#push-options "--fuel 1 --ifuel 1 --z3rlimit 20"

let rA (rx : S.seq real) : chest2 real 256 512 =
  mk2 (fun t k -> ridx rx (t * 512 + k))

let rB (rw : S.seq real) : chest2 real 512 8 =
  mk2 (fun k n -> ridx rw (k * 8 + n))

let subA (#et : Type) (a : chest2 et 256 512) (ti : natlt 16) (kb : natlt 32)
  : chest2 et 16 16 = ematrix_subtile a 16 16 ti kb

let subB (#et : Type) (b : chest2 et 512 8) (kb : natlt 32)
  : chest2 et 16 8 = ematrix_subtile b 16 8 kb 0

let rec ntile (xa : chest2 bf16 256 512) (wb : chest2 bf16 512 8)
  (ti : natlt 16) (n : nat { n <= 32 })
  : Tot (chest2 float 16 8) (decreases n)
  = if n = 0 then const (16 @| 8 @| INil) (zero #float)
    else emma (ntile xa wb ti (n - 1)) (subA xa ti (n - 1)) (subB wb (n - 1))

let rec rtile (rx rw : S.seq real) (ti : natlt 16) (n : nat { n <= 32 })
  : GTot (chest2 real 16 8) (decreases n)
  = if n = 0 then const (16 @| 8 @| INil) 0.0R
    else matplus (rtile rx rw ti (n - 1))
                 (matmul (subA (rA rx) ti (n - 1)) (subB (rB rw) (n - 1)))

let tile_bound (t : nat) (tsz : pos) (n : nat) (j : natlt tsz)
  : Lemma (requires t < n / tsz) (ensures t * tsz + j < n)
  = FStar.Math.Lemmas.lemma_div_mod n tsz;
    FStar.Math.Lemmas.distributivity_add_left t 1 tsz

let subtile_approx (#rows #cols : nat)
  (a : chest2 bf16 rows cols) (ra : chest2 real rows cols)
  (trows : pos { trows /? rows }) (tcols : pos { tcols /? cols })
  (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
  : Lemma (requires a %~ ra)
          (ensures ematrix_subtile a trows tcols tr tc %~ ematrix_subtile ra trows tcols tr tc)
  = introduce forall (i : natlt trows) (j : natlt tcols).
      acc2 (ematrix_subtile a trows tcols tr tc) i j %~ acc2 (ematrix_subtile ra trows tcols tr tc) i j
    with begin
      tile_bound tr trows rows i;
      tile_bound tc tcols cols j;
      subtile_acc2 a trows tcols tr tc i j;
      subtile_acc2 ra trows tcols tr tc i j;
      assert (acc a (tr * trows + i, (tc * tcols + j, ())) %~ acc ra (tr * trows + i, (tc * tcols + j, ())))
    end;
    lemma_approximates_intro (ematrix_subtile a trows tcols tr tc) (ematrix_subtile ra trows tcols tr tc)

let rec ntile_approx (xa : chest2 bf16 256 512) (wb : chest2 bf16 512 8)
  (rx rw : S.seq real) (ti : natlt 16) (n : nat { n <= 32 })
  : Lemma (requires xa %~ rA rx /\ wb %~ rB rw)
          (ensures ntile xa wb ti n %~ rtile rx rw ti n)
          (decreases n)
  = if n = 0 then begin
      let z : chest2 float 16 8 = const (16 @| 8 @| INil) (zero #float) in
      let rz : chest2 real 16 8 = const (16 @| 8 @| INil) 0.0R in
      introduce forall (i : natlt 16) (j : natlt 8). acc2 z i j %~ acc2 rz i j
      with (a0 #float);
      lemma_approximates_intro z rz
    end else begin
      ntile_approx xa wb rx rw ti (n - 1);
      subtile_approx xa (rA rx) 16 16 ti (n - 1);
      subtile_approx wb (rB rw) 16 8 (n - 1) 0;
      emma_approx_lemma (ntile xa wb ti (n - 1)) (subA xa ti (n - 1)) (subB wb (n - 1))
        (rtile rx rw ti (n - 1)) (subA (rA rx) ti (n - 1)) (subB (rB rw) (n - 1))
    end

let rec partial_lemma (rx rw : S.seq real) (ti : natlt 16) (nb : natlt 32)
  (i : natlt 16) (j : natlt 8) (m : nat { m <= 16 })
  : Lemma (ensures
      dot_prefix 512 8 rx rw (ti * 16 + i) j (16 * nb + m) ==
      dot_prefix 512 8 rx rw (ti * 16 + i) j (16 * nb) +.
        __matmul_single (subA (rA rx) ti nb) (subB (rB rw) nb) i j m)
    (decreases m)
  = if m = 0 then ()
    else begin
      partial_lemma rx rw ti nb i j (m - 1);
      matmul_single_lemma (subA (rA rx) ti nb) (subB (rB rw) nb) i j m;
      subtile_acc2 (rA rx) 16 16 ti nb i (m - 1);
      subtile_acc2 (rB rw) 16 8 nb 0 (m - 1) j
    end

let rec rtile_value (rx rw : S.seq real) (ti : natlt 16) (n : nat { n <= 32 })
  (i : natlt 16) (j : natlt 8)
  : Lemma (ensures acc2 (rtile rx rw ti n) i j ==
                   dot_prefix 512 8 rx rw (ti * 16 + i) j (16 * n))
          (decreases n)
  = if n = 0 then ()
    else begin
      rtile_value rx rw ti (n - 1) i j;
      chest_comb_acc2 add (rtile rx rw ti (n - 1))
        (matmul (subA (rA rx) ti (n - 1)) (subB (rB rw) (n - 1))) i j;
      lemma_matmul_index (subA (rA rx) ti (n - 1)) (subB (rB rw) (n - 1)) i j;
      partial_lemma rx rw ti (n - 1) i j 16
    end

let rec scores_after (s0 : chest2 float 256 8)
  (xa : chest2 bf16 256 512) (wb : chest2 bf16 512 8) (n : nat { n <= 16 })
  : Tot (chest2 float 256 8) (decreases n)
  = if n = 0 then s0
    else update_tile (scores_after s0 xa wb (n - 1)) 16 8 (n - 1) 0 (ntile xa wb (n - 1) 32)

let rec scores_after_value (s0 : chest2 float 256 8)
  (xa : chest2 bf16 256 512) (wb : chest2 bf16 512 8) (n : nat { n <= 16 })
  (t : natlt 256) (j : natlt 8)
  : Lemma (requires t < 16 * n)
          (ensures acc2 (scores_after s0 xa wb n) t j == acc2 (ntile xa wb (t / 16) 32) (t % 16) j)
          (decreases n)
  = if t / 16 = n - 1 then ()
    else scores_after_value s0 xa wb (n - 1) t j

#push-options "--fuel 2 --ifuel 2 --z3rlimit 40"
let row_major_imap (m n : nat) (i : natlt m) (j : natlt n)
  : Lemma ((l2_row_major m n).imap.f (i, (j, ())) == i * n + j)
  = ()
#pop-options

let to_seq_row_major (#et : Type) (#m #n : nat) (s : chest2 et m n) (i : natlt m) (j : natlt n)
  : Lemma (i * n + j < m * n /\
           S.index (to_seq (l2_row_major m n) s) (i * n + j) == acc2 s i j)
  = row_major_imap m n i j;
    let l = l2_row_major m n in
    Kuiper.Injection.lem_pat l.imap (i, (j, ()))

let chest_of_seq_approx (#m #n : nat) (s : chest2 bf16 m n) (r : S.seq real)
  : Lemma (requires to_seq (l2_row_major m n) s %~ r)
          (ensures s %~ (mk2 (fun i j -> ridx r (i * n + j)) <: chest2 real m n))
  = introduce forall (i : natlt m) (j : natlt n).
      acc2 s i j %~ acc2 (mk2 (fun i j -> ridx r (i * n + j)) <: chest2 real m n) i j
    with to_seq_row_major s i j;
    lemma_approximates_intro s (mk2 (fun i j -> ridx r (i * n + j)))

let score_entry_ok (s0 : chest2 float 256 8)
  (xa : chest2 bf16 256 512) (wb : chest2 bf16 512 8)
  (rx rw : S.seq real) (t : natlt 256) (e : natlt 8)
  : Lemma (requires xa %~ rA rx /\ wb %~ rB rw)
          (ensures acc2 (scores_after s0 xa wb 16) t e %~ dot_prefix 512 8 rx rw t e 512)
  = let ti : natlt 16 = t / 16 in
    let i : natlt 16 = t % 16 in
    assert (ti * 16 + i == t);
    scores_after_value s0 xa wb 16 t e;
    ntile_approx xa wb rx rw ti 32;
    rtile_value rx rw ti 32 i e;
    assert (acc2 (ntile xa wb ti 32) i e %~ acc2 (rtile rx rw ti 32) i e)

let final_scores_ok (s0 : chest2 float 256 8)
  (xa : chest2 bf16 256 512) (wb : chest2 bf16 512 8)
  : Lemma (scores_ok (to_seq (l2_row_major 256 512) xa) (to_seq (l2_row_major 512 8) wb)
                     (to_seq (l2_row_major 256 8) (scores_after s0 xa wb 16)))
  = let sc = scores_after s0 xa wb 16 in
    introduce forall (rx rw : S.seq real).
      (to_seq (l2_row_major 256 512) xa %~ rx /\ to_seq (l2_row_major 512 8) wb %~ rw) ==>
      to_seq (l2_row_major 256 8) sc %~ ref_scores rx rw
    with introduce _ ==> _
    with begin
      chest_of_seq_approx xa rx;
      chest_of_seq_approx wb rw;
      introduce forall (k : nat { k < 2048 }).
        S.index (to_seq (l2_row_major 256 8) sc) k %~ S.index (ref_scores rx rw) k
      with begin
        let t : natlt 256 = k / 8 in
        let e : natlt 8 = k % 8 in
        to_seq_row_major sc t e;
        score_entry_ok s0 xa wb rx rw t e
      end
    end

#pop-options

(* Kernel A helpers: fills and the weight transpose. *)
inline_for_extraction noextract
instance c_w_rm : ctlayout (l2_row_major 512 8) = Kuiper.Tensor.Layout.Alg.c_l2_row_major 512 8sz

inline_for_extraction noextract
instance c_w_cm : ctlayout (l2_col_major 512 8) = Kuiper.Tensor.Layout.Alg.c_l2_col_major 512sz 8

inline_for_extraction noextract
fn fill1 (#et : Type0) (n : szp) (a : array1 et (l1_forward (SZ.v n))) (v : et)
  (#s0 : chest1 et (SZ.v n))
  requires a |-> s0
  ensures exists* (s : chest1 et (SZ.v n)). a |-> s ** pure (forall (j : natlt (SZ.v n)). acc s (j, ()) == v)
{
  let mut i : szle (SZ.v n) = 0sz;
  while (!i <^ n)
    invariant exists* (vi : szle (SZ.v n)) (s : chest1 et (SZ.v n)).
      i |-> vi ** a |-> s ** pure (forall (j : natlt (SZ.v n)). j < SZ.v vi ==> acc s (j, ()) == v)
    decreases (SZ.v n - SZ.v !i)
  {
    let iv = !i;
    Kuiper.Tensor.write a ((iv <: szlt (SZ.v n)), ()) v;
    i := iv +^ 1sz;
  };
}

inline_for_extraction noextract
fn transpose_w
  (w : array2 bf16 (l2_row_major 512 8))
  (wt : array2 bf16 (l2_col_major 512 8))
  (#fw : perm) (#ws : chest2 bf16 512 8) (#wt0 : chest2 bf16 512 8)
  preserves w |-> Frac fw ws
  requires wt |-> wt0
  ensures wt |-> ws
{
  let mut k : szle 512 = 0sz;
  while (!k <^ 512sz)
    invariant exists* (vk : szle 512) (s : chest2 bf16 512 8).
      k |-> vk ** wt |-> s **
      pure (forall (a : natlt 512) (b : natlt 8). a < SZ.v vk ==> acc2 s a b == acc2 ws a b)
    decreases (512 - SZ.v !k)
  {
    let kv = !k;
    let mut n : szle 8 = 0sz;
    while (!n <^ 8sz)
      invariant exists* (vn : szle 8) (s : chest2 bf16 512 8).
        n |-> vn ** wt |-> s **
        pure (forall (a : natlt 512) (b : natlt 8).
          (a < SZ.v kv \/ (a == SZ.v kv /\ b < SZ.v vn)) ==> acc2 s a b == acc2 ws a b)
      decreases (8 - SZ.v !n)
    {
      let nv = !n;
      let v = Kuiper.Tensor.read w ((kv <: szlt 512), ((nv <: szlt 8), ())) #fw;
      Kuiper.Tensor.write wt ((kv <: szlt 512), ((nv <: szlt 8), ())) v;
      n := nv +^ 1sz;
    };
    k := kv +^ 1sz;
  };
  with s. assert wt |-> s;
  assert (pure (Kuiper.EMatrix.equal s ws));
  rewrite (wt |-> s) as (wt |-> ws);
}

(* Per-thread MMA loop of kernel B. *)
inline_for_extraction noextract
instance c512 : concrete_sz 512 = { x = 512sz; }

inline_for_extraction noextract
fn mma_thread
  (x : array2 bf16 (l2_row_major 256 512))
  (wt : array2 bf16 (l2_col_major 512 8))
  (sc : array2 float (l2_row_major 256 8))
  (#fx #fw : perm) (#xs : chest2 bf16 256 512) (#ws : chest2 bf16 512 8)
  (#s0 : chest2 float 256 8)
  preserves gpu ** x |-> Frac fx xs ** wt |-> Frac fw ws
  requires sc |-> Frac (1.0R /. 32) s0
  ensures sc |-> Frac (1.0R /. 32) (scores_after s0 xs ws 16)
{
  tensor_pts_to_ref x;
  tensor_pts_to_ref wt;
  tensor_pts_to_ref sc;
  let mut ti : szle 16 = 0sz;
  while (!ti <^ 16sz)
    invariant exists* (vti : szle 16).
      ti |-> vti ** sc |-> Frac (1.0R /. 32) (scores_after s0 xs ws (SZ.v vti))
    decreases (16 - SZ.v !ti)
  {
    let t = !ti;
    let fr = alloc_fragment ();
    fill fr zero;
    rewrite (fr |-> const (16 @| 8 @| INil) (zero #float)) as (fr |-> ntile xs ws (SZ.v t) 0);
    let mut kb : szle 32 = 0sz;
    while (!kb <^ 32sz)
      invariant exists* (vkb : szle 32).
        kb |-> vkb ** fr |-> ntile xs ws (SZ.v t) (SZ.v vkb)
      decreases (32 - SZ.v !kb)
    {
      let k = !kb;
      let a_tile = array2_extract_tile_ro' x 16 16 (SZ.v t) (SZ.v k);
      let b_tile = array2_extract_tile_ro' wt 16 8 (SZ.v k) 0;
      mma_sync a_tile b_tile fr;
      with eta. assert (a_tile |-> Frac fx eta);
      Pulse.Lib.Trade.elim_trade (a_tile |-> Frac fx eta) (x |-> Frac fx xs);
      with etb. assert (b_tile |-> Frac fw etb);
      Pulse.Lib.Trade.elim_trade (b_tile |-> Frac fw etb) (wt |-> Frac fw ws);
      rewrite (fr |-> emma (ntile xs ws (SZ.v t) (SZ.v k))
                         (ematrix_subtile xs 16 16 (SZ.v t) (SZ.v k))
                         (ematrix_subtile ws 16 8 (SZ.v k) 0))
           as (fr |-> ntile xs ws (SZ.v t) (SZ.v k + 1));
      kb := k +^ 1sz;
    };
    let c_tile = array2_extract_tile_st sc 16 8 (SZ.v t) 0;
    store fr c_tile;
    Pulse.Lib.Forall.elim_forall (ntile xs ws (SZ.v t) 32);
    Pulse.Lib.Trade.elim_trade _ _;
    rewrite (sc |-> Frac (1.0R /. 32)
               (update_tile (scores_after s0 xs ws (SZ.v t)) 16 8 (SZ.v t) 0 (ntile xs ws (SZ.v t) 32)))
         as (sc |-> Frac (1.0R /. 32) (scores_after s0 xs ws (SZ.v t + 1)));
    drop_ (fr |-> ntile xs ws (SZ.v t) 32);
    ti := t +^ 1sz;
  };
}

(* Kernel descriptors and host sequence. *)
let kpre_b
  (x : array2 bf16 (l2_row_major 256 512))
  (wt : array2 bf16 (l2_col_major 512 8))
  (sc : array2 float (l2_row_major 256 8))
  (fx : perm) (xs : chest2 bf16 256 512) (ws : chest2 bf16 512 8)
  (s0 : chest2 float 256 8) (tid : natlt 32) : slprop
  = x |-> Frac (fx /. 32) xs ** wt |-> Frac (1.0R /. 32) ws ** sc |-> Frac (1.0R /. 32) s0

let kpost_b
  (x : array2 bf16 (l2_row_major 256 512))
  (wt : array2 bf16 (l2_col_major 512 8))
  (sc : array2 float (l2_row_major 256 8))
  (fx : perm) (xs : chest2 bf16 256 512) (ws : chest2 bf16 512 8)
  (s0 : chest2 float 256 8) (tid : natlt 32) : slprop
  = x |-> Frac (fx /. 32) xs ** wt |-> Frac (1.0R /. 32) ws ** sc |-> Frac (1.0R /. 32) (scores_after s0 xs ws 16)

ghost
fn setup_b
  (x : array2 bf16 (l2_row_major 256 512))
  (wt : array2 bf16 (l2_col_major 512 8))
  (sc : array2 float (l2_row_major 256 8))
  (fx : perm) (xs : chest2 bf16 256 512) (ws : chest2 bf16 512 8)
  (s0 : chest2 float 256 8)
  ()
  norewrite
  requires x |-> Frac fx xs ** wt |-> ws ** sc |-> s0
  ensures (forall+ (tid : natlt 32). kpre_b x wt sc fx xs ws s0 tid) ** emp
{
  tensor_share_n x 32;
  tensor_share_n wt 32;
  tensor_share_n sc 32;
  forevery_zip3 #(natlt 32)
    (fun _ -> x |-> Frac (fx /. 32) xs)
    (fun _ -> wt |-> Frac (1.0R /. 32) ws)
    (fun _ -> sc |-> Frac (1.0R /. 32) s0);
  forevery_ext #(natlt 32)
    (fun _ -> x |-> Frac (fx /. 32) xs ** wt |-> Frac (1.0R /. 32) ws ** sc |-> Frac (1.0R /. 32) s0)
    (fun tid -> kpre_b x wt sc fx xs ws s0 tid);
}

ghost
fn teardown_b
  (x : array2 bf16 (l2_row_major 256 512))
  (wt : array2 bf16 (l2_col_major 512 8))
  (sc : array2 float (l2_row_major 256 8))
  (fx : perm) (xs : chest2 bf16 256 512) (ws : chest2 bf16 512 8)
  (s0 : chest2 float 256 8)
  ()
  norewrite
  requires (forall+ (tid : natlt 32). kpost_b x wt sc fx xs ws s0 tid) ** emp
  ensures x |-> Frac fx xs ** wt |-> ws ** sc |-> scores_after s0 xs ws 16
{
  forevery_ext #(natlt 32)
    (fun tid -> kpost_b x wt sc fx xs ws s0 tid)
    (fun _ -> x |-> Frac (fx /. 32) xs ** wt |-> Frac (1.0R /. 32) ws **
              sc |-> Frac (1.0R /. 32) (scores_after s0 xs ws 16));
  forevery_unzip3 #(natlt 32) _ _ _;
  tensor_gather_n x 32;
  tensor_gather_n wt 32;
  tensor_gather_n sc 32;
}

inline_for_extraction noextract
fn kf_b
  (x : array2 bf16 (l2_row_major 256 512))
  (wt : array2 bf16 (l2_col_major 512 8))
  (sc : array2 float (l2_row_major 256 8))
  (fx : perm) (xs : chest2 bf16 256 512) (ws : chest2 bf16 512 8)
  (s0 : chest2 float 256 8)
  (tid : szlt 32)
  ()
  norewrite
  requires gpu ** kpre_b x wt sc fx xs ws s0 tid ** thread_id 32sz tid
  ensures gpu ** kpost_b x wt sc fx xs ws s0 tid ** thread_id 32sz tid
{
  unfold kpre_b x wt sc fx xs ws s0 tid;
  mma_thread x wt sc;
  fold kpost_b x wt sc fx xs ws s0 tid;
}

inline_for_extraction noextract
let kdesc_b
  (x : array2 bf16 (l2_row_major 256 512) { is_global x })
  (wt : array2 bf16 (l2_col_major 512 8) { is_global wt })
  (sc : array2 float (l2_row_major 256 8) { is_global sc })
  (fx : perm) (xs : chest2 bf16 256 512) (ws : chest2 bf16 512 8)
  (s0 : chest2 float 256 8)
  : kernel_desc_1_n
      (x |-> Frac fx xs ** wt |-> ws ** sc |-> s0)
      (x |-> Frac fx xs ** wt |-> ws ** sc |-> scores_after s0 xs ws 16)
  = {
    nthr = 32sz;
    frame = emp;
    kpre = kpre_b x wt sc fx xs ws s0;
    kpost = kpost_b x wt sc fx xs ws s0;
    block_setup = setup_b x wt sc fx xs ws s0;
    block_teardown = teardown_b x wt sc fx xs ws s0;
    f = kf_b x wt sc fx xs ws s0;
    full_pre_sendable = solve;
    full_post_sendable = solve;
    kpre_sendable = solve;
    kpost_sendable = solve;
  }

let init_ok (cn : chest1 i32 8) (ix : chest1 i64 256) (pr : chest1 f32 256) (sv : chest1 i32 2048) : prop =
  (forall (j : natlt 8). acc cn (j, ()) == 0l) /\
  (forall (j : natlt 256). acc ix (j, ()) == (-1L)) /\
  (forall (j : natlt 256). acc pr (j, ()) == zero #f32) /\
  (forall (j : natlt 2048). acc sv (j, ()) == 0l)

inline_for_extraction noextract
fn kf_a
  (w : array2 bf16 (l2_row_major 512 8))
  (wt : array2 bf16 (l2_col_major 512 8))
  (cn : array1 i32 (l1_forward 8))
  (ix : array1 i64 (l1_forward 256))
  (pr : array1 f32 (l1_forward 256))
  (sv : array1 i32 (l1_forward 2048))
  (#fw : perm) (#ws : chest2 bf16 512 8) (#wt0 : chest2 bf16 512 8)
  (#cn0 : chest1 i32 8) (#ix0 : chest1 i64 256) (#pr0 : chest1 f32 256) (#sv0 : chest1 i32 2048)
  preserves gpu ** w |-> Frac fw ws
  requires wt |-> wt0 ** cn |-> cn0 ** ix |-> ix0 ** pr |-> pr0 ** sv |-> sv0
  ensures wt |-> ws ** (exists* c i p s. cn |-> c ** ix |-> i ** pr |-> p ** sv |-> s ** pure (init_ok c i p s))
{
  transpose_w w wt;
  fill1 256sz ix (-1L);
  fill1 256sz pr (zero #f32);
  fill1 2048sz sv 0l;
  fill1 8sz cn 0l;
}

inline_for_extraction noextract
fn run
  (x : array2 bf16 (l2_row_major 256 512) { is_global x })
  (w : array2 bf16 (l2_row_major 512 8) { is_global w })
  (sc : array2 float (l2_row_major 256 8) { is_global sc })
  (cn : array1 i32 (l1_forward 8) { is_global cn })
  (ix : array1 i64 (l1_forward 256) { is_global ix })
  (pr : array1 f32 (l1_forward 256) { is_global pr })
  (sv : array1 i32 (l1_forward 2048) { is_global sv })
  (#fx #fw : perm) (#xs : chest2 bf16 256 512) (#ws : chest2 bf16 512 8)
  (#s0 : chest2 float 256 8)
  (#cn0 : chest1 i32 8) (#ix0 : chest1 i64 256) (#pr0 : chest1 f32 256) (#sv0 : chest1 i32 2048)
  preserves cpu ** on gpu_loc (x |-> Frac fx xs) ** on gpu_loc (w |-> Frac fw ws)
  requires on gpu_loc (sc |-> s0) ** on gpu_loc (cn |-> cn0) ** on gpu_loc (ix |-> ix0) **
           on gpu_loc (pr |-> pr0) ** on gpu_loc (sv |-> sv0)
  ensures on gpu_loc (sc |-> scores_after s0 xs ws 16) **
          (exists* c i p s. on gpu_loc (cn |-> c) ** on gpu_loc (ix |-> i) ** on gpu_loc (pr |-> p) **
             on gpu_loc (sv |-> s) ** pure (init_ok c i p s))
{
  let wt = alloc0 #bf16 4096sz (l2_col_major 512 8);
  with wt0. assert (on gpu_loc (wt |-> wt0));
  launch_kernel_1 (fun () -> kf_a w wt cn ix pr sv #fw #ws #wt0 #cn0 #ix0 #pr0 #sv0);
  launch_sync (kdesc_b x wt sc fx xs ws s0);
  free wt;
}

(* Final relation. *)
#push-options "--fuel 2 --ifuel 2 --z3rlimit 40"
let l1_imap (n : nat) (i : natlt n)
  : Lemma ((l1_forward n).imap.f (i, ()) == i)
  = ()
#pop-options

let to_seq_l1 (#et : Type) (#n : nat) (s : chest1 et n) (i : natlt n)
  : Lemma (S.length (to_seq (l1_forward n) s) == n /\
           S.index (to_seq (l1_forward n) s) i == acc s (i, ()))
  = l1_imap n i;
    Kuiper.Injection.lem_pat (l1_forward n).imap (i, ())

let relation_ok (xs : chest2 bf16 256 512) (ws : chest2 bf16 512 8) (s0 : chest2 float 256 8)
  (cn : chest1 i32 8) (ix : chest1 i64 256) (pr : chest1 f32 256) (sv : chest1 i32 2048)
  : Lemma (requires init_ok cn ix pr sv)
          (ensures optimized_projection_relation
             (to_seq (l2_row_major 256 512) xs) (to_seq (l2_row_major 512 8) ws)
             (to_seq (l2_row_major 256 8) (scores_after s0 xs ws 16))
             (to_seq (l1_forward 8) cn) (to_seq (l1_forward 256) ix)
             (to_seq (l1_forward 256) pr) (to_seq (l1_forward 2048) sv))
  = final_scores_ok s0 xs ws;
    to_seq_l1 cn 0; to_seq_l1 ix 0; to_seq_l1 pr 0; to_seq_l1 sv 0;
    introduce forall (e : natlt 8). I32.v (S.index (to_seq (l1_forward 8) cn) e) == 0
      with to_seq_l1 cn e;
    introduce forall (k : natlt 256). I64.v (S.index (to_seq (l1_forward 256) ix) k) == -1
      with to_seq_l1 ix k;
    introduce forall (k : natlt 256). S.index (to_seq (l1_forward 256) pr) k == zero
      with to_seq_l1 pr k;
    introduce forall (k : natlt 2048). I32.v (S.index (to_seq (l1_forward 2048) sv) k) == 0
      with to_seq_l1 sv k

#push-options "--z3rlimit 40"
fn kplay_run
  (x : array2 bf16 (l2_row_major 256 512) { is_global x })
  (w : array2 bf16 (l2_row_major 512 8) { is_global w })
  (sc : array2 float (l2_row_major 256 8) { is_global sc })
  (cn : array1 i32 (l1_forward 8) { is_global cn })
  (ix : array1 i64 (l1_forward 256) { is_global ix })
  (pr : array1 f32 (l1_forward 256) { is_global pr })
  (sv : array1 i32 (l1_forward 2048) { is_global sv })
  (#fx #fw : perm) (#xs : chest2 bf16 256 512) (#ws : chest2 bf16 512 8)
  (#s0 : chest2 float 256 8)
  (#cn0 : chest1 i32 8) (#ix0 : chest1 i64 256) (#pr0 : chest1 f32 256) (#sv0 : chest1 i32 2048)
  preserves cpu ** on gpu_loc (x |-> Frac fx xs) ** on gpu_loc (w |-> Frac fw ws)
  requires on gpu_loc (sc |-> s0) ** on gpu_loc (cn |-> cn0) ** on gpu_loc (ix |-> ix0) **
           on gpu_loc (pr |-> pr0) ** on gpu_loc (sv |-> sv0)
  ensures exists* (scs : chest2 float 256 8) (c : chest1 i32 8) (i : chest1 i64 256)
                  (p : chest1 f32 256) (s : chest1 i32 2048).
    on gpu_loc (sc |-> scs) ** on gpu_loc (cn |-> c) ** on gpu_loc (ix |-> i) **
    on gpu_loc (pr |-> p) ** on gpu_loc (sv |-> s) **
    pure (optimized_projection_relation
            (to_seq (l2_row_major 256 512) xs) (to_seq (l2_row_major 512 8) ws)
            (to_seq (l2_row_major 256 8) scs)
            (to_seq (l1_forward 8) c) (to_seq (l1_forward 256) i)
            (to_seq (l1_forward 256) p) (to_seq (l1_forward 2048) s))
{
  run x w sc cn ix pr sv;
  with (c : chest1 i32 8) (i : chest1 i64 256) (p : chest1 f32 256) (s : chest1 i32 2048).
    assert (on gpu_loc (cn |-> c) ** on gpu_loc (ix |-> i) ** on gpu_loc (pr |-> p) ** on gpu_loc (sv |-> s));
  relation_ok xs ws s0 c i p s;
  assert (pure (optimized_projection_relation
            (to_seq (l2_row_major 256 512) xs) (to_seq (l2_row_major 512 8) ws)
            (to_seq (l2_row_major 256 8) (scores_after s0 xs ws 16))
            (to_seq (l1_forward 8) c) (to_seq (l1_forward 256) i)
            (to_seq (l1_forward 256) p) (to_seq (l1_forward 2048) s)));
}
#pop-options
