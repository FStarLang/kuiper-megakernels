module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg
open Kuiper.Tensor.Tiling
open Kuiper.EMatrix.Tiling
open Kuiper.Spec.GEMM
open Kuiper.TensorCore.MMA
open Kuiper.Float.Casts
open Kuiper.Float.Casts.Base { cast_f32_to_bf16 }
open Kuiper.Array2.Strided
open Kuiper.TradeHelpers
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module C = Kuiper.Chest

(* ------------------------------------------------------------------ *)
(* Ghost model of the warp kernel                                      *)

let nv : nat = 151936
let ntiles : nat = 9496

let wch = chest2 bf16 151936 1024
let pch = chest2 bf16 1024 8
let sch = chest2 float 151936 8

(* Accumulator of row tile r after s MMA steps (deterministic). *)
let rec tile_acc (sw : wch) (sp : pch) (r : natlt 9496) (s : nat {s <= 64})
  : Tot (chest2 float 16 8) (decreases s)
  = if s = 0 then const (16 @| 8 @| INil) (zero #float)
    else emma (tile_acc sw sp r (s - 1))
              (ematrix_subtile sw 16 16 r (s - 1))
              (ematrix_subtile sp 16 8 (s - 1) 0)

(* Scratch contents after the first r row tiles were stored. *)
let scr_after (sw : wch) (sp : pch) (sc0 : sch) (r : nat {r <= 9496}) : sch =
  mk2 (fun (i : natlt 151936) (j : natlt 8) ->
    if i / 16 < r then acc2 (tile_acc sw sp (i / 16) 64) (i % 16) j
    else acc2 sc0 i j)

let scr_after_step (sw : wch) (sp : pch) (sc0 : sch) (r : natlt 9496)
  : Lemma (update_tile (scr_after sw sp sc0 r) 16 8 r 0 (tile_acc sw sp r 64)
           == scr_after sw sp sc0 (r + 1))
  = assert (C.equal (update_tile (scr_after sw sp sc0 r) 16 8 r 0 (tile_acc sw sp r 64))
                    (scr_after sw sp sc0 (r + 1)))

(* ------------------------------------------------------------------ *)
(* Real model                                                          *)

let rW (we : S.seq real) : chest2 real 151936 1024 =
  mk2 (fun (i : natlt 151936) (j : natlt 1024) -> at we (i * 1024 + j))

let rP (rh : S.seq real) : chest2 real 1024 8 =
  mk2 (fun (k : natlt 1024) (n : natlt 8) ->
    if n < 7 then at (window rh (n * 1024) 1024) k else 0.0R)

let rec rtile (we rh : S.seq real) (r : natlt 9496) (s : nat {s <= 64})
  : Tot (chest2 real 16 8) (decreases s)
  = if s = 0 then const (16 @| 8 @| INil) 0.0R
    else matplus (rtile we rh r (s - 1))
                 (matmul (ematrix_subtile (rW we) 16 16 r (s - 1))
                         (ematrix_subtile (rP rh) 16 8 (s - 1) 0))

let tile_bound (rows trows tr i : nat)
  : Lemma (requires trows > 0 /\ tr < rows / trows /\ i < trows)
          (ensures tr * trows + i < rows)
  = FStar.Math.Lemmas.lemma_mult_le_right trows tr (rows / trows - 1);
    FStar.Math.Lemmas.multiply_fractions rows trows

let subtile_approx (#rows #cols : nat) (#et : Type0) {| scalar et, real_like et |}
  (m : chest2 et rows cols) (rm : chest2 real rows cols)
  (trows : pos {trows /? rows}) (tcols : pos {tcols /? cols})
  (tr : natlt (rows / trows)) (tc : natlt (cols / tcols))
  : Lemma (requires m %~ rm)
          (ensures ematrix_subtile m trows tcols tr tc %~ ematrix_subtile rm trows tcols tr tc)
  = let aux (x : abs (trows @| tcols @| INil))
      : Lemma (acc (ematrix_subtile m trows tcols tr tc) x %~
               acc (ematrix_subtile rm trows tcols tr tc) x)
      = let (i, (j, ())) = x in
        tile_bound rows trows tr i;
        tile_bound cols tcols tc j;
        assert (acc m (tr * trows + i, (tc * tcols + j, ())) %~
                acc rm (tr * trows + i, (tc * tcols + j, ())))
    in
    Classical.forall_intro aux

let rec tile_acc_approx (sw : wch) (sp : pch) (we rh : S.seq real)
  (r : natlt 9496) (s : nat {s <= 64})
  : Lemma (requires sw %~ rW we /\ sp %~ rP rh)
          (ensures tile_acc sw sp r s %~ rtile we rh r s)
          (decreases s)
  = if s = 0 then begin
      let a0' : squash (zero #float %~ 0.0R) = a0 in
      ()
    end else begin
      tile_acc_approx sw sp we rh r (s - 1);
      subtile_approx sw (rW we) 16 16 r (s - 1);
      subtile_approx sp (rP rh) 16 8 (s - 1) 0;
      emma_approx_lemma (tile_acc sw sp r (s - 1))
                        (ematrix_subtile sw 16 16 r (s - 1))
                        (ematrix_subtile sp 16 8 (s - 1) 0)
                        (rtile we rh r (s - 1))
                        (ematrix_subtile (rW we) 16 16 r (s - 1))
                        (ematrix_subtile (rP rh) 16 8 (s - 1) 0)
    end

(* One 16-wide K step equals the next 16 terms of sdot. *)
let rec step_sdot (we rh : S.seq real) (r : natlt 9496) (s : natlt 64)
  (i : natlt 16) (n : natlt 7) (q : nat {q <= 16})
  : Lemma (ensures
      sdot (16 * s + q) we ((16 * r + i) * 1024) 1 (window rh (n * 1024) 1024) 0 1 ==
      sdot (16 * s) we ((16 * r + i) * 1024) 1 (window rh (n * 1024) 1024) 0 1 +.
      __matmul_single (ematrix_subtile (rW we) 16 16 r s)
                      (ematrix_subtile (rP rh) 16 8 s 0) i n q)
      (decreases q)
  = if q = 0 then ()
    else begin
      step_sdot we rh r s i n (q - 1);
      matmul_single_lemma (ematrix_subtile (rW we) 16 16 r s)
                          (ematrix_subtile (rP rh) 16 8 s 0) i n q
    end

let rec rtile_value (we rh : S.seq real) (r : natlt 9496) (s : nat {s <= 64})
  (i : natlt 16) (n : natlt 7)
  : Lemma (ensures acc2 (rtile we rh r s) i n ==
                   sdot (16 * s) we ((16 * r + i) * 1024) 1 (window rh (n * 1024) 1024) 0 1)
          (decreases s)
  = if s = 0 then ()
    else begin
      rtile_value we rh r (s - 1) i n;
      step_sdot we rh r (s - 1) i n 16;
      lemma_matmul_index (ematrix_subtile (rW we) 16 16 r (s - 1))
                         (ematrix_subtile (rP rh) 16 8 (s - 1) 0) i n
    end

(* ------------------------------------------------------------------ *)
(* From buffer contents to base_projection_relation                    *)

let w_ok (s0 : S.seq bf16) (a2 : nat) (sw : wch) : prop =
  a2 + embed_elements <= S.length s0 /\
  (forall (i : natlt 151936) (j : natlt 1024).
     acc2 sw i j == S.index s0 (a2 + i * 1024 + j))

let p_ok (s1 : S.seq bf16) (sp : pch) : prop =
  S.length s1 == 7168 /\
  (forall (k : natlt 1024) (n : natlt 8).
     acc2 sp k n == (if n < 7 then S.index s1 (n * 1024 + k) else zero #bf16))

let o_ok (scf : sch) (so : S.seq bf16) : prop =
  S.length so == 7 * 151936 /\
  (forall (k : natlt (7 * 151936)).
     S.index so k == cast_f32_to_bf16 (acc2 scf (k % 151936) (k / 151936)))

let w_approx (s0 : S.seq bf16) (a2 : nat) (sw : wch) (rw : S.seq real)
  : Lemma (requires w_ok s0 a2 sw /\ s0 %~ rw)
          (ensures sw %~ rW (window rw a2 embed_elements))
  = let aux (x : abs (151936 @| 1024 @| INil))
      : Lemma (acc sw x %~ acc (rW (window rw a2 embed_elements)) x)
      = let (i, (j, ())) = x in
        assert (acc2 sw i j == S.index s0 (a2 + i * 1024 + j))
    in
    Classical.forall_intro aux

let p_approx (s1 : S.seq bf16) (sp : pch) (rh : S.seq real)
  : Lemma (requires p_ok s1 sp /\ s1 %~ rh)
          (ensures sp %~ rP rh)
  = let a0' : squash (zero #bf16 %~ 0.0R) = a0 in
    let aux (x : abs (1024 @| 8 @| INil))
      : Lemma (acc sp x %~ acc (rP rh) x)
      = let (k, (n, ())) = x in
        assert (acc2 sp k n == (if n < 7 then S.index s1 (n * 1024 + k) else zero #bf16))
    in
    Classical.forall_intro aux

let out_elem (sw : wch) (sp : pch) (sc0 : sch) (so : S.seq bf16)
  (rw rh : S.seq real) (a2 : nat) (t : natlt 7) (v : natlt 151936)
  : Lemma (requires o_ok (scr_after sw sp sc0 9496) so /\
                    sw %~ rW (window rw a2 embed_elements) /\ sp %~ rP rh)
          (ensures S.index so (t * 151936 + v) %~
                   S.index (logits_row vocab_size (window rw a2 embed_elements)
                                      (window rh (t * 1024) 1024)) v)
  = let we = window rw a2 embed_elements in
    let k = t * 151936 + v in
    FStar.Math.Lemmas.lemma_mod_plus v t 151936;
    FStar.Math.Lemmas.lemma_div_plus v t 151936;
    assert (k % 151936 == v /\ k / 151936 == t);
    let r : natlt 9496 = v / 16 in
    tile_acc_approx sw sp we rh r 64;
    rtile_value we rh r 64 (v % 16) t;
    assert (acc2 (tile_acc sw sp r 64) (v % 16) t %~ acc2 (rtile we rh r 64) (v % 16) t);
    reveal_opaque (`%logits_row) (logits_row vocab_size we (window rh (t * 1024) 1024));
    assert (S.index so k == cast_f32_to_bf16 (acc2 (tile_acc sw sp r 64) (v % 16) t))

let final_relation (s0 s1 : S.seq bf16) (a2 : nat) (sw : wch) (sp : pch) (sc0 : sch)
  (so : S.seq bf16)
  : Lemma (requires w_ok s0 a2 sw /\ p_ok s1 sp /\ o_ok (scr_after sw sp sc0 9496) so)
          (ensures base_projection_relation s0 s1 a2 so)
  = let aux (rw rh : S.seq real)
      : Lemma (requires s0 %~ rw /\ s1 %~ rh)
              (ensures forall (t : nat). t < 7 ==>
                 window so (t * vocab_size) vocab_size %~
                 logits_row vocab_size (window rw a2 embed_elements) (window rh (t * 1024) 1024))
      = w_approx s0 a2 sw rw;
        p_approx s1 sp rh;
        let row (t : nat)
          : Lemma (requires t < 7)
                  (ensures window so (t * vocab_size) vocab_size %~
                           logits_row vocab_size (window rw a2 embed_elements)
                                      (window rh (t * 1024) 1024))
          = assert (S.length so == 7 * 151936);
            FStar.Math.Lemmas.lemma_mult_le_right 151936 (t + 1) 7;
            assert (S.length (window so (t * vocab_size) vocab_size) == vocab_size);
            let el (v : nat)
              : Lemma ((v < vocab_size /\
                        S.length (window so (t * vocab_size) vocab_size) == vocab_size) ==>
                       (S.index (window so (t * vocab_size) vocab_size) v %~
                        S.index (logits_row vocab_size (window rw a2 embed_elements)
                                           (window rh (t * 1024) 1024)) v))
              = if v < vocab_size then out_elem sw sp sc0 so rw rh a2 t v
            in
            Classical.forall_intro el
        in
        Classical.forall_intro (Classical.move_requires row)
    in
    Classical.forall_intro_2 (fun rw rh -> Classical.move_requires (aux rw) rh)

(* ------------------------------------------------------------------ *)
(* Layout facts                                                        *)

let l1_imap (n : nat) (k : natlt n) : Lemma ((l1_forward n).imap.f (k, ()) == k) = ()

let from_seq_l1 (#et : Type) (n : nat) (s : S.seq et {S.length s == n}) (k : natlt n)
  : Lemma (acc (from_seq (l1_forward n) s) (k, ()) == S.index s k)
  = l1_imap n k

let to_seq_l1 (#et : Type) (n : nat) (m : chest (n @| INil) et) (k : natlt n)
  : Lemma (S.length (to_seq (l1_forward n) m) == n /\
           S.index (to_seq (l1_forward n) m) k == acc m (k, ()))
  = let l = l1_forward n in
    let i = Kuiper.Injection.inverse_f l.imap k in
    let (j, ()) = i in
    l1_imap n j

let up1 (#n : erased nat) (i : conc (n @| INil))
  : Lemma (up i == (SZ.v (fst (i <: szlt n & conc INil)), ()))
  = ()

let up2 (#m #n : erased nat) (i : conc (m @| n @| INil))
  : Lemma (up i == (SZ.v (fst (i <: szlt m & conc (n @| INil))),
                    (SZ.v (fst (snd (i <: szlt m & conc (n @| INil)) <: szlt n & conc INil)), ())))
  = ()

inline_for_extraction noextract
let c1_0 (#n : erased nat) (i : conc (n @| INil)) : szlt n =
  fst (i <: szlt n & conc INil)

inline_for_extraction noextract
let c2_0 (#m #n : erased nat) (i : conc (m @| n @| INil)) : szlt m =
  fst (i <: szlt m & conc (n @| INil))

inline_for_extraction noextract
let c2_1 (#m #n : erased nat) (i : conc (m @| n @| INil)) : szlt n =
  fst (snd (i <: szlt m & conc (n @| INil)) <: szlt n & conc INil)

inline_for_extraction noextract
let szV : (x : SZ.t {SZ.v x == 151936}) = SZ.uint32_to_sizet 151936ul
inline_for_extraction noextract
let szE : (x : SZ.t {SZ.v x == 155582464}) = SZ.uint32_to_sizet 155582464ul
inline_for_extraction noextract
let szS : (x : SZ.t {SZ.v x == 1215488}) = SZ.uint32_to_sizet 1215488ul
inline_for_extraction noextract
let szO : (x : SZ.t {SZ.v x == 1063552}) = SZ.uint32_to_sizet 1063552ul

let lW = l2_row_major 151936 1024
let lP = l2_col_major 1024 8
let lS = l2_row_major 151936 8

(* ------------------------------------------------------------------ *)
(* Element relations of the three map kernels                          *)

let vcopy (#n : nat) (e0 : chest (n @| INil) bf16) (a2 : nat)
  (x : abs (151936 @| 1024 @| INil)) (_old o : bf16) : prop =
  let i = fst x in
  let j = fst (snd x) in
  a2 + i * 1024 + j < n /\ o == acc e0 (a2 + i * 1024 + j, ())

let vpad (e1 : chest (7168 @| INil) bf16)
  (x : abs (1024 @| 8 @| INil)) (_old o : bf16) : prop =
  let k = fst x in
  let n = fst (snd x) in
  o == (if n < 7 then acc e1 (n * 1024 + k, ()) else zero #bf16)

let vout (esc : sch) (x : abs (1063552 @| INil)) (_old o : bf16) : prop =
  let k = fst x in
  o == cast_f32_to_bf16 (acc2 esc (k % 151936) (k / 151936))

unfold
let inframe (#n0 : nat)
  (t0 : tensor bf16 (l1_forward n0) {is_global t0})
  (t1 : tensor bf16 (l1_forward 7168) {is_global t1})
  (e0 : chest (n0 @| INil) bf16) (e1 : chest (7168 @| INil) bf16)
  (f0 f1 fr : perm) : slprop =
  tensor_pts_to t0 #(f0 *. fr) e0 ** tensor_pts_to t1 #(f1 *. fr) e1

instance inframe_shareable (#n0 : nat)
  (t0 : tensor bf16 (l1_forward n0) {is_global t0})
  (t1 : tensor bf16 (l1_forward 7168) {is_global t1})
  (e0 : chest (n0 @| INil) bf16) (e1 : chest (7168 @| INil) bf16)
  (f0 f1 : perm)
  : Kuiper.Shareable.shareable (inframe t0 t1 e0 e1 f0 f1) =
  Kuiper.Shareable.double_shareable
    (fun fr -> tensor_pts_to t0 #fr e0)
    (fun fr -> tensor_pts_to t1 #fr e1)
    f0 f1

let idx_bound (i : natlt 151936) (j : natlt 1024)
  : Lemma (i * 1024 + j < 155582464) = ()

inline_for_extraction noextract
fn fcopy (n0 : SZ.t)
  (t0 : tensor bf16 (l1_forward (SZ.v n0)) {is_global t0})
  (t1 : tensor bf16 (l1_forward 7168) {is_global t1})
  (e0 : chest (SZ.v n0 @| INil) bf16) (e1 : chest (7168 @| INil) bf16)
  (a2 : SZ.t {SZ.v a2 + embed_elements <= SZ.v n0})
  (#f0 #f1 : perm) (#fr : perm) (i : conc (151936 @| 1024 @| INil)) (x : bf16)
  norewrite
  preserves gpu ** inframe t0 t1 e0 e1 f0 f1 fr
  returns r : bf16
  ensures pure (vcopy e0 (SZ.v a2) (up i) x r)
{
  let ii = c2_0 i;
  let jj = c2_1 i;
  up2 i;
  idx_bound (SZ.v ii) (SZ.v jj);
  let k : szlt (SZ.v n0) = a2 +^ (ii *^ 1024sz +^ jj);
  let v = read t0 (cidx1 k);
  v
}

inline_for_extraction noextract
fn fpad (n0 : SZ.t)
  (t0 : tensor bf16 (l1_forward (SZ.v n0)) {is_global t0})
  (t1 : tensor bf16 (l1_forward 7168) {is_global t1})
  (e0 : chest (SZ.v n0 @| INil) bf16) (e1 : chest (7168 @| INil) bf16)
  (#f0 #f1 : perm) (#fr : perm) (i : conc (1024 @| 8 @| INil)) (x : bf16)
  norewrite
  preserves gpu ** inframe t0 t1 e0 e1 f0 f1 fr
  returns r : bf16
  ensures pure (vpad e1 (up i) x r)
{
  let kk = c2_0 i;
  let nn = c2_1 i;
  up2 i;
  if (nn <^ 7sz) {
    let q : szlt 7168 = nn *^ 1024sz +^ kk;
    let v = read t1 (cidx1 q);
    v
  } else {
    zero #bf16
  }
}

inline_for_extraction noextract
fn fout (sc : tensor float lS {is_global sc}) (esc : sch)
  (#fr : perm) (i : conc (1063552 @| INil)) (x : bf16)
  norewrite
  preserves gpu ** tensor_pts_to sc #fr esc
  returns r : bf16
  ensures pure (vout esc (up i) x r)
{
  let k = c1_0 i;
  up1 i;
  let q : szlt 151936 = k %^ szV;
  let d : szlt 8 = k /^ szV;
  let v = read #_ #_ #_ #_ #(c_l2_row_major 151936 8sz) sc (cidx2 q d);
  cast_f32_to_bf16 v
}

(* ------------------------------------------------------------------ *)
(* The warp MMA kernel                                                 *)

inline_for_extraction noextract
instance c1024 : concrete_sz 1024 = { x = 1024sz }

inline_for_extraction noextract
let instA (ri : szlt 9496) (si : szlt 64)
  : strided_row_major (subtile_layout lW 16 16 (SZ.v ri) (SZ.v si))
  = strided_row_major_subtile lW #()
      #(strided_row_major_l2_row_major #151936 #1024 #() #c1024)
      16 16 (SZ.v ri) (SZ.v si) #_ #_ #(concrete_sz_sz ri) #(concrete_sz_sz si)

inline_for_extraction noextract
let instB (si : szlt 64)
  : strided_col_major (subtile_layout lP 16 8 (SZ.v si) 0)
  = strided_col_major_subtile lP #()
      #(strided_col_major_l2_col_major #1024 #8 #() #c1024)
      16 8 (SZ.v si) 0 #_ #_ #(concrete_sz_sz si) #_

inline_for_extraction noextract
let instC (ri : szlt 9496)
  : strided_row_major (subtile_layout lS 16 8 (SZ.v ri) 0)
  = strided_row_major_subtile lS #()
      #(strided_row_major_l2_row_major #151936 #8 #() #_)
      16 8 (SZ.v ri) 0 #_ #_ #(concrete_sz_sz ri) #_

let tile_acc_zero (sw : wch) (sp : pch) (r : natlt 9496)
  : Lemma (tile_acc sw sp r 0 == const (16 @| 8 @| INil) (zero #float)) = ()

let tile_acc_succ (sw : wch) (sp : pch) (r : natlt 9496) (s : natlt 64)
  : Lemma (tile_acc sw sp r (s + 1) ==
           emma (tile_acc sw sp r s) (ematrix_subtile sw 16 16 r s) (ematrix_subtile sp 16 8 s 0))
  = ()

let scr_after_zero (sw : wch) (sp : pch) (sc0 : sch)
  : Lemma (scr_after sw sp sc0 0 == sc0)
  = assert (C.equal (scr_after sw sp sc0 0) sc0)

inline_for_extraction noextract
fn warp_body
  (wb : tensor bf16 lW {is_global wb})
  (pd : tensor bf16 lP {is_global pd})
  (sc : tensor float lS {is_global sc})
  (sw : wch) (sp : pch) (sc0 : sch)
  preserves gpu
  preserves wb |-> Frac (1.0R /. 32) sw
  preserves pd |-> Frac (1.0R /. 32) sp
  requires sc |-> Frac (1.0R /. 32) sc0
  ensures sc |-> Frac (1.0R /. 32) (scr_after sw sp sc0 9496)
{
  scr_after_zero sw sp sc0;
  rewrite (sc |-> Frac (1.0R /. 32) sc0)
       as (sc |-> Frac (1.0R /. 32) (scr_after sw sp sc0 0));
  let mut r = 0sz;
  while (!r <^ 9496sz)
    invariant exists* (vr : szle 9496).
      (r |-> vr) **
      (sc |-> Frac (1.0R /. 32) (scr_after sw sp sc0 (SZ.v vr)))
    decreases (9496 - SZ.v !r)
  {
    let ri = !r;
    let fr = alloc_fragment ();
    fill fr (zero #float);
    tile_acc_zero sw sp (SZ.v ri);
    rewrite (fr |-> const (16 @| 8 @| INil) (zero #float))
         as (fr |-> tile_acc sw sp (SZ.v ri) 0);
    let mut s = 0sz;
    while (!s <^ 64sz)
      invariant exists* (vs : szle 64).
        (s |-> vs) **
        (fr |-> tile_acc sw sp (SZ.v ri) (SZ.v vs))
      decreases (64 - SZ.v !s)
    {
      let si = !s;
      let ta = array2_extract_tile_ro' wb 16 16 (SZ.v ri) (SZ.v si);
      let tb = array2_extract_tile_ro' pd 16 8 (SZ.v si) 0;
      mma_sync #_ #(instA ri si) ta #_ #(instB si) tb fr;
      ambig_trade_elim ();
      ambig_trade_elim ();
      tile_acc_succ sw sp (SZ.v ri) (SZ.v si);
      rewrite (fr |-> emma (tile_acc sw sp (SZ.v ri) (SZ.v si))
                           (ematrix_subtile sw 16 16 (SZ.v ri) (SZ.v si))
                           (ematrix_subtile sp 16 8 (SZ.v si) 0))
           as (fr |-> tile_acc sw sp (SZ.v ri) (SZ.v si + 1));
      s := si +^ 1sz;
    };
    with vs. assert (s |-> vs);
    rewrite (fr |-> tile_acc sw sp (SZ.v ri) (SZ.v vs))
         as (fr |-> tile_acc sw sp (SZ.v ri) 64);
    let tc = array2_extract_tile_st sc 16 8 (SZ.v ri) 0;
    store fr #_ #(instC ri) tc;
    Pulse.Lib.Forall.elim_forall (tile_acc sw sp (SZ.v ri) 64);
    ambig_trade_elim ();
    scr_after_step sw sp sc0 (SZ.v ri);
    rewrite (sc |-> Frac (1.0R /. 32)
               (update_tile (scr_after sw sp sc0 (SZ.v ri)) 16 8 (SZ.v ri) 0
                            (tile_acc sw sp (SZ.v ri) 64)))
         as (sc |-> Frac (1.0R /. 32) (scr_after sw sp sc0 (SZ.v ri + 1)));
    drop_ (fr |-> tile_acc sw sp (SZ.v ri) 64);
    r := ri +^ 1sz;
  };
  with vr. assert (r |-> vr);
  rewrite (sc |-> Frac (1.0R /. 32) (scr_after sw sp sc0 (SZ.v vr)))
       as (sc |-> Frac (1.0R /. 32) (scr_after sw sp sc0 9496));
}

unfold
let kpre_w (wb : tensor bf16 lW) (pd : tensor bf16 lP) (sc : tensor float lS)
  (sw : wch) (sp : pch) (sc0 : sch) (tid : nat) : slprop =
  tensor_pts_to wb #(1.0R /. 32) sw **
  tensor_pts_to pd #(1.0R /. 32) sp **
  tensor_pts_to sc #(1.0R /. 32) sc0

unfold
let kpost_w (wb : tensor bf16 lW) (pd : tensor bf16 lP) (sc : tensor float lS)
  (sw : wch) (sp : pch) (sc0 : sch) (tid : nat) : slprop =
  tensor_pts_to wb #(1.0R /. 32) sw **
  tensor_pts_to pd #(1.0R /. 32) sp **
  tensor_pts_to sc #(1.0R /. 32) (scr_after sw sp sc0 9496)

inline_for_extraction noextract
fn wthread
  (wb : tensor bf16 lW {is_global wb})
  (pd : tensor bf16 lP {is_global pd})
  (sc : tensor float lS {is_global sc})
  (sw : wch) (sp : pch) (sc0 : sch)
  (tid : szlt 32)
  ()
  norewrite
  preserves gpu
  requires kpre_w wb pd sc sw sp sc0 tid ** thread_id 32 tid
  ensures kpost_w wb pd sc sw sp sc0 tid ** thread_id 32 tid
{
  warp_body wb pd sc sw sp sc0;
}

ghost
fn wsetup
  (wb : tensor bf16 lW) (pd : tensor bf16 lP) (sc : tensor float lS)
  (sw : wch) (sp : pch) (sc0 : sch)
  ()
  norewrite
  requires tensor_pts_to wb sw ** tensor_pts_to pd sp ** tensor_pts_to sc sc0
  ensures (forall+ (tid : natlt 32). kpre_w wb pd sc sw sp sc0 tid) ** emp
{
  tensor_share_n wb 32;
  tensor_share_n pd 32;
  tensor_share_n sc 32;
  forevery_zip3 #(natlt 32)
    (fun _ -> tensor_pts_to wb #(1.0R /. 32) sw)
    (fun _ -> tensor_pts_to pd #(1.0R /. 32) sp)
    (fun _ -> tensor_pts_to sc #(1.0R /. 32) sc0);
}

ghost
fn wteardown
  (wb : tensor bf16 lW) (pd : tensor bf16 lP) (sc : tensor float lS)
  (sw : wch) (sp : pch) (sc0 : sch)
  ()
  norewrite
  requires (forall+ (tid : natlt 32). kpost_w wb pd sc sw sp sc0 tid) ** emp
  ensures tensor_pts_to wb sw ** tensor_pts_to pd sp **
          tensor_pts_to sc (scr_after sw sp sc0 9496)
{
  forevery_unzip3 #(natlt 32)
    (fun _ -> tensor_pts_to wb #(1.0R /. 32) sw)
    (fun _ -> tensor_pts_to pd #(1.0R /. 32) sp)
    (fun _ -> tensor_pts_to sc #(1.0R /. 32) (scr_after sw sp sc0 9496));
  tensor_gather_n wb 32;
  tensor_gather_n pd 32;
  tensor_gather_n sc 32;
}

inline_for_extraction noextract
let wkd
  (wb : tensor bf16 lW {is_global wb})
  (pd : tensor bf16 lP {is_global pd})
  (sc : tensor float lS {is_global sc})
  (sw : wch) (sp : pch) (sc0 : sch)
  : kernel_desc
      (tensor_pts_to wb sw ** tensor_pts_to pd sp ** tensor_pts_to sc sc0)
      (tensor_pts_to wb sw ** tensor_pts_to pd sp ** tensor_pts_to sc (scr_after sw sp sc0 9496))
  = {
      f = wthread wb pd sc sw sp sc0;
      nthr = 32sz;
      frame = emp;
      kpre = kpre_w wb pd sc sw sp sc0;
      kpost = kpost_w wb pd sc sw sp sc0;
      block_setup = wsetup wb pd sc sw sp sc0;
      block_teardown = wteardown wb pd sc sw sp sc0;
      kpost_sendable = solve;
      kpre_sendable = solve;
      full_post_sendable = solve;
      full_pre_sendable = solve;
    } <: kernel_desc_1_n _ _

(* ------------------------------------------------------------------ *)
(* Host side                                                           *)

inline_for_extraction noextract
instance cW : ctlayout lW = c_l2_row_major 151936 1024sz
inline_for_extraction noextract
instance cP : ctlayout lP = c_l2_col_major 1024sz 8
inline_for_extraction noextract
instance cS : ctlayout lS = c_l2_row_major 151936 8sz

let ch1 (n : nat) (s : S.seq bf16 {S.length s == n}) : chest (n @| INil) bf16 =
  from_seq (l1_forward n) s

let w_from_copy (n0 : nat) (s0 : S.seq bf16 {S.length s0 == n0}) (a2 : nat) (ew sw : wch)
  : Lemma (requires a2 + embed_elements <= n0 /\
                    chest_foralli (fun i x -> vcopy (ch1 n0 s0) a2 i (acc ew i) x) sw)
          (ensures w_ok s0 a2 sw)
  = let aux (i : natlt 151936) (j : natlt 1024)
      : Lemma (acc2 sw i j == S.index s0 (a2 + i * 1024 + j))
      = let x : abs (151936 @| 1024 @| INil) = (i, (j, ())) in
        assert (vcopy (ch1 n0 s0) a2 x (acc ew x) (acc sw x));
        from_seq_l1 n0 s0 (a2 + i * 1024 + j)
    in
    Classical.forall_intro_2 aux

let p_from_pad (s1 : S.seq bf16 {S.length s1 == 7168}) (ep sp : pch)
  : Lemma (requires chest_foralli (fun i x -> vpad (ch1 7168 s1) i (acc ep i) x) sp)
          (ensures p_ok s1 sp)
  = let aux (k : natlt 1024) (n : natlt 8)
      : Lemma (acc2 sp k n == (if n < 7 then S.index s1 (n * 1024 + k) else zero #bf16))
      = let x : abs (1024 @| 8 @| INil) = (k, (n, ())) in
        assert (vpad (ch1 7168 s1) x (acc ep x) (acc sp x));
        if n < 7 then from_seq_l1 7168 s1 (n * 1024 + k)
    in
    Classical.forall_intro_2 aux

let o_from_out (esc : sch) (so : S.seq bf16 {S.length so == 1063552})
  (eo : chest (1063552 @| INil) bf16)
  : Lemma (requires chest_foralli (fun i x -> vout esc i (acc (ch1 1063552 so) i) x) eo)
          (ensures o_ok esc (to_seq (l1_forward 1063552) eo))
  = let aux (k : natlt (7 * 151936))
      : Lemma (S.index (to_seq (l1_forward 1063552) eo) k ==
               cast_f32_to_bf16 (acc2 esc (k % 151936) (k / 151936)))
      = let x : abs (1063552 @| INil) = (k, ()) in
        assert (vout esc x (acc (ch1 1063552 so) x) (acc eo x));
        to_seq_l1 1063552 eo k
    in
    to_seq_l1 1063552 eo 0;
    Classical.forall_intro aux

let a2_of_i64 (x : I64.t)
  : Lemma (requires 0 <= I64.v x /\ I64.v x <= 1991901183)
          (ensures SZ.v (SZ.uint32_to_sizet (FStar.Int.Cast.int64_to_uint32 x)) == I64.v x)
  = assert_norm (pow2 32 == 4294967296);
    FStar.Math.Lemmas.small_mod (I64.v x) (pow2 32)

inline_for_extraction noextract
fn run_main (n0 : SZ.t) (in0 : larray bf16 (SZ.v n0)) (in1 : larray bf16 7168)
  (a2 : SZ.t {SZ.v a2 + embed_elements <= SZ.v n0})
  (out0 : larray bf16 (7 * 151936))
  (#f0 #f1 : perm) (#s0 #s1 #so : erased (S.seq bf16))
  preserves cpu
  preserves on gpu_loc (in0 |-> Frac f0 s0)
  preserves on gpu_loc (in1 |-> Frac f1 s1)
  requires on gpu_loc (out0 |-> so)
  requires pure (is_global in0 /\ is_global in1 /\ is_global out0)
  requires pure (S.length s0 == SZ.v n0 /\ S.length s1 == 7168 /\ S.length so == 7 * 151936)
  ensures exists* (so' : S.seq bf16).
    on gpu_loc (out0 |-> so') ** pure (base_projection_relation s0 s1 (SZ.v a2) so')
{
  let t0 : (t : tensor bf16 (l1_forward (SZ.v n0)) {is_global t}) =
    from_array (l1_forward (SZ.v n0)) in0;
  let t1 : (t : tensor bf16 (l1_forward 7168) {is_global t}) =
    from_array (l1_forward 7168) in1;
  let tout : (t : tensor bf16 (l1_forward 1063552) {is_global t}) =
    from_array (l1_forward 1063552) out0;
  Kuiper.Locs.map_loc gpu_loc
    #(in0 |-> Frac f0 s0)
    #(tensor_pts_to t0 #f0 (ch1 (SZ.v n0) s0))
    fn () {
      tensor_abs' (l1_forward (SZ.v n0)) in0;
      rewrite (tensor_pts_to (from_array (l1_forward (SZ.v n0)) in0) #f0 (from_seq (l1_forward (SZ.v n0)) s0))
           as (tensor_pts_to t0 #f0 (ch1 (SZ.v n0) s0))
    };
  Kuiper.Locs.map_loc gpu_loc
    #(in1 |-> Frac f1 s1)
    #(tensor_pts_to t1 #f1 (ch1 7168 s1))
    fn () {
      tensor_abs' (l1_forward 7168) in1;
      rewrite (tensor_pts_to (from_array (l1_forward 7168) in1) #f1 (from_seq (l1_forward 7168) s1))
           as (tensor_pts_to t1 #f1 (ch1 7168 s1))
    };
  Kuiper.Locs.map_loc gpu_loc
    #(out0 |-> so)
    #(tensor_pts_to tout (ch1 1063552 so))
    fn () {
      tensor_abs' (l1_forward 1063552) out0;
      rewrite (tensor_pts_to (from_array (l1_forward 1063552) out0) (from_seq (l1_forward 1063552) so))
           as (tensor_pts_to tout (ch1 1063552 so))
    };

  (* 1. weight window copy *)
  let wb = alloc0 #bf16 szE lW;
  with ew. assert (on gpu_loc (tensor_pts_to wb ew));
  launch_sync
    (Kuiper.Kernel.TMap.kmap (CCons szV (CCons 1024sz CNil))
      (inframe t0 t1 (ch1 (SZ.v n0) s0) (ch1 7168 s1) f0 f1)
      #(inframe_shareable t0 t1 (ch1 (SZ.v n0) s0) (ch1 7168 s1) f0 f1)
      (vcopy (ch1 (SZ.v n0) s0) (SZ.v a2))
      (fcopy n0 t0 t1 (ch1 (SZ.v n0) s0) (ch1 7168 s1) a2 #f0 #f1)
      szE wb #ew #_ #1.0R);
  with sw. assert (on gpu_loc (tensor_pts_to wb sw));
  w_from_copy (SZ.v n0) s0 (SZ.v a2) ew sw;

  (* 2. padded hidden matrix *)
  let pd = alloc0 #bf16 8192sz lP;
  with ep. assert (on gpu_loc (tensor_pts_to pd ep));
  launch_sync
    (Kuiper.Kernel.TMap.kmap (CCons 1024sz (CCons 8sz CNil))
      (inframe t0 t1 (ch1 (SZ.v n0) s0) (ch1 7168 s1) f0 f1)
      #(inframe_shareable t0 t1 (ch1 (SZ.v n0) s0) (ch1 7168 s1) f0 f1)
      (vpad (ch1 7168 s1))
      (fpad n0 t0 t1 (ch1 (SZ.v n0) s0) (ch1 7168 s1) #f0 #f1)
      8192sz pd #ep #_ #1.0R);
  with sp. assert (on gpu_loc (tensor_pts_to pd sp));
  p_from_pad s1 ep sp;

  (* 3. warp MMA kernel *)
  let sc = alloc0 #float szS lS;
  with sc0. assert (on gpu_loc (tensor_pts_to sc sc0));
  launch_sync (wkd wb pd sc sw sp sc0);

  (* 4. cast / transpose into the output *)
  launch_sync
    (Kuiper.Kernel.TMap.kmap (CCons szO CNil)
      (fun fr -> tensor_pts_to sc #fr (scr_after sw sp sc0 9496))
      #(tensor_pts_to_shareable sc (scr_after sw sp sc0 9496))
      (vout (scr_after sw sp sc0 9496))
      (fout sc (scr_after sw sp sc0 9496))
      szO tout #(ch1 1063552 so) #_ #1.0R);
  with eo. assert (on gpu_loc (tensor_pts_to tout eo));
  o_from_out (scr_after sw sp sc0 9496) so eo;
  final_relation s0 s1 (SZ.v a2) sw sp sc0 (to_seq (l1_forward 1063552) eo);

  (* back to flat buffers *)
  Kuiper.Locs.map_loc gpu_loc
    #(tensor_pts_to tout eo)
    #(out0 |-> to_seq (l1_forward 1063552) eo)
    fn () { tensor_concr tout; rewrite each (core tout) as out0 };
  Kuiper.Locs.map_loc gpu_loc
    #(tensor_pts_to t0 #f0 (ch1 (SZ.v n0) s0))
    #(in0 |-> Frac f0 s0)
    fn () { tensor_concr t0; rewrite each (core t0) as in0 };
  Kuiper.Locs.map_loc gpu_loc
    #(tensor_pts_to t1 #f1 (ch1 7168 s1))
    #(in1 |-> Frac f1 s1)
    fn () { tensor_concr t1; rewrite each (core t1) as in1 };
  free wb;
  free pd;
  free sc;
}

fn kplay_run (n0:SZ.t) (in0: larray bf16 (SZ.v n0)) (in1: larray bf16 7168) (in2: I64.t)
  (out0: larray bf16 (7*151936)) (#f0 #f1: perm) (#s0 #s1 #so: erased (S.seq bf16))
  preserves cpu
  preserves on gpu_loc (in0 |-> Frac f0 s0)
  preserves on gpu_loc (in1 |-> Frac f1 s1)
  requires on gpu_loc (out0 |-> so)
  requires pure (is_global in0 /\ is_global in1 /\ is_global out0)
  ensures exists* (so' : S.seq bf16).
    on gpu_loc (out0 |-> so') **
    pure (base_projection_relation s0 s1 (I64.v in2) so')
{
  Kuiper.Array.Derived.gpu_pts_to_ref_located in0;
  Kuiper.Array.Derived.gpu_pts_to_ref_located in1;
  Kuiper.Array.Derived.gpu_pts_to_ref_located out0;
  if (I64.gte in2 0L && I64.lte in2 1991901183L) {
    a2_of_i64 in2;
    let a2 = SZ.uint32_to_sizet (FStar.Int.Cast.int64_to_uint32 in2);
    if (szE <=^ n0) {
      if (a2 <=^ n0 -^ szE) {
        run_main n0 in0 in1 a2 out0;
      } else {
        ()
      }
    } else {
      ()
    }
  } else {
    ()
  }
}
