module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open KPlay.SharedSpec
module S = FStar.Seq
module I64 = FStar.Int64
module F = Kuiper.Floating
module F32 = Kuiper.Float32
module SZ = FStar.SizeT

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

(*** Native order facts *)

let lt_irrefl (x:F32.t) : Lemma (~(lt x x)) =
  if lt x x then begin
    F32.lt_ordered x x;
    F.ieee_eq_refl x;
    F.lte_is_lt_or_eq x x;
    F.negate_lt_is_lte x x
  end

let nan_not_lt_l (x y:F32.t) : Lemma (requires F.NaN? (F.kind x)) (ensures ~(lt x y)) =
  if lt x y then F32.lt_ordered x y

let nan_not_lt_r (x y:F32.t) : Lemma (requires F.NaN? (F.kind y)) (ensures ~(lt x y)) =
  if lt x y then F32.lt_ordered x y

(* For non-NaN a, b: a >= b >= c implies a >= c (c arbitrary). *)
let not_lt_trans (a b c:F32.t)
  : Lemma (requires ~(F.NaN? (F.kind a)) /\ ~(F.NaN? (F.kind b)) /\ ~(lt a b) /\ ~(lt b c))
          (ensures ~(lt a c))
  = if lt a c then begin
      F32.lt_ordered a c;
      F.negate_lt_is_lte a b;
      F.lte_is_lt_or_eq b a;
      F.negate_lt_is_lte b c;
      F.lte_is_lt_or_eq c b;
      if lt b a then F32.lt_transitive b a c
      else if lt c b then F32.lt_transitive a c b
      else begin
        F.eq_spec b a;
        F.eq_spec c b;
        F.bit_eq_spec b a;
        F.bit_eq_spec c b;
        F.bit_eq_spec a c;
        F.eq_spec a c;
        F.lte_is_lt_or_eq c a;
        F.negate_lt_is_lte a c
      end
    end

(*** Candidates dominating an index *)

let cand_ok (sc:S.seq F32.t) (c:F32.t & int) : prop =
  snd c == -1 \/ (0 <= snd c /\ snd c < S.length sc /\ fst c == S.index sc (snd c))

let dom (sc:S.seq F32.t) (c:F32.t & int) (k:nat) : prop =
  k < S.length sc /\
  0 <= snd c /\ snd c < S.length sc /\ fst c == S.index sc (snd c) /\
  (F.NaN? (F.kind (S.index sc k)) ==> F.NaN? (F.kind (fst c))) /\
  (~(F.NaN? (F.kind (fst c))) ==> ~(lt (fst c) (S.index sc k)))

#push-options "--ifuel 1"
let pick_ok (sc:S.seq F32.t) (a b:F32.t & int)
  : Lemma (requires cand_ok sc a /\ cand_ok sc b) (ensures cand_ok sc (pick a b))
  = ()

let pick_dom_l (sc:S.seq F32.t) (a b:F32.t & int) (k:nat)
  : Lemma (requires dom sc a k /\ cand_ok sc b) (ensures dom sc (pick a b) k)
  = let (av, ai) = a in
    let (bv, bi) = b in
    if better bv bi av ai then begin
      if F.NaN? (F.kind (S.index sc k)) then begin
        nan_not_lt_l av bv
      end;
      if not (F.NaN? (F.kind bv)) then begin
        assert (lt av bv);
        F32.lt_ordered av bv;
        if lt bv (S.index sc k) then F32.lt_transitive av bv (S.index sc k)
      end
    end

let pick_dom_r (sc:S.seq F32.t) (a b:F32.t & int) (k:nat)
  : Lemma (requires cand_ok sc a /\ dom sc b k) (ensures dom sc (pick a b) k)
  = let (av, ai) = a in
    let (bv, bi) = b in
    if better bv bi av ai then ()
    else if not (F.NaN? (F.kind av)) then
      not_lt_trans av bv (S.index sc k)
#pop-options

(*** Tree unfolding *)

let stage (f:nat -> GTot (F32.t & int)) (d:nat) (k:nat) : GTot (F32.t & int) =
  if k < d then pick (f k) (f (k + d)) else f k

#push-options "--fuel 1"
let tree_unfold_raw (f:nat -> GTot (F32.t & int)) (d:pos) (i:nat)
  : Lemma (tree_at f d i == tree_at (fun (k:nat) -> if k < d then pick (f k) (f (k + d)) else f k) (d / 2) i)
  = ()

let tree_zero (f:nat -> GTot (F32.t & int)) (i:nat)
  : Lemma (tree_at f 0 i == f i)
  = ()
#pop-options

let rec tree_cong (f g:nat -> GTot (F32.t & int)) (n:nat) (d:nat) (i:nat)
  : Lemma (requires (forall (k:nat). k < n ==> f k == g k) /\ i < n /\ 2 * d <= n)
          (ensures tree_at f d i == tree_at g d i)
          (decreases d)
  = if d = 0 then begin
      tree_zero f i; tree_zero g i
    end else begin
      tree_unfold_raw f d i;
      tree_unfold_raw g d i;
      tree_cong (fun (k:nat) -> if k < d then pick (f k) (f (k + d)) else f k)
                (fun (k:nat) -> if k < d then pick (g k) (g (k + d)) else g k)
                n (d / 2) i
    end

let tree_unfold (f:nat -> GTot (F32.t & int)) (n:nat) (d:pos) (i:nat)
  : Lemma (requires i < n /\ 2 * d <= n)
          (ensures tree_at f d i == tree_at (stage f d) (d / 2) i)
  = tree_unfold_raw f d i;
    tree_cong (fun (k:nat) -> if k < d then pick (f k) (f (k + d)) else f k) (stage f d) n (d / 2) i

(*** Tree dominance *)

let rec tree_ok (sc:S.seq F32.t) (f:nat -> GTot (F32.t & int)) (n:nat)
  : Lemma (requires forall (j:nat). j < 2 * pow2 n ==> cand_ok sc (f j))
          (ensures cand_ok sc (tree_at f (pow2 n) 0))
          (decreases n)
  = let d = pow2 n in
    tree_unfold f (2 * d) d 0;
    let g = stage f d in
    let aux (j:nat) : Lemma (requires j < d) (ensures cand_ok sc (g j)) =
      pick_ok sc (f j) (f (j + d))
    in
    Classical.forall_intro (Classical.move_requires aux);
    if n = 0 then begin
      assert_spinoff (d == 1);
      tree_zero g 0;
      aux 0
    end else begin
      FStar.Math.Lemmas.pow2_double_mult (n - 1);
      tree_ok sc g (n - 1)
    end

let rec tree_dom (sc:S.seq F32.t) (f:nat -> GTot (F32.t & int)) (n:nat) (j0:nat) (k:nat)
  : Lemma (requires (forall (j:nat). j < 2 * pow2 n ==> cand_ok sc (f j)) /\
                    j0 < 2 * pow2 n /\ dom sc (f j0) k)
          (ensures dom sc (tree_at f (pow2 n) 0) k)
          (decreases n)
  = let d = pow2 n in
    tree_unfold f (2 * d) d 0;
    let g = stage f d in
    let aux (j:nat) : Lemma (requires j < d) (ensures cand_ok sc (g j)) =
      pick_ok sc (f j) (f (j + d))
    in
    Classical.forall_intro (Classical.move_requires aux);
    let j1 : nat = if j0 < d then j0 else j0 - d in
    if j0 < d then pick_dom_l sc (f j0) (f (j0 + d)) k
    else pick_dom_r sc (f j1) (f j0) k;
    if n = 0 then begin
      assert_spinoff (d == 1);
      tree_zero g 0
    end else begin
      FStar.Math.Lemmas.pow2_double_mult (n - 1);
      tree_dom sc g (n - 1) j1 k
    end

(*** Lane scans *)

#push-options "--fuel 1 --ifuel 1"
let rec lane_ok (sc:S.seq F32.t) (m stop:int) (acc:F32.t & int)
  : Lemma (requires cand_ok sc acc /\ 0 <= m /\ stop <= S.length sc)
          (ensures cand_ok sc (lane_scan sc m stop acc))
          (decreases (if m < stop then stop - m else 0))
  = if m >= stop then ()
    else begin
      pick_ok sc acc (f32_at sc m, m);
      lane_ok sc (m + 8) stop (pick acc (f32_at sc m, m))
    end

let rec lane_keep (sc:S.seq F32.t) (m stop:int) (acc:F32.t & int) (k:nat)
  : Lemma (requires dom sc acc k /\ 0 <= m /\ stop <= S.length sc)
          (ensures dom sc (lane_scan sc m stop acc) k)
          (decreases (if m < stop then stop - m else 0))
  = if m >= stop then ()
    else begin
      pick_dom_l sc acc (f32_at sc m, m) k;
      lane_keep sc (m + 8) stop (pick acc (f32_at sc m, m)) k
    end

let lt_dom_self (sc:S.seq F32.t) (k:nat)
  : Lemma (requires k < S.length sc) (ensures dom sc (S.index sc k, k) k)
  = lt_irrefl (S.index sc k)

let rec lane_dom (sc:S.seq F32.t) (m stop:int) (acc:F32.t & int) (k:nat)
  : Lemma (requires cand_ok sc acc /\ 0 <= m /\ m <= k /\ k < stop /\ stop <= S.length sc /\
                    (k - m) % 8 == 0)
          (ensures dom sc (lane_scan sc m stop acc) k)
          (decreases (if m < stop then stop - m else 0))
  = let acc' = pick acc (f32_at sc m, m) in
    if m = k then begin
      lt_dom_self sc k;
      pick_dom_r sc acc (f32_at sc m, m) k;
      lane_keep sc (m + 8) stop acc' k
    end else begin
      pick_ok sc acc (f32_at sc m, m);
      lane_dom sc (m + 8) stop acc' k
    end
#pop-options

let pow2_4 () : Lemma (pow2 4 == 16) =
  FStar.Math.Lemmas.pow2_plus 2 2;
  FStar.Math.Lemmas.pow2_plus 1 1

let pow2_8 () : Lemma (pow2 8 == 256) =
  pow2_4 ();
  FStar.Math.Lemmas.pow2_plus 4 4

let lane_cand_ok (sc:S.seq F32.t) (b w:nat)
  : Lemma (requires b < 512 /\ S.length sc == 248320) (ensures cand_ok sc (lane_cand sc b w))
  = if w < 8 then lane_ok sc (b * 485 + w) ((b + 1) * 485) (zero, -1)

let block_cand_ok (sc:S.seq F32.t) (b:nat)
  : Lemma (requires b < 512 /\ S.length sc == 248320) (ensures cand_ok sc (block_cand sc b))
  = let aux (w:nat) : Lemma (cand_ok sc (lane_cand sc b w)) = lane_cand_ok sc b w in
    Classical.forall_intro aux;
    pow2_4 ();
    tree_ok sc (fun (w:nat) -> lane_cand sc b w) 4

let block_cand_dom (sc:S.seq F32.t) (k:nat)
  : Lemma (requires k < 248320 /\ S.length sc == 248320)
          (ensures dom sc (block_cand sc (k / 485)) k)
  = let b = k / 485 in
    let r = k % 485 in
    let w = r % 8 in
    let aux (w:nat) : Lemma (cand_ok sc (lane_cand sc b w)) = lane_cand_ok sc b w in
    Classical.forall_intro aux;
    lane_dom sc (b * 485 + w) ((b + 1) * 485) (zero, -1) k;
    pow2_4 ();
    tree_dom sc (fun (w:nat) -> lane_cand sc b w) 4 w k

let tree_argmax_dom (sc:S.seq F32.t) (k:nat)
  : Lemma (requires k < 248320 /\ S.length sc == 248320)
          (ensures dom sc (tree_at (fun (b:nat) -> block_cand sc b) 256 0) k)
  = let aux (b:nat) : Lemma (requires b < 512) (ensures cand_ok sc (block_cand sc b)) =
      block_cand_ok sc b in
    Classical.forall_intro (Classical.move_requires aux);
    block_cand_dom sc k;
    pow2_8 ();
    tree_dom sc (fun (b:nat) -> block_cand sc b) 8 (k / 485) k

let tree_argmax_allowed (sc:S.seq F32.t)
  : Lemma (requires S.length sc == 248320) (ensures allowed_choice sc (tree_argmax sc))
  = let c = tree_at (fun (b:nat) -> block_cand sc b) 256 0 in
    let aux (k:nat) : Lemma (requires k < 248320) (ensures dom sc c k) = tree_argmax_dom sc k in
    Classical.forall_intro (Classical.move_requires aux);
    aux 0

(*** Kernel state *)

inline_for_extraction noextract
instance sized_i64 : sized I64.t = { size = 8sz; default = 0L }

let cand_of (#n:nat) (vs:chest1 f32 n) (is:chest1 I64.t n) (k:nat) : GTot (F32.t & int) =
  if k < n then (acc1 vs k, I64.v (acc1 is k)) else (zero, -1)

inline_for_extraction noextract
let better_x (v:f32) (i:I64.t) (best:f32) (bi:I64.t) : bool =
  I64.gte i 0L && (I64.lt bi 0L || lt #f32 best v || (not (eq #f32 v v) && eq #f32 best best))

let better_x_ok (v:f32) (i:I64.t) (best:f32) (bi:I64.t)
  : Lemma (better_x v i best bi == better v (I64.v i) best (I64.v bi))
  = F.eq_spec v v; F.eq_spec best best;
    F.bit_eq_spec v v; F.bit_eq_spec best best

#push-options "--fuel 1 --ifuel 1"
let acc1_upd (#et:Type) (#n:nat) (s:chest1 et n) (i:natlt n) (v:et) (j:natlt n)
  : Lemma (acc1 (upd s (i, ()) v) j == (if j = i then v else acc1 s j))
          [SMTPat (acc1 (upd s (i, ()) v) j)]
  = ()
#pop-options

let stage_inv (#n:nat) (vs1 vs:chest1 f32 n) (is1 is:chest1 I64.t n) (d i:nat) : prop =
  forall (k:nat). k < n ==>
    cand_of vs is k == (if k < i then stage (cand_of vs1 is1) d k else cand_of vs1 is1 k)

#push-options "--ifuel 1"
let stage_step (#n:nat) (vs1 vs:chest1 f32 n) (is1 is:chest1 I64.t n) (d i:nat)
               (c:bool) (b:f32) (bi:I64.t)
  : Lemma (requires i < d /\ 2 * d <= n /\ stage_inv vs1 vs is1 is d i /\
                    b == acc1 vs (i + d) /\ bi == acc1 is (i + d) /\
                    c == better b (I64.v bi) (acc1 vs i) (I64.v (acc1 is i)))
          (ensures stage_inv vs1 (if c then upd vs (i, ()) b else vs)
                             is1 (if c then upd is (i, ()) bi else is) d (i + 1))
  = assert (cand_of vs is i == cand_of vs1 is1 i);
    assert (cand_of vs is (i + d) == cand_of vs1 is1 (i + d));
    let vs' = if c then upd vs (i, ()) b else vs in
    let is' = if c then upd is (i, ()) bi else is in
    let aux (k:nat)
      : Lemma (requires k < n)
              (ensures cand_of vs' is' k ==
                       (if k < i + 1 then stage (cand_of vs1 is1) d k else cand_of vs1 is1 k))
      = assert (cand_of vs is k == (if k < i then stage (cand_of vs1 is1) d k else cand_of vs1 is1 k));
        if k = i then begin
          assert (stage (cand_of vs1 is1) d k == pick (cand_of vs1 is1 i) (cand_of vs1 is1 (i + d)));
          if c then assert (cand_of vs' is' k == (b, I64.v bi))
          else assert (cand_of vs' is' k == cand_of vs is k)
        end else begin
          assert (cand_of vs' is' k == cand_of vs is k)
        end
    in
    Classical.forall_intro (Classical.move_requires aux)
#pop-options

inline_for_extraction noextract
fn reduce_tree
  (n : SZ.t)
  (vv : array1 f32 (l1_forward (SZ.v n)) { is_global vv })
  (ii : array1 I64.t (l1_forward (SZ.v n)) { is_global ii })
  (d0 : SZ.t)
  (#vs0 : chest1 f32 (SZ.v n))
  (#is0 : chest1 I64.t (SZ.v n))
  requires
    vv |-> vs0 ** ii |-> is0 **
    pure (0 < SZ.v n /\ 2 * SZ.v d0 <= SZ.v n)
  ensures
    exists* (vs : chest1 f32 (SZ.v n)) (is : chest1 I64.t (SZ.v n)).
      vv |-> vs ** ii |-> is ** pure (cand_of vs is 0 == tree_at (cand_of vs0 is0) (SZ.v d0) 0)
{
  let mut d : SZ.t = d0;
  while (!d >^ 0sz)
    invariant live d
    invariant exists* (vs : chest1 f32 (SZ.v n)) (is : chest1 I64.t (SZ.v n)).
      vv |-> vs ** ii |-> is **
      pure (2 * SZ.v !d <= SZ.v n /\ tree_at (cand_of vs is) (SZ.v !d) 0 == tree_at (cand_of vs0 is0) (SZ.v d0) 0)
    decreases (SZ.v !d)
  {
    let dv = !d;
    with vs1 is1. assert (vv |-> vs1 ** ii |-> is1);
    let mut i : SZ.t = 0sz;
    while (!i <^ dv)
      invariant live i
      invariant exists* (vs : chest1 f32 (SZ.v n)) (is : chest1 I64.t (SZ.v n)).
        vv |-> vs ** ii |-> is **
        pure (SZ.v !i <= SZ.v dv /\
              (forall (k:nat). k < SZ.v n ==>
                 cand_of vs is k == (if k < SZ.v !i then stage (cand_of vs1 is1) (SZ.v dv) k
                                     else cand_of vs1 is1 k)))
      decreases (SZ.v dv - SZ.v !i)
    {
      let iv = !i;
      let jv = iv +^ dv;
      with vsc isc. assert (vv |-> vsc ** ii |-> isc);
      let a = Kuiper.Tensor.read vv ((iv <: szlt (SZ.v n)), ());
      let ai = Kuiper.Tensor.read ii ((iv <: szlt (SZ.v n)), ());
      let b = Kuiper.Tensor.read vv ((jv <: szlt (SZ.v n)), ());
      let bi = Kuiper.Tensor.read ii ((jv <: szlt (SZ.v n)), ());
      better_x_ok b bi a ai;
      stage_step vs1 vsc is1 isc (SZ.v dv) (SZ.v iv) (better_x b bi a ai) b bi;
      if (better_x b bi a ai) {
        Kuiper.Tensor.write vv ((iv <: szlt (SZ.v n)), ()) b;
        Kuiper.Tensor.write ii ((iv <: szlt (SZ.v n)), ()) bi;
      };
      i := iv +^ 1sz;
    };
    with vs2 is2. assert (vv |-> vs2 ** ii |-> is2);
    tree_unfold (cand_of vs1 is1) (SZ.v n) (SZ.v dv) 0;
    tree_cong (stage (cand_of vs1 is1) (SZ.v dv)) (cand_of vs2 is2) (SZ.v n) (SZ.v dv / 2) 0;
    d := SZ.div dv 2sz;
  };
  with vs3 is3. assert (vv |-> vs3 ** ii |-> is3);
  let dfin = !d;
  assert pure (SZ.v dfin == 0);
  assert pure (tree_at (cand_of vs3 is3) 0 0 == tree_at (cand_of vs0 is0) (SZ.v d0) 0);
  tree_zero (cand_of vs3 is3) 0;
  assert pure (cand_of vs3 is3 0 == tree_at (cand_of vs0 is0) (SZ.v d0) 0);
}

(*** Lane scan *)

#push-options "--fuel 1 --ifuel 1"
let lane_step (#n:nat) (ss:chest1 f32 n) (m stop:nat) (best:f32) (idx:I64.t) (s:f32) (mi:I64.t) (c:bool)
  : Lemma (requires m < stop /\ stop <= n /\ s == acc1 ss m /\ I64.v mi == m /\
                    c == better s m best (I64.v idx))
          (ensures lane_scan (chest1_to_seq ss) m stop (best, I64.v idx) ==
                   lane_scan (chest1_to_seq ss) (m + 8) stop
                     (if c then (s, I64.v mi) else (best, I64.v idx)))
  = assert (f32_at (chest1_to_seq ss) m == s)

let lane_init (sc:S.seq F32.t) (b w:nat) (start stop:nat)
  : Lemma (requires start == b * 485 + w /\ stop == (if w < 8 then (b + 1) * 485 else 0))
          (ensures lane_scan sc start stop (zero, I64.v (-1L)) == lane_cand sc b w)
  = ()

let lane_done (sc:S.seq F32.t) (m stop:nat) (acc:F32.t & int)
  : Lemma (requires m >= stop) (ensures lane_scan sc m stop acc == acc)
  = ()
#pop-options

let lanes_inv (sc:S.seq F32.t) (b:nat) (lvs:chest1 f32 32) (lis:chest1 I64.t 32) (w:nat) : prop =
  forall (j:nat). j < w ==> cand_of lvs lis j == lane_cand sc b j

let blocks_inv (sc:S.seq F32.t) (bvs:chest1 f32 512) (bis:chest1 I64.t 512) (b:nat) : prop =
  forall (j:nat). j < b ==> cand_of bvs bis j == block_cand sc j

let lanes_extend (sc:S.seq F32.t) (b:nat) (lvs:chest1 f32 32) (lis:chest1 I64.t 32) (w:natlt 32)
                 (best:f32) (idx:I64.t)
  : Lemma (requires lanes_inv sc b lvs lis w /\ (best, I64.v idx) == lane_cand sc b w)
          (ensures lanes_inv sc b (upd lvs (w, ()) best) (upd lis (w, ()) idx) (w + 1))
  = ()

let blocks_extend (sc:S.seq F32.t) (bvs:chest1 f32 512) (bis:chest1 I64.t 512) (b:natlt 512)
                  (lvs:chest1 f32 32) (lis:chest1 I64.t 32) (lvs0:chest1 f32 32) (lis0:chest1 I64.t 32)
  : Lemma (requires blocks_inv sc bvs bis b /\ lanes_inv sc b lvs0 lis0 32 /\
                    cand_of lvs lis 0 == tree_at (cand_of lvs0 lis0) 16 0)
          (ensures blocks_inv sc (upd bvs (b, ()) (acc1 lvs 0)) (upd bis (b, ()) (acc1 lis 0)) (b + 1))
  = tree_cong (cand_of lvs0 lis0) (fun (w:nat) -> lane_cand sc b w) 32 16 0

inline_for_extraction noextract
let to_i64 (x:SZ.t) : r:I64.t { I64.v r == SZ.v x } =
  FStar.Int.Cast.uint32_to_int64 (Kuiper.SizeT.sizet_to_u32 x)

inline_for_extraction noextract
fn scan_lane
  (scores : array1 f32 (l1_forward 248320) { is_global scores })
  (lv : array1 f32 (l1_forward 32) { is_global lv })
  (li : array1 I64.t (l1_forward 32) { is_global li })
  (bv : SZ.t { SZ.v bv < 512 })
  (wv : SZ.t { SZ.v wv < 32 })
  (#sp : perm)
  (#ss : chest1 f32 248320)
  (#lvs : chest1 f32 32)
  (#lis : chest1 I64.t 32)
  preserves
    scores |-> Frac sp ss
  requires
    lv |-> lvs ** li |-> lis ** pure (lanes_inv (chest1_to_seq ss) (SZ.v bv) lvs lis (SZ.v wv))
  ensures
    exists* (lvs' : chest1 f32 32) (lis' : chest1 I64.t 32).
      lv |-> lvs' ** li |-> lis' ** pure (lanes_inv (chest1_to_seq ss) (SZ.v bv) lvs' lis' (SZ.v wv + 1))
{
  let start = bv *^ 485sz +^ wv;
  let stop = (if wv <^ 8sz then (bv +^ 1sz) *^ 485sz else 0sz);
  lane_init (chest1_to_seq ss) (SZ.v bv) (SZ.v wv) (SZ.v start) (SZ.v stop);
  let mut m : SZ.t = start;
  let mut best : f32 = zero;
  let mut idx : I64.t = (-1L);
  while (!m <^ stop)
    invariant live m ** live best ** live idx
    invariant pure (
      lane_scan (chest1_to_seq ss) (SZ.v !m) (SZ.v stop) (!best, I64.v !idx) ==
      lane_cand (chest1_to_seq ss) (SZ.v bv) (SZ.v wv))
    decreases (if SZ.v !m < SZ.v stop then SZ.v stop - SZ.v !m else 0)
  {
    let mv = !m;
    let s = Kuiper.Tensor.read scores ((mv <: szlt 248320), ()) #sp;
    let mi = to_i64 mv;
    let bst = !best;
    let ix = !idx;
    better_x_ok s mi bst ix;
    lane_step ss (SZ.v mv) (SZ.v stop) bst ix s mi (better_x s mi bst ix);
    if (better_x s mi bst ix) {
      best := s;
      idx := mi;
    };
    m := mv +^ 8sz;
  };
  let bst = !best;
  let ix = !idx;
  lane_done (chest1_to_seq ss) (SZ.v !m) (SZ.v stop) (bst, I64.v ix);
  lanes_extend (chest1_to_seq ss) (SZ.v bv) lvs lis (SZ.v wv) bst ix;
  Kuiper.Tensor.write lv ((wv <: szlt 32), ()) bst;
  Kuiper.Tensor.write li ((wv <: szlt 32), ()) ix;
}

(*** Kernel *)

let blocks_done (sc:S.seq F32.t) (bvs:chest1 f32 512) (bis:chest1 I64.t 512)
                (bvs':chest1 f32 512) (bis':chest1 I64.t 512)
  : Lemma (requires blocks_inv sc bvs bis 512 /\
                    cand_of bvs' bis' 0 == tree_at (cand_of bvs bis) 256 0)
          (ensures I64.v (acc1 bis' 0) == tree_argmax sc)
  = tree_cong (cand_of bvs bis) (fun (b:nat) -> block_cand sc b) 512 256 0

inline_for_extraction noextract
let choose_tok (i d r x:I64.t) : I64.t = if I64.eq i d then r else x

let out_inv (#n:nat) (ts touts:chest1 I64.t n) (d:int) (r:I64.t) (i:nat) : prop =
  forall (j:nat). j < i /\ j < n ==> acc1 touts j == (if j = d then r else acc1 ts j)

let out_extend (#n:nat) (ts touts:chest1 I64.t n) (d:I64.t) (r:I64.t) (i:natlt n) (ii:I64.t)
  : Lemma (requires out_inv ts touts (I64.v d) r i /\ I64.v ii == i)
          (ensures out_inv ts (upd touts (i, ()) (choose_tok ii d r (acc1 ts i))) (I64.v d) r (i + 1))
  = ()

let kernel_post (sc:S.seq F32.t) (#n:nat) (ts:chest1 I64.t n) (d:int)
                (sel':chest1 I64.t 1) (tout':chest1 I64.t n) : prop =
  I64.v (acc1 sel' 0) == tree_argmax sc /\
  (forall (j:nat). j < n ==> acc1 tout' j == (if j = d then acc1 sel' 0 else acc1 ts j))

inline_for_extraction noextract
fn kf
  (scores : array1 f32 (l1_forward 248320) { is_global scores })
  (n : SZ.t)
  (tokens : array1 I64.t (l1_forward (SZ.v n)) { is_global tokens })
  (d : I64.t)
  (sel : array1 I64.t (l1_forward 1) { is_global sel })
  (tout : array1 I64.t (l1_forward (SZ.v n)) { is_global tout })
  (lv : array1 f32 (l1_forward 32) { is_global lv })
  (li : array1 I64.t (l1_forward 32) { is_global li })
  (bvv : array1 f32 (l1_forward 512) { is_global bvv })
  (bii : array1 I64.t (l1_forward 512) { is_global bii })
  (#sp : perm)
  (#ss : chest1 f32 248320)
  (#tp : perm)
  (#ts : chest1 I64.t (SZ.v n))
  (#sel0 : chest1 I64.t 1)
  (#tout0 : chest1 I64.t (SZ.v n))
  (#lvs0 : chest1 f32 32)
  (#lis0 : chest1 I64.t 32)
  (#bvs0 : chest1 f32 512)
  (#bis0 : chest1 I64.t 512)
  preserves
    gpu **
    scores |-> Frac sp ss **
    tokens |-> Frac tp ts
  requires
    sel |-> sel0 ** tout |-> tout0 **
    lv |-> lvs0 ** li |-> lis0 ** bvv |-> bvs0 ** bii |-> bis0
  ensures
    exists* (sel' : chest1 I64.t 1) (tout' : chest1 I64.t (SZ.v n))
            (lvs : chest1 f32 32) (lis : chest1 I64.t 32)
            (bvs : chest1 f32 512) (bis : chest1 I64.t 512).
      sel |-> sel' ** tout |-> tout' **
      lv |-> lvs ** li |-> lis ** bvv |-> bvs ** bii |-> bis **
      pure (kernel_post (chest1_to_seq ss) ts (I64.v d) sel' tout')
{
  let mut b : SZ.t = 0sz;
  while (!b <^ 512sz)
    invariant live b
    invariant exists* (lvs : chest1 f32 32) (lis : chest1 I64.t 32)
                      (bvs : chest1 f32 512) (bis : chest1 I64.t 512).
      lv |-> lvs ** li |-> lis ** bvv |-> bvs ** bii |-> bis **
      pure (SZ.v !b <= 512 /\ blocks_inv (chest1_to_seq ss) bvs bis (SZ.v !b))
    decreases (512 - SZ.v !b)
  {
    let bv = !b;
    let mut w : SZ.t = 0sz;
    while (!w <^ 32sz)
      invariant live w
      invariant exists* (lvs : chest1 f32 32) (lis : chest1 I64.t 32).
        lv |-> lvs ** li |-> lis **
        pure (SZ.v !w <= 32 /\ lanes_inv (chest1_to_seq ss) (SZ.v bv) lvs lis (SZ.v !w))
      decreases (32 - SZ.v !w)
    {
      let wv = !w;
      scan_lane scores lv li bv wv;
      w := wv +^ 1sz;
    };
    with lvs1 lis1. assert (lv |-> lvs1 ** li |-> lis1);
    reduce_tree 32sz lv li 16sz;
    with lvs2 lis2. assert (lv |-> lvs2 ** li |-> lis2);
    let v0 = Kuiper.Tensor.read lv ((0sz <: szlt 32), ());
    let i0 = Kuiper.Tensor.read li ((0sz <: szlt 32), ());
    with bvs1 bis1. assert (bvv |-> bvs1 ** bii |-> bis1);
    blocks_extend (chest1_to_seq ss) bvs1 bis1 (SZ.v bv) lvs2 lis2 lvs1 lis1;
    Kuiper.Tensor.write bvv ((bv <: szlt 512), ()) v0;
    Kuiper.Tensor.write bii ((bv <: szlt 512), ()) i0;
    b := bv +^ 1sz;
  };
  with bvs1 bis1. assert (bvv |-> bvs1 ** bii |-> bis1);
  reduce_tree 512sz bvv bii 256sz;
  with bvs2 bis2. assert (bvv |-> bvs2 ** bii |-> bis2);
  blocks_done (chest1_to_seq ss) bvs1 bis1 bvs2 bis2;
  let r = Kuiper.Tensor.read bii ((0sz <: szlt 512), ());
  Kuiper.Tensor.write sel ((0sz <: szlt 1), ()) r;
  let mut i : SZ.t = 0sz;
  while (!i <^ n)
    invariant live i
    invariant exists* (touts : chest1 I64.t (SZ.v n)).
      tout |-> touts **
      pure (SZ.v !i <= SZ.v n /\ out_inv ts touts (I64.v d) r (SZ.v !i))
    decreases (SZ.v n - SZ.v !i)
  {
    let iv = !i;
    let x = Kuiper.Tensor.read tokens ((iv <: szlt (SZ.v n)), ()) #tp;
    let ii = to_i64 iv;
    with touts. assert (tout |-> touts);
    out_extend ts touts d r (SZ.v iv) ii;
    Kuiper.Tensor.write tout ((iv <: szlt (SZ.v n)), ()) (choose_tok ii d r x);
    i := iv +^ 1sz;
  };
}

(*** Host entry point *)

let final_rel (sc:S.seq F32.t) (#n:nat) (ts:chest1 I64.t n) (d:int)
              (sel':chest1 I64.t 1) (tout':chest1 I64.t n)
  : Lemma (requires S.length sc == 248320 /\ kernel_post sc ts d sel' tout')
          (ensures optimized_argmax_relation sc (chest1_to_seq ts) d (chest1_to_seq sel') (chest1_to_seq tout'))
  = tree_argmax_allowed sc

fn kplay_run
  (scores : array1 f32 (l1_forward 248320) { is_global scores })
  (n : SZ.t)
  (tokens : array1 I64.t (l1_forward (SZ.v n)) { is_global tokens })
  (d : I64.t)
  (sel : array1 I64.t (l1_forward 1) { is_global sel })
  (tout : array1 I64.t (l1_forward (SZ.v n)) { is_global tout })
  (#sp : perm)
  (#ss : chest1 f32 248320)
  (#tp : perm)
  (#ts : chest1 I64.t (SZ.v n))
  (#sel0 : chest1 I64.t 1)
  (#tout0 : chest1 I64.t (SZ.v n))
  preserves
    cpu **
    on gpu_loc (scores |-> Frac sp ss) **
    on gpu_loc (tokens |-> Frac tp ts)
  requires
    on gpu_loc (sel |-> sel0) **
    on gpu_loc (tout |-> tout0)
  ensures
    exists* (sel' : chest1 I64.t 1) (tout' : chest1 I64.t (SZ.v n)).
      on gpu_loc (sel |-> sel') **
      on gpu_loc (tout |-> tout') **
      pure (optimized_argmax_relation (chest1_to_seq ss) (chest1_to_seq ts) (I64.v d)
              (chest1_to_seq sel') (chest1_to_seq tout'))
{
  let lv = Kuiper.Tensor.alloc0 #f32 32sz (l1_forward 32);
  let li = Kuiper.Tensor.alloc0 #I64.t 32sz (l1_forward 32);
  let bvv = Kuiper.Tensor.alloc0 #f32 512sz (l1_forward 512);
  let bii = Kuiper.Tensor.alloc0 #I64.t 512sz (l1_forward 512);
  with lvs0. assert (on gpu_loc (lv |-> lvs0));
  with lis0. assert (on gpu_loc (li |-> lis0));
  with bvs0. assert (on gpu_loc (bvv |-> bvs0));
  with bis0. assert (on gpu_loc (bii |-> bis0));
  launch_kernel_1
    (fun () ->
      kf scores n tokens d sel tout lv li bvv bii
        #sp #ss #tp #ts #sel0 #tout0 #lvs0 #lis0 #bvs0 #bis0);
  Kuiper.Tensor.free lv;
  Kuiper.Tensor.free li;
  Kuiper.Tensor.free bvv;
  Kuiper.Tensor.free bii;
  with sel' tout'. assert (on gpu_loc (sel |-> sel') ** on gpu_loc (tout |-> tout'));
  final_rel (chest1_to_seq ss) ts (I64.v d) sel' tout';
  assert pure (optimized_argmax_relation (chest1_to_seq ss) (chest1_to_seq ts) (I64.v d)
                 (chest1_to_seq sel') (chest1_to_seq tout'));
}
