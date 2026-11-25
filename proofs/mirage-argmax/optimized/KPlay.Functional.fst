module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward }
open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I64 = FStar.Int64
module SZ = FStar.SizeT
module A = Pulse.Lib.Array
module GR = Pulse.Lib.GhostReference
module F = Kuiper.Floating.Base
module FL = Kuiper.Floating
module C = Kuiper.Float.Casts

#set-options "--fuel 0 --ifuel 0 --z3rlimit 20"


(* ------------------------------------------------------------------ *)
(* Bit reversal                                                        *)

#push-options "--fuel 1"
noextract
let rec rev_bits_bound (n:nat) (x:nat)
  : Lemma (ensures rev_bits n x < pow2 n) (decreases n)
  = if n = 0 then () else rev_bits_bound (n - 1) (x / 2)

noextract
let rec rev_bits_split (k:nat) (x:nat{x < pow2 k})
  : Lemma (ensures rev_bits (k + 1) x == 2 * rev_bits k x /\
                   rev_bits (k + 1) (x + pow2 k) == 2 * rev_bits k x + 1)
          (decreases k)
  = if k = 0 then ()
    else begin
      rev_bits_split (k - 1) (x / 2);
      assert ((x + pow2 k) % 2 == x % 2);
      assert ((x + pow2 k) / 2 == x / 2 + pow2 (k - 1));
      assert (pow2 k == 2 * pow2 (k - 1))
    end

noextract
let rec rev_bits_inj (n:nat) (x y:nat)
  : Lemma (requires x < pow2 n /\ y < pow2 n /\ rev_bits n x == rev_bits n y)
          (ensures x == y) (decreases n)
  = if n = 0 then ()
    else begin
      rev_bits_bound (n - 1) (x / 2);
      rev_bits_bound (n - 1) (y / 2);
      assert (pow2 n == 2 * pow2 (n - 1));
      rev_bits_inj (n - 1) (x / 2) (y / 2)
    end
#pop-options

(* ------------------------------------------------------------------ *)
(* Key structure                                                       *)

#push-options "--fuel 6"
noextract
let pow2_small () : Lemma (pow2 0 == 1 /\ pow2 1 == 2 /\ pow2 2 == 4 /\ pow2 3 == 8 /\ pow2 4 == 16 /\ pow2 5 == 32) = ()
#pop-options

noextract
let crank (c:nat) : nat = rev_bits 3 (c / 32) * 32 + rev_bits 5 (c % 32)
noextract
let lrank (t:nat) : nat = rev_bits 2 (t / 32) * 32 + rev_bits 5 (t % 32)

noextract
let crank_bound (c:nat{c < 256}) : Lemma (crank c < 256)
  = pow2_small (); rev_bits_bound 3 (c / 32); rev_bits_bound 5 (c % 32)

noextract
let lrank_bound (t:nat{t < 128}) : Lemma (lrank t < 128)
  = pow2_small (); rev_bits_bound 2 (t / 32); rev_bits_bound 5 (t % 32)

noextract
let crank_inj (c d:nat)
  : Lemma (requires c < 256 /\ d < 256 /\ crank c == crank d) (ensures c == d)
  = rev_bits_bound 5 (c % 32); rev_bits_bound 5 (d % 32);
    pow2_small ();
    rev_bits_inj 3 (c / 32) (d / 32);
    rev_bits_inj 5 (c % 32) (d % 32)

noextract
let lrank_inj (c d:nat)
  : Lemma (requires c < 128 /\ d < 128 /\ lrank c == lrank d) (ensures c == d)
  = rev_bits_bound 5 (c % 32); rev_bits_bound 5 (d % 32);
    pow2_small ();
    rev_bits_inj 2 (c / 32) (d / 32);
    rev_bits_inj 5 (c % 32) (d % 32)

noextract
let key_parts (j:nat)
  : Lemma (opt_key j == (crank (j / 1600) * 128 + lrank ((j % 1600) % 128)) * 13 + (j % 1600) / 128)
  = ()

noextract
let nV : nat = 151936

(* Division facts: 13 X + m in [13 Y, 13 Y + w), m < 13, w <= 13 ==> X = Y, m < w. *)
noextract
let div13 (x y m w:nat)
  : Lemma (requires m < 13 /\ w <= 13 /\ 13 * y <= 13 * x + m /\ 13 * x + m < 13 * y + w)
          (ensures x == y /\ m < w)
  = ()

noextract
let div128 (x y a b:nat)
  : Lemma (requires a < 128 /\ b < 128 /\ x * 128 + a == y * 128 + b)
          (ensures x == y /\ a == b)
  = ()

noextract
let key_parts_bound (j:nat{j < nV})
  : Lemma (j / 1600 < 95 /\ (j % 1600) / 128 < 13 /\ (j % 1600) % 128 < 128 /\
           crank (j / 1600) < 256 /\ lrank ((j % 1600) % 128) < 128)
  = crank_bound (j / 1600); lrank_bound ((j % 1600) % 128)

(* Lane interval: keys [K, K + w) of chunk c, lane t are exactly the strided
   elements c*1600 + t + 128 m with m < w. *)
noextract
let lane_base (c t:nat) : nat = (crank c * 128 + lrank t) * 13

noextract
let lane_member (c t w:nat) (j:nat)
  : Lemma (requires c < 256 /\ t < 128 /\ w <= 13 /\ j < nV /\
                    lane_base c t <= opt_key j /\ opt_key j < lane_base c t + w)
          (ensures j / 1600 == c /\ (j % 1600) % 128 == t /\ (j % 1600) / 128 < w /\
                   j == c * 1600 + t + 128 * ((j % 1600) / 128))
  = key_parts j; key_parts_bound j;
    crank_bound c; lrank_bound t;
    let x = crank (j / 1600) * 128 + lrank ((j % 1600) % 128) in
    let y = crank c * 128 + lrank t in
    div13 x y ((j % 1600) / 128) w;
    div128 (crank (j / 1600)) (crank c) (lrank ((j % 1600) % 128)) (lrank t);
    crank_inj (j / 1600) c;
    lrank_inj ((j % 1600) % 128) t

noextract
let lane_key (c t m:nat)
  : Lemma (requires t < 128 /\ t + 128 * m < 1600)
          (ensures (let j = c * 1600 + t + 128 * m in
                    j / 1600 == c /\ j % 1600 == t + 128 * m /\
                    opt_key j == lane_base c t + m))
  = let j = c * 1600 + t + 128 * m in
    FStar.Math.Lemmas.lemma_mod_plus (t + 128 * m) c 1600;
    FStar.Math.Lemmas.lemma_div_plus (t + 128 * m) c 1600;
    FStar.Math.Lemmas.lemma_mod_plus t m 128;
    FStar.Math.Lemmas.lemma_div_plus t m 128;
    FStar.Math.Lemmas.small_mod t 128;
    FStar.Math.Lemmas.small_div t 128;
    key_parts j

(* Chunk interval: keys [1664 crank c, +1664) are exactly chunk c. *)
noextract
let chunk_member (c:nat) (j:nat)
  : Lemma (requires c < 256 /\ j < nV /\
                    crank c * 1664 <= opt_key j /\ opt_key j < crank c * 1664 + 1664)
          (ensures j / 1600 == c)
  = key_parts j; key_parts_bound j;
    crank_bound c;
    let a = lrank ((j % 1600) % 128) * 13 + (j % 1600) / 128 in
    assert (opt_key j == crank (j / 1600) * 1664 + a);
    assert (a < 1664);
    assert (crank (j / 1600) == crank c);
    crank_inj (j / 1600) c

noextract
let key_range (j:nat{j < nV}) : Lemma (opt_key j < 425984)
  = key_parts j; key_parts_bound j

noextract
let key_inj (j k:nat)
  : Lemma (requires j < nV /\ k < nV /\ opt_key j == opt_key k) (ensures j == k)
  = key_parts j; key_parts k; key_parts_bound j; key_parts_bound k;
    let x = crank (j / 1600) * 128 + lrank ((j % 1600) % 128) in
    let y = crank (k / 1600) * 128 + lrank ((k % 1600) % 128) in
    assert (x == y);
    div128 (crank (j / 1600)) (crank (k / 1600)) (lrank ((j % 1600) % 128)) (lrank ((k % 1600) % 128));
    crank_inj (j / 1600) (k / 1600);
    lrank_inj ((j % 1600) % 128) ((k % 1600) % 128);
    FStar.Math.Lemmas.euclidean_division_definition j 1600;
    FStar.Math.Lemmas.euclidean_division_definition k 1600;
    FStar.Math.Lemmas.euclidean_division_definition (j % 1600) 128;
    FStar.Math.Lemmas.euclidean_division_definition (k % 1600) 128

(* ------------------------------------------------------------------ *)
(* Derived binary32 order facts                                        *)

noextract
let nanf (x:F32.t) : prop = F.NaN? (F.kind x)

noextract
let sem (p:prop) : GTot bool = FStar.IndefiniteDescription.strong_excluded_middle p

noextract
let eq_sym (x y:F32.t) : Lemma (requires F.eq x y) (ensures F.eq y x)
  = F.eq_spec x y; F.eq_spec y x; F.bit_eq_spec x y; F.bit_eq_spec y x

noextract
let eq_trans (x y z:F32.t) : Lemma (requires F.eq x y /\ F.eq y z) (ensures F.eq x z)
  = F.eq_spec x y; F.eq_spec y z; F.eq_spec x z;
    F.bit_eq_spec x y; F.bit_eq_spec y z; F.bit_eq_spec x z;
    if F.bit_eq x y then F.bits_spec x y else ();
    if F.bit_eq y z then F.bits_spec y z else ()

noextract
let eq_refl (x:F32.t) : Lemma (requires ~(nanf x)) (ensures F.eq x x /\ F.lte x x)
  = FL.ieee_eq_refl x; F.lte_is_lt_or_eq x x

noextract
let lte_cases (x y:F32.t)
  : Lemma (requires ~(nanf x) /\ ~(nanf y))
          (ensures (F.lte x y <==> (F.lt x y \/ F.eq x y)) /\ (F.lt x y <==> ~(F.lte y x)))
  = F.lte_is_lt_or_eq x y; F.negate_lt_is_lte x y

noextract
let eq_lt (x y z:F32.t)
  : Lemma (requires F.eq x y /\ F.lt y z) (ensures F.lt x z)
  = F.eq_spec x y; F32.lt_ordered y z;
    lte_cases x z; lte_cases z x; lte_cases x y; lte_cases y x; lte_cases y z; lte_cases z y;
    if F.lt z x then F32.lt_transitive y z x
    else if F.eq z x then (eq_trans y x z; eq_sym x y; eq_trans y x z)
    else ()

noextract
let lt_eq (x y z:F32.t)
  : Lemma (requires F.lt x y /\ F.eq y z) (ensures F.lt x z)
  = F.eq_spec y z; F32.lt_ordered x y;
    lte_cases x z; lte_cases z x; lte_cases x y; lte_cases y x; lte_cases y z; lte_cases z y;
    if F.lt z x then F32.lt_transitive z x y
    else if F.eq z x then (eq_trans y z x)
    else ()

noextract
let lte_lt (x y z:F32.t)
  : Lemma (requires ~(nanf x) /\ ~(nanf y) /\ F.lte x y /\ F.lt y z) (ensures F.lt x z)
  = lte_cases x y;
    if F.lt x y then F32.lt_transitive x y z else eq_lt x y z

noextract
let lte_lte (x y z:F32.t)
  : Lemma (requires ~(nanf x) /\ ~(nanf y) /\ ~(nanf z) /\ F.lte x y /\ F.lte y z) (ensures F.lte x z)
  = lte_cases x y; lte_cases y z; lte_cases x z;
    if F.lt y z then lte_lt x y z
    else if F.lt x y then lt_eq x y z
    else eq_trans x y z

(* ------------------------------------------------------------------ *)
(* Best candidate over a key interval                                  *)

noextract
let ws (s:S.seq BF.t) (j:int) : F32.t =
  if 0 <= j && j < S.length s then C.c_bf16_32.fcast (S.index s j) else F.zero

noextract
let in_iv (s:S.seq BF.t) (lo hi:int) (j:int) : prop =
  0 <= j /\ j < S.length s /\ lo <= opt_key j /\ opt_key j < hi

noextract
let best (s:S.seq BF.t) (lo hi:int) (g:int) : prop =
  (g < 0 ==> (forall (j:nat). ~(in_iv s lo hi j))) /\
  (g >= 0 ==> in_iv s lo hi g /\
    (nanf (ws s g) ==> (forall (j:nat). in_iv s lo hi j /\ j < g ==> ~(nanf (ws s j)))) /\
    (~(nanf (ws s g)) ==> (forall (j:nat). in_iv s lo hi j ==>
        ~(nanf (ws s j)) /\ F.lte (ws s j) (ws s g) /\
        (opt_key j < opt_key g ==> F.lt (ws s j) (ws s g)))))

noextract
let wins_g (s:S.seq BF.t) (ga gb:int) : prop =
  gb >= 0 /\
  (ga < 0 \/
   (nanf (ws s gb) /\ (~(nanf (ws s ga)) \/ gb < ga)) \/
   (~(nanf (ws s gb)) /\ ~(nanf (ws s ga)) /\ F.lt (ws s ga) (ws s gb)))

noextract
let best_empty (s:S.seq BF.t) (k:int) : Lemma (best s k k (-1)) = ()

noextract
let best_single (s:S.seq BF.t) (j:nat)
  : Lemma (requires j < S.length s /\ S.length s == nV)
          (ensures best s (opt_key j) (opt_key j + 1) j)
  = let aux (i:nat) : Lemma (in_iv s (opt_key j) (opt_key j + 1) i ==> i == j)
      = if i < nV && opt_key i = opt_key j then key_inj i j in
    Classical.forall_intro aux;
    if sem (nanf (ws s j)) then () else eq_refl (ws s j)

#push-options "--z3rlimit 80"
noextract
let best_merge (s:S.seq BF.t) (lo mid hi:int) (ga gb:int)
  : Lemma (requires lo <= mid /\ mid <= hi /\ best s lo mid ga /\ best s mid hi gb)
          (ensures best s lo hi (if sem (wins_g s ga gb) then gb else ga))
  = let w = sem (wins_g s ga gb) in
    let split (j:nat) : Lemma (in_iv s lo hi j <==> (in_iv s lo mid j \/ in_iv s mid hi j)) = () in
    Classical.forall_intro split;
    if gb < 0 then ()
    else if ga < 0 then ()
    else if sem (nanf (ws s gb)) then ()
    else if sem (nanf (ws s ga)) then ()
    else begin
      lte_cases (ws s ga) (ws s gb);
      if w then begin
        let aux (j:nat) : Lemma (in_iv s lo mid j ==>
            ~(nanf (ws s j)) /\ F.lte (ws s j) (ws s gb) /\ F.lt (ws s j) (ws s gb))
          = if sem (in_iv s lo mid j) then begin
              lte_lt (ws s j) (ws s ga) (ws s gb);
              lte_cases (ws s j) (ws s gb)
            end in
        Classical.forall_intro aux
      end else begin
        let aux (j:nat) : Lemma (in_iv s mid hi j ==>
            ~(nanf (ws s j)) /\ F.lte (ws s j) (ws s ga))
          = if sem (in_iv s mid hi j) then
              lte_lte (ws s j) (ws s gb) (ws s ga) in
        Classical.forall_intro aux
      end
    end
#pop-options

(* Full interval ==> optimized tree argmax over the BF16 row. *)
noextract
let best_final (s:S.seq BF.t) (g:int)
  : Lemma (requires S.length s == nV /\ best s 0 425984 g)
          (ensures tree_argmax s g)
  = let full (j:nat) : Lemma (j < nV ==> in_iv s 0 425984 j)
      = if j < nV then key_range j in
    Classical.forall_intro full;
    let kind (j:nat) : Lemma (j < nV ==> (nanf (ws s j) <==> is_nan (S.index s j)))
      = if j < nV then C.bf16_to_f32_kind (S.index s j) in
    Classical.forall_intro kind;
    assert (g >= 0);
    if sem (nanf (ws s g)) then begin
      assert (has_nan s)
    end else begin
      let ord (j:nat) : Lemma (j < nV ==>
          F.lte (S.index s j) (S.index s g) /\
          (opt_key j < opt_key g ==> F.lt (S.index s j) (S.index s g)))
        = if j < nV then begin
            C.bf16_to_f32_lte (S.index s j) (S.index s g);
            C.bf16_to_f32_lte (S.index s g) (S.index s j);
            lte_cases (ws s j) (ws s g);
            C.bf16_to_f32_kind (S.index s j);
            C.bf16_to_f32_kind (S.index s g);
            F.negate_lt_is_lte (S.index s j) (S.index s g)
          end in
      Classical.forall_intro ord;
      assert (~(has_nan s))
    end

(* ------------------------------------------------------------------ *)
(* Kernel index maps and the executable comparison                     *)

(* Global row index of a kernel index: chunk stage (c >= 0) holds local
   indices in chunk c; final stage (c < 0) holds chunk * 2^32 + local. *)
noextract
let gm (c:int) (i:int) : int =
  if i < 0 then -1
  else if c >= 0 then c * 1600 + i
  else (i / 4294967296) * 1600 + i % 4294967296

noextract
let dom (c:int) (i:int) : prop =
  i == -1 \/
  (0 <= i /\ (if c >= 0 then i < 1600 else (i % 4294967296 < 1600 /\ i / 4294967296 < 96)))

noextract
let gm_sign (c:int) (i:int) : Lemma (requires dom c i) (ensures (gm c i < 0 <==> i < 0)) = ()

noextract
let gm_mono (c:int) (i k:int)
  : Lemma (requires dom c i /\ dom c k /\ 0 <= i /\ i < k) (ensures gm c i < gm c k)
  = if c >= 0 then ()
    else begin
      FStar.Math.Lemmas.euclidean_division_definition i 4294967296;
      FStar.Math.Lemmas.euclidean_division_definition k 4294967296;
      let qi = i / 4294967296 in
      let qk = k / 4294967296 in
      if qi < qk then FStar.Math.Lemmas.lemma_mult_le_right 1600 (qi + 1) qk
      else if qi = qk then ()
      else begin
        FStar.Math.Lemmas.lemma_mult_le_right 4294967296 (qk + 1) qi
      end
    end

[@@"opaque_to_smt"]
noextract
let slot_ok (s:S.seq BF.t) (c:int) (lo hi:int) (v:F32.t) (i:I64.t) : prop =
  dom c (I64.v i) /\ best s lo hi (gm c (I64.v i)) /\
  (I64.v i >= 0 ==> v == ws s (gm c (I64.v i)))

inline_for_extraction noextract
let isnan (x:F32.t) : bool = neq x x

inline_for_extraction noextract
let argmax_wins (cand:F32.t) (idx:I64.t) (cur:F32.t) (cur_idx:I64.t) : bool =
  if I64.lt idx 0L then false
  else if I64.lt cur_idx 0L then true
  else if isnan cand then not (isnan cur) || I64.lt idx cur_idx
  else not (isnan cur) && gt cand cur

noextract
let isnan_spec (x:F32.t) : Lemma (isnan x <==> nanf x)
  = F.eq_spec x x; F.bit_eq_spec x x

noextract
let wins_ok (s:S.seq BF.t) (c:int) (va:F32.t) (ia:I64.t) (vb:F32.t) (ib:I64.t)
  : Lemma (requires dom c (I64.v ia) /\ dom c (I64.v ib) /\
                    (I64.v ia >= 0 ==> va == ws s (gm c (I64.v ia))) /\
                    (I64.v ib >= 0 ==> vb == ws s (gm c (I64.v ib))))
          (ensures (argmax_wins vb ib va ia <==> wins_g s (gm c (I64.v ia)) (gm c (I64.v ib))))
  = isnan_spec va; isnan_spec vb;
    gm_sign c (I64.v ia); gm_sign c (I64.v ib);
    if I64.v ia >= 0 && I64.v ib >= 0 then begin
      if I64.v ib < I64.v ia then gm_mono c (I64.v ib) (I64.v ia)
      else if I64.v ia < I64.v ib then gm_mono c (I64.v ia) (I64.v ib)
      else ()
    end

noextract
let merge_ok (s:S.seq BF.t) (c:int) (lo mid hi:int) (va:F32.t) (ia:I64.t) (vb:F32.t) (ib:I64.t)
  : Lemma (requires lo <= mid /\ mid <= hi /\ slot_ok s c lo mid va ia /\ slot_ok s c mid hi vb ib)
          (ensures (let w = argmax_wins vb ib va ia in
                    slot_ok s c lo hi (if w then vb else va) (if w then ib else ia)))
  = reveal_opaque (`%slot_ok) slot_ok;
    wins_ok s c va ia vb ib;
    best_merge s lo mid hi (gm c (I64.v ia)) (gm c (I64.v ib))

(* ------------------------------------------------------------------ *)
(* Slot arrays and tree levels                                         *)

noextract
let vat (sv:S.seq F32.t) (k:int) : F32.t =
  if 0 <= k && k < S.length sv then S.index sv k else F.zero
noextract
let iat (si:S.seq I64.t) (k:int) : I64.t =
  if 0 <= k && k < S.length si then S.index si k else (-1L)

noextract
let ivlo (off u:int) (n:nat) (p:nat) : int = off + rev_bits n p * u

(* Slots base+p, plo <= p < phi, hold the best over their level-lv interval. *)
[@@"opaque_to_smt"]
noextract
let lvl_ok (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
           (off u:int) (base:nat) (lv:nat) (plo phi:nat) : prop =
  forall (p:nat). plo <= p /\ p < phi ==>
    slot_ok s c (ivlo off u lv p) (ivlo off u lv p + u) (vat sv (base + p)) (iat si (base + p))

noextract
let lvl_get (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
            (off u:int) (base:nat) (lv:nat) (plo phi:nat) (p:nat)
  : Lemma (requires lvl_ok s c sv si off u base lv plo phi /\ plo <= p /\ p < phi)
          (ensures slot_ok s c (ivlo off u lv p) (ivlo off u lv p + u) (vat sv (base + p)) (iat si (base + p)))
  = reveal_opaque (`%lvl_ok) (lvl_ok s c sv si off u base lv plo phi)

noextract
let lvl_intro (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
              (off u:int) (base:nat) (lv:nat) (plo phi:nat)
  : Lemma (requires forall (p:nat). plo <= p /\ p < phi ==>
             slot_ok s c (ivlo off u lv p) (ivlo off u lv p + u) (vat sv (base + p)) (iat si (base + p)))
          (ensures lvl_ok s c sv si off u base lv plo phi)
  = reveal_opaque (`%lvl_ok) (lvl_ok s c sv si off u base lv plo phi)

noextract
let lvl_empty (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
              (off u:int) (base:nat) (lv:nat) (p:nat)
  : Lemma (lvl_ok s c sv si off u base lv p p)
  = lvl_intro s c sv si off u base lv p p

[@@"opaque_to_smt"]
noextract
let same_out (sv:S.seq F32.t) (si:S.seq I64.t) (sv':S.seq F32.t) (si':S.seq I64.t) (lo hi:int) : prop =
  S.length sv' == S.length sv /\ S.length si' == S.length si /\
  (forall (k:int). (k < lo \/ k >= hi) ==> vat sv' k == vat sv k /\ iat si' k == iat si k)

noextract
let same_get (sv:S.seq F32.t) (si:S.seq I64.t) (sv':S.seq F32.t) (si':S.seq I64.t) (lo hi:int) (k:int)
  : Lemma (requires same_out sv si sv' si' lo hi /\ (k < lo \/ k >= hi))
          (ensures S.length sv' == S.length sv /\ S.length si' == S.length si /\
                   vat sv' k == vat sv k /\ iat si' k == iat si k)
  = reveal_opaque (`%same_out) (same_out sv si sv' si' lo hi)

noextract
let same_refl (sv:S.seq F32.t) (si:S.seq I64.t) (lo hi:int)
  : Lemma (same_out sv si sv si lo hi)
  = reveal_opaque (`%same_out) (same_out sv si sv si lo hi)

noextract
let same_upd (sv:S.seq F32.t) (si:S.seq I64.t) (lo hi:int) (a:nat) (x:F32.t) (y:I64.t)
  : Lemma (requires lo <= a /\ a < hi /\ a < S.length sv /\ a < S.length si)
          (ensures same_out sv si (S.upd sv a x) (S.upd si a y) lo hi)
  = reveal_opaque (`%same_out) (same_out sv si (S.upd sv a x) (S.upd si a y) lo hi)

noextract
let same_trans (sv0:S.seq F32.t) (si0:S.seq I64.t) (sv1:S.seq F32.t) (si1:S.seq I64.t)
               (sv2:S.seq F32.t) (si2:S.seq I64.t) (lo hi lo2 hi2:int)
  : Lemma (requires same_out sv0 si0 sv1 si1 lo hi /\ same_out sv1 si1 sv2 si2 lo2 hi2 /\
                    lo <= lo2 /\ hi2 <= hi)
          (ensures same_out sv0 si0 sv2 si2 lo hi)
  = reveal_opaque (`%same_out) (same_out sv0 si0 sv1 si1 lo hi);
    reveal_opaque (`%same_out) (same_out sv1 si1 sv2 si2 lo2 hi2);
    reveal_opaque (`%same_out) (same_out sv0 si0 sv2 si2 lo hi)

#push-options "--fuel 1"
noextract
let pow2_succ (k:nat) : Lemma (pow2 (k + 1) == 2 * pow2 k) = ()
noextract
let rev0 (x:nat) : Lemma (rev_bits 0 x == 0) = ()
#pop-options

noextract
let pow2_down (n lv:nat)
  : Lemma (requires 1 <= lv /\ lv <= n) (ensures pow2 (n - (lv - 1)) == 2 * pow2 (n - lv))
  = pow2_succ (n - lv)

noextract
let unit_down (u:nat) (n lv:nat)
  : Lemma (requires 1 <= lv /\ lv <= n)
          (ensures u * pow2 (n - (lv - 1)) == 2 * (u * pow2 (n - lv)))
  = pow2_down n lv;
    FStar.Math.Lemmas.paren_mul_right u 2 (pow2 (n - lv));
    FStar.Math.Lemmas.swap_mul u 2;
    FStar.Math.Lemmas.paren_mul_right 2 u (pow2 (n - lv))

(* Halving the tree delta: pow2 (lv - 1) / 2 is pow2 (lv - 2), or 0 at lv = 1. *)
noextract
let delta_down (lv:pos)
  : Lemma (pow2 (lv - 1) / 2 == (if lv - 1 = 0 then 0 else pow2 (lv - 1 - 1)))
  = pow2_small ();
    if lv - 1 = 0 then () else pow2_succ (lv - 1 - 1)

noextract
let ivlo_split (off:int) (u:nat) (lv:pos) (t:nat)
  : Lemma (requires t < pow2 (lv - 1))
          (ensures ivlo off u lv t == ivlo off (2 * u) (lv - 1) t /\
                   ivlo off u lv (t + pow2 (lv - 1)) == ivlo off u lv t + u)
  = rev_bits_split (lv - 1) t;
    let r = rev_bits (lv - 1) t in
    FStar.Math.Lemmas.paren_mul_right 2 r u;
    FStar.Math.Lemmas.paren_mul_right r 2 u;
    FStar.Math.Lemmas.swap_mul 2 r;
    FStar.Math.Lemmas.distributivity_add_left (2 * r) 1 u

noextract
let step_lemma (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
               (off:int) (u:nat) (base:nat) (lv:pos) (t:nat)
  : Lemma (requires t < pow2 (lv - 1) /\ base + pow2 lv <= S.length sv /\ S.length si == S.length sv /\
                    lvl_ok s c sv si off (2 * u) base (lv - 1) 0 t /\
                    lvl_ok s c sv si off u base lv t (pow2 lv))
          (ensures (let a = base + t in
                    let b = a + pow2 (lv - 1) in
                    let w = argmax_wins (vat sv b) (iat si b) (vat sv a) (iat si a) in
                    let sv' = S.upd sv a (if w then vat sv b else vat sv a) in
                    let si' = S.upd si a (if w then iat si b else iat si a) in
                    lvl_ok s c sv' si' off (2 * u) base (lv - 1) 0 (t + 1) /\
                    lvl_ok s c sv' si' off u base lv (t + 1) (pow2 lv)))
  = let d = pow2 (lv - 1) in
    pow2_succ (lv - 1);
    let a = base + t in
    let b = a + d in
    let w = argmax_wins (vat sv b) (iat si b) (vat sv a) (iat si a) in
    let sv' = S.upd sv a (if w then vat sv b else vat sv a) in
    let si' = S.upd si a (if w then iat si b else iat si a) in
    ivlo_split off u lv t;
    lvl_get s c sv si off u base lv t (pow2 lv) t;
    lvl_get s c sv si off u base lv t (pow2 lv) (t + d);
    let lo = ivlo off u lv t in
    merge_ok s c lo (lo + u) (lo + u + u) (vat sv a) (iat si a) (vat sv b) (iat si b);
    assert (vat sv' a == (if w then vat sv b else vat sv a));
    assert (iat si' a == (if w then iat si b else iat si a));
    assert (slot_ok s c lo (lo + u + u) (vat sv' a) (iat si' a));
    assert (ivlo off (2 * u) (lv - 1) t == lo);
    assert (slot_ok s c (ivlo off (2 * u) (lv - 1) t) (ivlo off (2 * u) (lv - 1) t + 2 * u) (vat sv' (base + t)) (iat si' (base + t)));
    let aux1 (p:nat) : Lemma (0 <= p /\ p < t + 1 ==>
        slot_ok s c (ivlo off (2 * u) (lv - 1) p) (ivlo off (2 * u) (lv - 1) p + 2 * u)
          (vat sv' (base + p)) (iat si' (base + p)))
      = if p < t then begin
          lvl_get s c sv si off (2 * u) base (lv - 1) 0 t p;
          assert (vat sv' (base + p) == vat sv (base + p));
          assert (iat si' (base + p) == iat si (base + p))
        end in
    Classical.forall_intro aux1;
    lvl_intro s c sv' si' off (2 * u) base (lv - 1) 0 (t + 1);
    let aux2 (p:nat) : Lemma (t + 1 <= p /\ p < pow2 lv ==>
        slot_ok s c (ivlo off u lv p) (ivlo off u lv p + u) (vat sv' (base + p)) (iat si' (base + p)))
      = if t + 1 <= p && p < pow2 lv then begin
          lvl_get s c sv si off u base lv t (pow2 lv) p;
          assert (vat sv' (base + p) == vat sv (base + p));
          assert (iat si' (base + p) == iat si (base + p))
        end in
    Classical.forall_intro aux2;
    lvl_intro s c sv' si' off u base lv (t + 1) (pow2 lv)

(* Level 0 of a 2^n-slot tree with unit u * 2^n is the whole interval. *)
noextract
let level0 (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t) (off u:int) (base:nat)
  : Lemma (requires lvl_ok s c sv si off u base 0 0 1)
          (ensures slot_ok s c off (off + u) (vat sv base) (iat si base))
  = rev0 0; lvl_get s c sv si off u base 0 0 1 0

noextract
let level_top (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t) (off u:int) (base n:nat)
  : Lemma (requires lvl_ok s c sv si off u base n 0 (pow2 n))
          (ensures lvl_ok s c sv si off (u * pow2 (n - n)) base n 0 (pow2 n))
  = pow2_small ()

(* ------------------------------------------------------------------ *)
(* Warps: lanes of warp w occupy a level-5 tree at OFF + rev nb w * 32U  *)

noextract
let lane_lo (off u:int) (nb:nat) (p:nat) : int =
  off + (rev_bits nb (p / 32) * 32 + rev_bits 5 (p % 32)) * u

[@@"opaque_to_smt"]
noextract
let lanes_ok (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
             (off u:int) (nb:nat) (plo phi:nat) : prop =
  forall (p:nat). plo <= p /\ p < phi ==>
    slot_ok s c (lane_lo off u nb p) (lane_lo off u nb p + u) (vat sv p) (iat si p)

noextract
let lanes_get (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
              (off u:int) (nb:nat) (plo phi:nat) (p:nat)
  : Lemma (requires lanes_ok s c sv si off u nb plo phi /\ plo <= p /\ p < phi)
          (ensures slot_ok s c (lane_lo off u nb p) (lane_lo off u nb p + u) (vat sv p) (iat si p))
  = reveal_opaque (`%lanes_ok) (lanes_ok s c sv si off u nb plo phi)

noextract
let lanes_intro (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
                (off u:int) (nb:nat) (plo phi:nat)
  : Lemma (requires forall (p:nat). plo <= p /\ p < phi ==>
             slot_ok s c (lane_lo off u nb p) (lane_lo off u nb p + u) (vat sv p) (iat si p))
          (ensures lanes_ok s c sv si off u nb plo phi)
  = reveal_opaque (`%lanes_ok) (lanes_ok s c sv si off u nb plo phi)

noextract
let warp_pre (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
             (off u:int) (nb:nat) (w:nat) (phi:nat)
  : Lemma (requires lanes_ok s c sv si off u nb (32 * w) phi /\ 32 * w + 32 <= phi)
          (ensures lvl_ok s c sv si (off + rev_bits nb w * (32 * u)) u (32 * w) 5 0 (pow2 5))
  = pow2_small ();
    let aux (q:nat) : Lemma (q < 32 ==>
        slot_ok s c (ivlo (off + rev_bits nb w * (32 * u)) u 5 q) (ivlo (off + rev_bits nb w * (32 * u)) u 5 q + u)
          (vat sv (32 * w + q)) (iat si (32 * w + q)))
      = if q < 32 then begin
          FStar.Math.Lemmas.lemma_div_plus q w 32;
          FStar.Math.Lemmas.small_div q 32;
          FStar.Math.Lemmas.lemma_mod_plus q w 32;
          FStar.Math.Lemmas.small_mod q 32;
          assert ((32 * w + q) / 32 == w);
          assert ((32 * w + q) % 32 == q);
          FStar.Math.Lemmas.distributivity_add_left (rev_bits nb w * 32) (rev_bits 5 q) u;
          FStar.Math.Lemmas.paren_mul_right (rev_bits nb w) 32 u;
          assert (lane_lo off u nb (32 * w + q) == ivlo (off + rev_bits nb w * (32 * u)) u 5 q);
          lanes_get s c sv si off u nb (32 * w) phi (32 * w + q)
        end in
    Classical.forall_intro aux;
    lvl_intro s c sv si (off + rev_bits nb w * (32 * u)) u (32 * w) 5 0 (pow2 5)

noextract
let warp_post (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
              (sv':S.seq F32.t) (si':S.seq I64.t)
              (off u:int) (nb:nat) (w:nat) (phi:nat)
  : Lemma (requires lanes_ok s c sv si off u nb (32 * w) phi /\ 32 * w + 32 <= phi /\
                    lvl_ok s c sv si off (32 * u) 0 nb 0 w /\
                    S.length sv == S.length si /\ phi <= S.length sv /\
                    S.length sv' == S.length sv /\ S.length si' == S.length si /\
                    same_out sv si sv' si' (32 * w) (32 * w + 32) /\
                    slot_ok s c (off + rev_bits nb w * (32 * u)) (off + rev_bits nb w * (32 * u) + 32 * u)
                      (vat sv' (32 * w)) (iat si' (32 * w)))
          (ensures (let sv'' = S.upd sv' w (vat sv' (32 * w)) in
                    let si'' = S.upd si' w (iat si' (32 * w)) in
                    lanes_ok s c sv'' si'' off u nb (32 * (w + 1)) phi /\
                    lvl_ok s c sv'' si'' off (32 * u) 0 nb 0 (w + 1)))
  = let sv'' = S.upd sv' w (vat sv' (32 * w)) in
    let si'' = S.upd si' w (iat si' (32 * w)) in
    let aux1 (p:nat) : Lemma (32 * (w + 1) <= p /\ p < phi ==>
        slot_ok s c (lane_lo off u nb p) (lane_lo off u nb p + u) (vat sv'' p) (iat si'' p))
      = if 32 * (w + 1) <= p && p < phi then begin
          lanes_get s c sv si off u nb (32 * w) phi p;
          same_get sv si sv' si' (32 * w) (32 * w + 32) p;
          assert (vat sv'' p == vat sv' p);
          assert (iat si'' p == iat si' p)
        end in
    Classical.forall_intro aux1;
    lanes_intro s c sv'' si'' off u nb (32 * (w + 1)) phi;
    let aux2 (p:nat) : Lemma (0 <= p /\ p < w + 1 ==>
        slot_ok s c (ivlo off (32 * u) nb p) (ivlo off (32 * u) nb p + 32 * u) (vat sv'' (0 + p)) (iat si'' (0 + p)))
      = if p < w then begin
          lvl_get s c sv si off (32 * u) 0 nb 0 w p;
          same_get sv si sv' si' (32 * w) (32 * w + 32) p;
          assert (vat sv'' p == vat sv' p);
          assert (iat si'' p == iat si' p)
        end
        else if p = w then begin
          assert (vat sv'' p == vat sv' (32 * w));
          assert (iat si'' p == iat si' (32 * w))
        end in
    Classical.forall_intro aux2;
    lvl_intro s c sv'' si'' off (32 * u) 0 nb 0 (w + 1)

(* ------------------------------------------------------------------ *)
(* Stage facts                                                         *)

noextract
let best_extend (s:S.seq BF.t) (lo mid hi:int) (g:int)
  : Lemma (requires lo <= mid /\ mid <= hi /\ best s lo mid g /\ (forall (j:nat). ~(in_iv s mid hi j)))
          (ensures best s lo hi g)
  = best_merge s lo mid hi g (-1)

noextract
let lane_exit (s:S.seq BF.t) (c t m:nat) (v:F32.t) (i:I64.t)
  : Lemma (requires S.length s == nV /\ c < 96 /\ t < 128 /\ m <= 13 /\
                    (t + 128 * m >= 1600 \/ c * 1600 + t + 128 * m >= nV) /\
                    slot_ok s c (lane_base c t) (lane_base c t + m) v i)
          (ensures slot_ok s c (lane_base c t) (lane_base c t + 13) v i)
  = reveal_opaque (`%slot_ok) slot_ok;
    let aux (j:nat) : Lemma (~(in_iv s (lane_base c t + m) (lane_base c t + 13) j))
      = if sem (in_iv s (lane_base c t + m) (lane_base c t + 13) j) then begin
          lane_member c t 13 j;
          key_parts j;
          let m' = (j % 1600) / 128 in
          assert (opt_key j == lane_base c t + m');
          assert (m <= m');
          FStar.Math.Lemmas.euclidean_division_definition (j % 1600) 128;
          assert (j % 1600 == t + 128 * m');
          assert (j == c * 1600 + t + 128 * m')
        end in
    Classical.forall_intro aux;
    best_extend s (lane_base c t) (lane_base c t + m) (lane_base c t + 13) (gm c (I64.v i))

noextract
let lane_step (s:S.seq BF.t) (c t m:nat) (v:F32.t) (i:I64.t) (cand:F32.t) (ci:I64.t)
  : Lemma (requires S.length s == nV /\ t < 128 /\ t + 128 * m < 1600 /\ c * 1600 + t + 128 * m < nV /\
                    I64.v ci == t + 128 * m /\ cand == ws s (c * 1600 + t + 128 * m) /\
                    slot_ok s c (lane_base c t) (lane_base c t + m) v i)
          (ensures (let w = argmax_wins cand ci v i in
                    slot_ok s c (lane_base c t) (lane_base c t + (m + 1))
                      (if w then cand else v) (if w then ci else i)))
  = let j = c * 1600 + t + 128 * m in
    lane_key c t m;
    best_single s j;
    reveal_opaque (`%slot_ok) (slot_ok s c (lane_base c t + m) (lane_base c t + m + 1) cand ci);
    assert (gm c (I64.v ci) == j);
    assert (slot_ok s c (lane_base c t + m) (lane_base c t + m + 1) cand ci);
    merge_ok s c (lane_base c t) (lane_base c t + m) (lane_base c t + m + 1) v i cand ci

noextract
let lane_init (s:S.seq BF.t) (c t:nat) (v:F32.t)
  : Lemma (slot_ok s c (lane_base c t) (lane_base c t + 0) v (-1L)) = reveal_opaque (`%slot_ok) slot_ok

(* Chunk stage lanes are the level structure with OFF = 1664 crank c, U = 13. *)
noextract
let lanes_from_lane_base (s:S.seq BF.t) (c:nat) (sv:S.seq F32.t) (si:S.seq I64.t)
  : Lemma (requires forall (t:nat). t < 128 ==>
             slot_ok s c (lane_base c t) (lane_base c t + 13) (vat sv t) (iat si t))
          (ensures lanes_ok s c sv si (crank c * 1664) 13 2 0 128)
  = lanes_intro s c sv si (crank c * 1664) 13 2 0 128

noextract
let enc (c:nat) (i:I64.t) : int = if I64.v i < 0 then -1 else c * 4294967296 + I64.v i

noextract
let enc_ok (s:S.seq BF.t) (c:nat) (lo hi:int) (v:F32.t) (i:I64.t) (e:I64.t)
  : Lemma (requires c < 96 /\ slot_ok s c lo hi v i /\ I64.v e == enc c i)
          (ensures slot_ok s (-1) lo hi v e)
  = reveal_opaque (`%slot_ok) slot_ok;
    if I64.v i >= 0 then begin
      FStar.Math.Lemmas.lemma_div_plus (I64.v i) c 4294967296;
      FStar.Math.Lemmas.small_div (I64.v i) 4294967296;
      FStar.Math.Lemmas.lemma_mod_plus (I64.v i) c 4294967296;
      FStar.Math.Lemmas.small_mod (I64.v i) 4294967296
    end

noextract
let chunk_pad (s:S.seq BF.t) (p:nat) (v:F32.t)
  : Lemma (requires S.length s == nV /\ 96 <= p /\ p < 256)
          (ensures slot_ok s (-1) (crank p * 1664) (crank p * 1664 + 1664) v (-1L))
  = reveal_opaque (`%slot_ok) slot_ok;
    let aux (j:nat) : Lemma (~(in_iv s (crank p * 1664) (crank p * 1664 + 1664) j))
      = if sem (in_iv s (crank p * 1664) (crank p * 1664 + 1664) j) then begin
          chunk_member p j; key_parts_bound j
        end in
    Classical.forall_intro aux

noextract
let chunks_lanes (s:S.seq BF.t) (sv:S.seq F32.t) (si:S.seq I64.t)
  : Lemma (requires forall (p:nat). p < 256 ==>
             slot_ok s (-1) (crank p * 1664) (crank p * 1664 + 1664) (vat sv p) (iat si p))
          (ensures lanes_ok s (-1) sv si 0 1664 3 0 256)
  = lanes_intro s (-1) sv si 0 1664 3 0 256

noextract
let decode (e:int) : int = (e / 4294967296) * 1600 + e % 4294967296

noextract
let final_ok (s:S.seq BF.t) (v:F32.t) (e:I64.t)
  : Lemma (requires S.length s == nV /\ slot_ok s (-1) 0 425984 v e)
          (ensures I64.v e >= 0 /\ I64.v e / 4294967296 < 96 /\ tree_argmax s (decode (I64.v e)))
  = reveal_opaque (`%slot_ok) slot_ok;
    best_final s (gm (-1) (I64.v e))

(* ------------------------------------------------------------------ *)
(* Pulse-facing wrappers                                               *)

noextract
let merged_v (sv:S.seq F32.t) (si:S.seq I64.t) (a b:nat) : S.seq F32.t =
  if a < S.length sv then
    S.upd sv a (if argmax_wins (vat sv b) (iat si b) (vat sv a) (iat si a) then vat sv b else vat sv a)
  else sv

noextract
let merged_i (sv:S.seq F32.t) (si:S.seq I64.t) (a b:nat) : S.seq I64.t =
  if a < S.length si then
    S.upd si a (if argmax_wins (vat sv b) (iat si b) (vat sv a) (iat si a) then iat si b else iat si a)
  else si

noextract
let tree_step (s:S.seq BF.t) (c:int) (sv0:S.seq F32.t) (si0:S.seq I64.t)
              (sv:S.seq F32.t) (si:S.seq I64.t)
              (off:int) (u:nat) (base:nat) (lv:pos) (t:nat)
  : Lemma (requires t < pow2 (lv - 1) /\ base + pow2 lv <= S.length sv /\ S.length si == S.length sv /\
                    lvl_ok s c sv si off (2 * u) base (lv - 1) 0 t /\
                    lvl_ok s c sv si off u base lv t (pow2 lv) /\
                    same_out sv0 si0 sv si base (base + pow2 lv))
          (ensures (let a = base + t in
                    let b = a + pow2 (lv - 1) in
                    let sv' = merged_v sv si a b in
                    let si' = merged_i sv si a b in
                    S.length sv' == S.length sv /\ S.length si' == S.length si /\
                    lvl_ok s c sv' si' off (2 * u) base (lv - 1) 0 (t + 1) /\
                    lvl_ok s c sv' si' off u base lv (t + 1) (pow2 lv) /\
                    same_out sv0 si0 sv' si' base (base + pow2 lv)))
  = pow2_succ (lv - 1);
    step_lemma s c sv si off u base lv t;
    let a = base + t in
    let b = a + pow2 (lv - 1) in
    let w = argmax_wins (vat sv b) (iat si b) (vat sv a) (iat si a) in
    same_upd sv si base (base + pow2 lv) a (if w then vat sv b else vat sv a) (if w then iat si b else iat si a);
    same_trans sv0 si0 sv si (merged_v sv si a b) (merged_i sv si a b) base (base + pow2 lv) base (base + pow2 lv)

noextract
let level_done (s:S.seq BF.t) (c:int) (sv0:S.seq F32.t) (si0:S.seq I64.t)
               (sv:S.seq F32.t) (si:S.seq I64.t) (sv1:S.seq F32.t) (si1:S.seq I64.t)
               (off:int) (u:nat) (base:nat) (n:nat) (lv:pos)
  : Lemma (requires lv <= n /\
                    lvl_ok s c sv1 si1 off (2 * (u * pow2 (n - lv))) base (lv - 1) 0 (pow2 (lv - 1)) /\
                    same_out sv0 si0 sv si base (base + pow2 n) /\
                    same_out sv si sv1 si1 base (base + pow2 lv))
          (ensures lvl_ok s c sv1 si1 off (u * pow2 (n - (lv - 1))) base (lv - 1) 0 (pow2 (lv - 1)) /\
                   same_out sv0 si0 sv1 si1 base (base + pow2 n) /\
                   pow2 (lv - 1) / 2 == (if lv - 1 = 0 then 0 else pow2 (lv - 1 - 1)))
  = unit_down u n lv;
    FStar.Math.Lemmas.pow2_le_compat n lv;
    same_trans sv0 si0 sv si sv1 si1 base (base + pow2 n) base (base + pow2 lv);
    delta_down lv

noextract
let tree_init (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
              (off:int) (u:nat) (base:nat) (n:pos)
  : Lemma (requires lvl_ok s c sv si off u base n 0 (pow2 n))
          (ensures lvl_ok s c sv si off (u * pow2 (n - n)) base n 0 (pow2 n) /\
                   same_out sv si sv si base (base + pow2 n) /\
                   pow2 (n - 1) > 0)
  = level_top s c sv si off u base n; same_refl sv si base (base + pow2 n)

noextract
let tree_exit (s:S.seq BF.t) (c:int) (sv:S.seq F32.t) (si:S.seq I64.t)
              (off:int) (u:nat) (base:nat) (n:nat) (lv:nat)
  : Lemma (requires lv <= n /\ (if lv = 0 then 0 else pow2 (lv - 1)) == 0 /\
                    lvl_ok s c sv si off (u * pow2 (n - lv)) base lv 0 (pow2 lv))
          (ensures slot_ok s c off (off + u * pow2 n) (vat sv base) (iat si base))
  = pow2_small ();
    level0 s c sv si off (u * pow2 n) base

(* Lane results of one chunk: slots t' < t hold their lane intervals. *)
[@@"opaque_to_smt"]
noextract
let lanes_pre (s:S.seq BF.t) (c:nat) (sv:S.seq F32.t) (si:S.seq I64.t) (t:nat) : prop =
  forall (t':nat). t' < t ==> slot_ok s c (lane_base c t') (lane_base c t' + 13) (vat sv t') (iat si t')

noextract
let lanes_pre_empty (s:S.seq BF.t) (c:nat) (sv:S.seq F32.t) (si:S.seq I64.t)
  : Lemma (lanes_pre s c sv si 0)
  = reveal_opaque (`%lanes_pre) (lanes_pre s c sv si 0)

noextract
let lanes_pre_step (s:S.seq BF.t) (c:nat) (sv:S.seq F32.t) (si:S.seq I64.t) (t:nat) (v:F32.t) (i:I64.t)
  : Lemma (requires lanes_pre s c sv si t /\ t < S.length sv /\ t < S.length si /\
                    slot_ok s c (lane_base c t) (lane_base c t + 13) v i)
          (ensures lanes_pre s c (S.upd sv t v) (S.upd si t i) (t + 1))
  = reveal_opaque (`%lanes_pre) (lanes_pre s c sv si t);
    reveal_opaque (`%lanes_pre) (lanes_pre s c (S.upd sv t v) (S.upd si t i) (t + 1));
    let sv' = S.upd sv t v in
    let si' = S.upd si t i in
    let aux (t':nat) : Lemma (t' < t + 1 ==>
        slot_ok s c (lane_base c t') (lane_base c t' + 13) (vat sv' t') (iat si' t'))
      = if t' < t then begin
          assert (vat sv' t' == vat sv t');
          assert (iat si' t' == iat si t')
        end
        else if t' = t then begin
          assert (vat sv' t' == v);
          assert (iat si' t' == i)
        end in
    Classical.forall_intro aux

noextract
let lanes_pre_done (s:S.seq BF.t) (c:nat) (sv:S.seq F32.t) (si:S.seq I64.t)
  : Lemma (requires lanes_pre s c sv si 128)
          (ensures lanes_ok s c sv si (crank c * 1664) 13 2 0 128)
  = reveal_opaque (`%lanes_pre) (lanes_pre s c sv si 128);
    lanes_from_lane_base s c sv si

(* Chunk results: slots p < k hold the best over chunk p's key block. *)
[@@"opaque_to_smt"]
noextract
let chunks_pre (s:S.seq BF.t) (sv:S.seq F32.t) (si:S.seq I64.t) (k:nat) : prop =
  forall (p:nat). p < k ==>
    slot_ok s (-1) (crank p * 1664) (crank p * 1664 + 1664) (vat sv p) (iat si p)

noextract
let chunks_pre_empty (s:S.seq BF.t) (sv:S.seq F32.t) (si:S.seq I64.t)
  : Lemma (chunks_pre s sv si 0)
  = reveal_opaque (`%chunks_pre) (chunks_pre s sv si 0)

noextract
let chunks_pre_get (s:S.seq BF.t) (sv:S.seq F32.t) (si:S.seq I64.t) (k p:nat)
  : Lemma (requires chunks_pre s sv si k /\ p < k)
          (ensures slot_ok s (-1) (crank p * 1664) (crank p * 1664 + 1664) (vat sv p) (iat si p))
  = reveal_opaque (`%chunks_pre) (chunks_pre s sv si k)

noextract
let chunks_pre_step (s:S.seq BF.t) (sv:S.seq F32.t) (si:S.seq I64.t) (k:nat) (v:F32.t) (i:I64.t)
  : Lemma (requires chunks_pre s sv si k /\ k < S.length sv /\ k < S.length si /\
                    slot_ok s (-1) (crank k * 1664) (crank k * 1664 + 1664) v i)
          (ensures chunks_pre s (S.upd sv k v) (S.upd si k i) (k + 1))
  = reveal_opaque (`%chunks_pre) (chunks_pre s sv si k);
    reveal_opaque (`%chunks_pre) (chunks_pre s (S.upd sv k v) (S.upd si k i) (k + 1));
    let sv' = S.upd sv k v in
    let si' = S.upd si k i in
    let aux (p:nat) : Lemma (p < k + 1 ==>
        slot_ok s (-1) (crank p * 1664) (crank p * 1664 + 1664) (vat sv' p) (iat si' p))
      = if p < k then begin
          assert (vat sv' p == vat sv p);
          assert (iat si' p == iat si p)
        end
        else if p = k then begin
          assert (vat sv' p == v);
          assert (iat si' p == i)
        end in
    Classical.forall_intro aux

noextract
let chunks_pre_done (s:S.seq BF.t) (sv:S.seq F32.t) (si:S.seq I64.t)
  : Lemma (requires chunks_pre s sv si 256)
          (ensures lanes_ok s (-1) sv si 0 1664 3 0 256)
  = reveal_opaque (`%chunks_pre) (chunks_pre s sv si 256);
    chunks_lanes s sv si

(* Chunk element bound: min (1600, N - 1600 c), clipped at 0. *)
noextract
let lim_of (c:nat) : nat =
  if c * 1600 + 1600 <= nV then 1600 else if c * 1600 <= nV then nV - c * 1600 else 0

noextract
let lane_exit' (s:S.seq BF.t) (c t m:nat) (v:F32.t) (i:I64.t)
  : Lemma (requires S.length s == nV /\ c < 96 /\ t < 128 /\ m <= 13 /\
                    t + 128 * m >= lim_of c /\
                    slot_ok s c (lane_base c t) (lane_base c t + m) v i)
          (ensures slot_ok s c (lane_base c t) (lane_base c t + 13) v i)
  = lane_exit s c t m v i

#push-options "--fuel 9"
noextract
let pow2_8 () : Lemma (pow2 8 == 256) = ()
#pop-options

noextract
let pow2_big () : Lemma (pow2 32 == 4294967296 /\ pow2 63 == 9223372036854775808)
  = pow2_8 ();
    FStar.Math.Lemmas.pow2_plus 8 8;
    FStar.Math.Lemmas.pow2_plus 16 16;
    FStar.Math.Lemmas.pow2_plus 16 8;
    FStar.Math.Lemmas.pow2_plus 24 8;
    FStar.Math.Lemmas.pow2_plus 32 24;
    pow2_small ();
    FStar.Math.Lemmas.pow2_plus 56 4;
    FStar.Math.Lemmas.pow2_plus 60 2;
    FStar.Math.Lemmas.pow2_plus 62 1

noextract
let supd (#a:Type) (s:S.seq a) (i:nat) (x:a) : S.seq a =
  if i < S.length s then S.upd s i x else s

noextract
let slot_dom (s:S.seq BF.t) (c:int) (lo hi:int) (v:F32.t) (i:I64.t)
  : Lemma (requires slot_ok s c lo hi v i) (ensures dom c (I64.v i))
  = reveal_opaque (`%slot_ok) (slot_ok s c lo hi v i)

noextract
let fin_slot (s:S.seq BF.t) (cv:S.seq F32.t) (ci:S.seq I64.t) (t:nat) (ninf:F32.t) (v:F32.t) (i:I64.t)
  : Lemma (requires S.length s == nV /\ chunks_pre s cv ci 96 /\ S.length cv == 96 /\ S.length ci == 96 /\
                    t < 256 /\ v == (if t < 96 then vat cv t else ninf) /\
                    i == (if t < 96 then iat ci t else (-1L)))
          (ensures slot_ok s (-1) (crank t * 1664) (crank t * 1664 + 1664) v i)
  = if t < 96 then chunks_pre_get s cv ci 96 t else chunk_pad s t v

noextract
let decode_ok (e:I64.t)
  : Lemma (requires I64.v e >= 0 /\ I64.v e / 4294967296 < 96 /\ I64.v e % 4294967296 < 1600)
          (ensures (let q = I64.div e 4294967296L in
                    let r = I64.rem e 4294967296L in
                    I64.v q == I64.v e / 4294967296 /\ I64.v r == I64.v e % 4294967296 /\
                    I64.v q * 1600 + I64.v r < 160000 /\
                    I64.v q * 1600 + I64.v r == decode (I64.v e)))
  = pow2_big ();
    let x = I64.v e in
    assert (FStar.Int.(x /- 4294967296) == x / 4294967296);
    assert (FStar.Int.mod #64 x 4294967296 == x - (x / 4294967296) * 4294967296);
    FStar.Math.Lemmas.euclidean_division_definition x 4294967296



inline_for_extraction noextract
fn merge_at (values : A.array F32.t) (indices : A.array I64.t) (a b : SZ.t)
  (#sv : erased (S.seq F32.t)) (#si : erased (S.seq I64.t))
  requires values |-> sv ** indices |-> si **
    pure (SZ.v a < S.length sv /\ SZ.v b < S.length sv /\ S.length si == S.length sv)
  ensures values |-> merged_v sv si (SZ.v a) (SZ.v b) **
    indices |-> merged_i sv si (SZ.v a) (SZ.v b)
{
  open Pulse.Lib.Array;
  let va = values.(a);
  let ia = indices.(a);
  let vb = values.(b);
  let ib = indices.(b);
  let w = argmax_wins vb ib va ia;
  values.(a) <- (if w then vb else va);
  indices.(a) <- (if w then ib else ia);
  rewrite values |-> S.upd sv (SZ.v a) (if w then vb else va)
       as values |-> merged_v sv si (SZ.v a) (SZ.v b);
  rewrite indices |-> S.upd si (SZ.v a) (if w then ib else ia)
       as indices |-> merged_i sv si (SZ.v a) (SZ.v b);
}

inline_for_extraction noextract
fn tree_level (values : A.array F32.t) (indices : A.array I64.t) (base d : SZ.t)
  (s : erased (S.seq BF.t)) (c : erased int) (off : erased int) (u : erased nat) (lv : erased nat{lv > 0})
  (#sv0 : erased (S.seq F32.t)) (#si0 : erased (S.seq I64.t))
  requires values |-> sv0 ** indices |-> si0 **
    pure (SZ.v d == pow2 (lv - 1) /\ SZ.v base + pow2 lv <= S.length sv0 /\
          S.length sv0 <= 256 /\ S.length si0 == S.length sv0 /\
          lvl_ok s c sv0 si0 off u (SZ.v base) lv 0 (pow2 lv))
  ensures exists* sv si. values |-> sv ** indices |-> si **
    pure (S.length sv == S.length sv0 /\ S.length si == S.length si0 /\
          lvl_ok s c sv si off (2 * u) (SZ.v base) (lv - 1) 0 (pow2 (lv - 1)) /\
          same_out sv0 si0 sv si (SZ.v base) (SZ.v base + pow2 lv))
{
  lvl_empty s c sv0 si0 off (2 * u) (SZ.v base) (lv - 1) 0;
  same_refl sv0 si0 (SZ.v base) (SZ.v base + pow2 lv);
  pow2_succ (lv - 1);
  let mut t = 0sz;
  while (!t <^ d)
    invariant live t ** pure (SZ.v !t <= SZ.v d)
    invariant exists* sv si. values |-> sv ** indices |-> si **
      pure (S.length sv == S.length sv0 /\ S.length si == S.length si0 /\
            lvl_ok s c sv si off (2 * u) (SZ.v base) (lv - 1) 0 (SZ.v !t) /\
            lvl_ok s c sv si off u (SZ.v base) lv (SZ.v !t) (pow2 lv) /\
            same_out sv0 si0 sv si (SZ.v base) (SZ.v base + pow2 lv))
    decreases (SZ.v d - SZ.v !t)
  {
    let vt = !t;
    with sv si. assert (values |-> sv ** indices |-> si);
    tree_step s c sv0 si0 sv si off u (SZ.v base) lv (SZ.v vt);
    merge_at values indices (base +^ vt) (base +^ vt +^ d);
    t := vt +^ 1sz;
  };
}

inline_for_extraction noextract
fn tree (values : A.array F32.t) (indices : A.array I64.t) (base d0 : SZ.t)
  (s : erased (S.seq BF.t)) (c : erased int) (off : erased int) (u : erased nat) (n : erased nat{n > 0})
  (#sv0 : erased (S.seq F32.t)) (#si0 : erased (S.seq I64.t))
  requires values |-> sv0 ** indices |-> si0 **
    pure (SZ.v d0 == pow2 (n - 1) /\ SZ.v base + pow2 n <= S.length sv0 /\
          S.length sv0 <= 256 /\ S.length si0 == S.length sv0 /\
          lvl_ok s c sv0 si0 off u (SZ.v base) n 0 (pow2 n))
  ensures exists* sv si. values |-> sv ** indices |-> si **
    pure (S.length sv == S.length sv0 /\ S.length si == S.length si0 /\
          slot_ok s c off (off + u * pow2 n) (vat sv (SZ.v base)) (iat si (SZ.v base)) /\
          same_out sv0 si0 sv si (SZ.v base) (SZ.v base + pow2 n))
{
  tree_init s c sv0 si0 off u (SZ.v base) n;
  let lvr = GR.alloc #nat (reveal n);
  let mut delta = d0;
  while (!delta >^ 0sz)
    invariant live delta
    invariant exists* (lv:nat) sv si. GR.pts_to lvr lv ** values |-> sv ** indices |-> si **
      pure (lv <= n /\ SZ.v !delta == (if lv = 0 then 0 else pow2 (lv - 1)) /\
            S.length sv == S.length sv0 /\ S.length si == S.length si0 /\
            lvl_ok s c sv si off (u * pow2 (n - lv)) (SZ.v base) lv 0 (pow2 lv) /\
            same_out sv0 si0 sv si (SZ.v base) (SZ.v base + pow2 n))
    decreases (SZ.v !delta)
  {
    let vd = !delta;
    with lv sv si. assert (GR.pts_to lvr lv ** values |-> sv ** indices |-> si);
    FStar.Math.Lemmas.pow2_le_compat n lv;
    tree_level values indices base vd s c off (u * pow2 (n - lv)) lv;
    with sv1 si1. assert (values |-> sv1 ** indices |-> si1);
    level_done s c sv0 si0 sv si sv1 si1 off u (SZ.v base) n lv;
    GR.op_Colon_Equals lvr (lv - 1);
    delta := vd /^ 2sz;
  };
  with lv sv si. assert (GR.pts_to lvr lv ** values |-> sv ** indices |-> si);
  tree_exit s c sv si off u (SZ.v base) n lv;
  GR.free lvr;
}

inline_for_extraction noextract
fn reduce_block (values : A.array F32.t) (indices : A.array I64.t) (nw dx : SZ.t)
  (s : erased (S.seq BF.t)) (c : erased int) (off : erased int) (u : erased nat) (nb : erased nat{nb > 0})
  (#sv0 : erased (S.seq F32.t)) (#si0 : erased (S.seq I64.t))
  requires values |-> sv0 ** indices |-> si0 **
    pure (S.length sv0 == 256 /\ S.length si0 == 256 /\
          SZ.v nw == pow2 nb /\ 32 * SZ.v nw <= 256 /\ SZ.v dx == pow2 (nb - 1) /\
          lanes_ok s c sv0 si0 off u nb 0 (32 * SZ.v nw))
  ensures exists* sv si. values |-> sv ** indices |-> si **
    pure (S.length sv == 256 /\ S.length si == 256 /\
          slot_ok s c off (off + (32 * u) * pow2 nb) (vat sv 0) (iat si 0))
{
  open Pulse.Lib.Array;
  lvl_empty s c sv0 si0 off (32 * u) 0 nb 0;
  pow2_small ();
  let mut w = 0sz;
  while (!w <^ nw)
    invariant live w ** pure (SZ.v !w <= SZ.v nw)
    invariant exists* sv si. values |-> sv ** indices |-> si **
      pure (S.length sv == 256 /\ S.length si == 256 /\
            lanes_ok s c sv si off u nb (32 * SZ.v !w) (32 * SZ.v nw) /\
            lvl_ok s c sv si off (32 * u) 0 nb 0 (SZ.v !w))
    decreases (SZ.v nw - SZ.v !w)
  {
    let vw = !w;
    with sv si. assert (values |-> sv ** indices |-> si);
    warp_pre s c sv si off u nb (SZ.v vw) (32 * SZ.v nw);
    tree values indices (32sz *^ vw) 16sz s c (off + rev_bits nb (SZ.v vw) * (32 * u)) u 5;
    with sv1 si1. assert (values |-> sv1 ** indices |-> si1);
    warp_post s c sv si sv1 si1 off u nb (SZ.v vw) (32 * SZ.v nw);
    let v0 = values.(32sz *^ vw);
    let i0 = indices.(32sz *^ vw);
    values.(vw) <- v0;
    indices.(vw) <- i0;
    rewrite values |-> S.upd sv1 (SZ.v vw) v0
         as values |-> S.upd sv1 (SZ.v vw) (vat sv1 (32 * SZ.v vw));
    rewrite indices |-> S.upd si1 (SZ.v vw) i0
         as indices |-> S.upd si1 (SZ.v vw) (iat si1 (32 * SZ.v vw));
    w := vw +^ 1sz;
  };
  with sv si. assert (values |-> sv ** indices |-> si);
  tree values indices 0sz dx s c off (32 * u) nb;
}

inline_for_extraction noextract
let nv_sz : (x:SZ.t{SZ.v x == 151936}) = SZ.uint32_to_sizet 151936ul

inline_for_extraction noextract
let sz_to_i64 (x:SZ.t{SZ.v x < 65536}) : r:I64.t{I64.v r == SZ.v x} =
  Kuiper.SizeT.fits_ok (SZ.v x);
  Kuiper.SizeT.fits_iff_u32 (SZ.v x);
  FStar.Int.Cast.uint32_to_int64 (Kuiper.SizeT.sizet_to_u32 x)

noextract
let read_eq (s0:chest1 bf16 151936) (i:SZ.t{SZ.v i < 151936}) (x:bf16)
  : Lemma (requires x == acc s0 ((SZ.v i <: natlt 151936), ()))
          (ensures x == S.index (chest1_to_seq s0) (SZ.v i))
  = ()

inline_for_extraction noextract
fn scan_lane (input0 : array1 bf16 (l1_forward 151936) { is_global input0 })
  (values : A.array F32.t) (indices : A.array I64.t) (c t lim : SZ.t)
  (#p : perm) (#s0 : chest1 bf16 151936)
  (#sv : erased (S.seq F32.t)) (#si : erased (S.seq I64.t))
  preserves input0 |-> Frac p s0
  requires values |-> sv ** indices |-> si **
    pure (S.length sv == 256 /\ S.length si == 256 /\ SZ.v c < 96 /\ SZ.v t < 128 /\
          SZ.v lim == lim_of (SZ.v c))
  ensures exists* v i. values |-> supd sv (SZ.v t) v ** indices |-> supd si (SZ.v t) i **
    pure (slot_ok (chest1_to_seq s0) (SZ.v c) (lane_base (SZ.v c) (SZ.v t)) (lane_base (SZ.v c) (SZ.v t) + 13) v i)
{
  open Pulse.Lib.Array;
  let ninf : F32.t = neg #f32 infinity;
  lane_init (chest1_to_seq s0) (SZ.v c) (SZ.v t) ninf;
  let mut bv : F32.t = ninf;
  let mut bi : I64.t = (-1L);
  let mut i = t;
  let mr = GR.alloc #nat 0;
  while (!i <^ lim)
    invariant live i ** live bv ** live bi
    invariant exists* (m:nat). GR.pts_to mr m **
      pure (SZ.v !i == SZ.v t + 128 * m /\ m <= 13 /\
            slot_ok (chest1_to_seq s0) (SZ.v c) (lane_base (SZ.v c) (SZ.v t)) (lane_base (SZ.v c) (SZ.v t) + m) !bv !bi)
    decreases (2000 - SZ.v !i)
  {
    let vi = !i;
    with m. assert (GR.pts_to mr m);
    let gidx : szlt 151936 = c *^ 1600sz +^ vi;
    let x = Kuiper.Tensor.read input0 (gidx, ()) #p;
    read_eq s0 gidx x;
    let cand : F32.t = C.fcast #bf16 #F32.t x;
    let ci = sz_to_i64 vi;
    let cv = !bv;
    let cx = !bi;
    lane_step (chest1_to_seq s0) (SZ.v c) (SZ.v t) m cv cx cand ci;
    let w = argmax_wins cand ci cv cx;
    bv := (if w then cand else cv);
    bi := (if w then ci else cx);
    GR.op_Colon_Equals mr (m + 1);
    i := vi +^ 128sz;
  };
  with m. assert (GR.pts_to mr m);
  lane_exit' (chest1_to_seq s0) (SZ.v c) (SZ.v t) m !bv !bi;
  GR.free mr;
  let fv = !bv;
  let fi = !bi;
  values.(t) <- fv;
  indices.(t) <- fi;
  rewrite values |-> S.upd sv (SZ.v t) fv as values |-> supd sv (SZ.v t) fv;
  rewrite indices |-> S.upd si (SZ.v t) fi as indices |-> supd si (SZ.v t) fi;
}

inline_for_extraction noextract
fn process_chunk (input0 : array1 bf16 (l1_forward 151936) { is_global input0 })
  (values : A.array F32.t) (indices : A.array I64.t)
  (cvals : A.array F32.t) (cidx : A.array I64.t) (c : SZ.t)
  (#p : perm) (#s0 : chest1 bf16 151936)
  (#sv0 : erased (S.seq F32.t)) (#si0 : erased (S.seq I64.t))
  (#cv0 : erased (S.seq F32.t)) (#ci0 : erased (S.seq I64.t))
  preserves input0 |-> Frac p s0
  requires values |-> sv0 ** indices |-> si0 ** cvals |-> cv0 ** cidx |-> ci0 **
    pure (S.length sv0 == 256 /\ S.length si0 == 256 /\ S.length cv0 == 96 /\ S.length ci0 == 96 /\
          SZ.v c < 96 /\ chunks_pre (chest1_to_seq s0) cv0 ci0 (SZ.v c))
  ensures exists* sv si cv ci. values |-> sv ** indices |-> si ** cvals |-> cv ** cidx |-> ci **
    pure (S.length sv == 256 /\ S.length si == 256 /\ S.length cv == 96 /\ S.length ci == 96 /\
          chunks_pre (chest1_to_seq s0) cv ci (SZ.v c + 1))
{
  open Pulse.Lib.Array;
  let lim : SZ.t =
    if (c *^ 1600sz +^ 1600sz) <=^ nv_sz then 1600sz
    else if (c *^ 1600sz) <=^ nv_sz then nv_sz -^ c *^ 1600sz
    else 0sz;
  lanes_pre_empty (chest1_to_seq s0) (SZ.v c) sv0 si0;
  let mut t = 0sz;
  while (!t <^ 128sz)
    invariant live t ** pure (SZ.v !t <= 128)
    invariant exists* sv si. values |-> sv ** indices |-> si **
      pure (S.length sv == 256 /\ S.length si == 256 /\
            lanes_pre (chest1_to_seq s0) (SZ.v c) sv si (SZ.v !t))
    decreases (128 - SZ.v !t)
  {
    let vt = !t;
    with sv si. assert (values |-> sv ** indices |-> si);
    scan_lane input0 values indices c vt lim;
    with v i. assert (values |-> supd sv (SZ.v vt) v ** indices |-> supd si (SZ.v vt) i);
    lanes_pre_step (chest1_to_seq s0) (SZ.v c) sv si (SZ.v vt) v i;
    t := vt +^ 1sz;
  };
  with sv si. assert (values |-> sv ** indices |-> si);
  lanes_pre_done (chest1_to_seq s0) (SZ.v c) sv si;
  pow2_small ();
  reduce_block values indices 4sz 2sz (chest1_to_seq s0) (SZ.v c) (crank (SZ.v c) * 1664) 13 2;
  with sv1 si1. assert (values |-> sv1 ** indices |-> si1);
  let v0 = values.(0sz);
  let i0 = indices.(0sz);
  slot_dom (chest1_to_seq s0) (SZ.v c) (crank (SZ.v c) * 1664) (crank (SZ.v c) * 1664 + 1664) v0 i0;
  pow2_big ();
  let e : I64.t =
    if I64.lt i0 0L then (-1L)
    else I64.add (I64.mul (sz_to_i64 c) 4294967296L) i0;
  enc_ok (chest1_to_seq s0) (SZ.v c) (crank (SZ.v c) * 1664) (crank (SZ.v c) * 1664 + 1664) v0 i0 e;
  chunks_pre_step (chest1_to_seq s0) cv0 ci0 (SZ.v c) v0 e;
  cvals.(c) <- v0;
  cidx.(c) <- e;
}

inline_for_extraction noextract
fn final_stage (values : A.array F32.t) (indices : A.array I64.t)
  (cvals : A.array F32.t) (cidx : A.array I64.t) (s : erased (S.seq BF.t))
  (#sv0 : erased (S.seq F32.t)) (#si0 : erased (S.seq I64.t))
  (#cv : erased (S.seq F32.t)) (#ci : erased (S.seq I64.t))
  preserves cvals |-> cv ** cidx |-> ci
  requires values |-> sv0 ** indices |-> si0 **
    pure (S.length sv0 == 256 /\ S.length si0 == 256 /\ S.length cv == 96 /\ S.length ci == 96 /\
          S.length s == nV /\ chunks_pre s cv ci 96)
  ensures exists* sv si. values |-> sv ** indices |-> si **
    pure (S.length sv == 256 /\ S.length si == 256 /\
          slot_ok s (-1) 0 425984 (vat sv 0) (iat si 0))
{
  open Pulse.Lib.Array;
  let ninf : F32.t = neg #f32 infinity;
  chunks_pre_empty s sv0 si0;
  let mut t = 0sz;
  while (!t <^ 256sz)
    invariant live t ** pure (SZ.v !t <= 256)
    invariant exists* sv si. values |-> sv ** indices |-> si **
      pure (S.length sv == 256 /\ S.length si == 256 /\ chunks_pre s sv si (SZ.v !t))
    decreases (256 - SZ.v !t)
  {
    let vt = !t;
    with sv si. assert (values |-> sv ** indices |-> si);
    let v : F32.t = if (vt <^ 96sz) then cvals.(vt) else ninf;
    let i : I64.t = if (vt <^ 96sz) then cidx.(vt) else (-1L);
    fin_slot s cv ci (SZ.v vt) ninf v i;
    chunks_pre_step s sv si (SZ.v vt) v i;
    values.(vt) <- v;
    indices.(vt) <- i;
    t := vt +^ 1sz;
  };
  with sv si. assert (values |-> sv ** indices |-> si);
  chunks_pre_done s sv si;
  pow2_small ();
  reduce_block values indices 8sz 4sz s (-1) 0 1664 3;
}

#push-options "--fuel 1 --ifuel 1"
noextract
let copy_cell (o:chest1 I64.t 8) (j:nat{j < 8}) (x:I64.t) (k:nat{k < 8})
  : Lemma (acc1 (upd o (j, ()) x) k == (if k = j then x else acc1 o k))
  = ()
#pop-options

noextract
let copy_step (o:chest1 I64.t 8) (s1:chest1 I64.t 7) (j:nat{j < 7}) (x:I64.t)
  : Lemma (requires (forall (k:nat). k < 7 /\ k < j ==> acc1 o k == acc1 s1 k) /\ x == acc1 s1 j)
          (ensures forall (k:nat). k < 7 /\ k < j + 1 ==> acc1 (upd o (j, ()) x) k == acc1 s1 k)
  = let aux (k:nat{k < 7 /\ k < j + 1}) : Lemma (acc1 (upd o (j, ()) x) k == acc1 s1 k) = copy_cell o j x k in
    FStar.Classical.forall_intro aux

#push-options "--fuel 1 --ifuel 1"
noextract
let cell_n (#n:nat) (o:chest1 I64.t n) (j:nat{j < n}) (x:I64.t) (k:nat{k < n})
  : Lemma (acc1 (upd o (j, ()) x) k == (if k = j then x else acc1 o k))
  = ()
#pop-options

noextract
let post_ok (s0:chest1 bf16 151936) (s1:chest1 I64.t 7) (o0:chest1 I64.t 1) (o1:chest1 I64.t 8) (tok t7:I64.t)
  : Lemma (requires (forall (k:nat). k < 7 ==> acc1 o1 k == acc1 s1 k) /\
                    t7 == acc1 (upd o1 (7, ()) tok) 7 /\
                    tree_argmax (chest1_to_seq s0) (I64.v tok))
          (ensures optimized_argmax_relation (chest1_to_seq s0) (chest1_to_seq s1)
                     (chest1_to_seq (upd o0 (0, ()) t7)) (chest1_to_seq (upd o1 (7, ()) tok)))
  = let aux (k:nat{k < 8}) : Lemma (acc1 (upd o1 (7, ()) tok) k == (if k = 7 then tok else acc1 o1 k)) = cell_n o1 7 tok k in
    FStar.Classical.forall_intro aux;
    cell_n o1 7 tok 7;
    cell_n o0 0 t7 0

inline_for_extraction noextract
fn kf
  (input0 : array1 bf16 (l1_forward 151936) { is_global input0 })
  (input1 : array1 I64.t (l1_forward 7) { is_global input1 })
  (output0 : array1 I64.t (l1_forward 1) { is_global output0 })
  (output1 : array1 I64.t (l1_forward 8) { is_global output1 })
  (#input0_perm : perm) (#input1_perm : perm)
  (#input0_state : chest1 bf16 151936) (#input1_state : chest1 I64.t 7)
  (#output0_state : chest1 I64.t 1) (#output1_state : chest1 I64.t 8)
  preserves
    gpu **
    input0 |-> Frac input0_perm input0_state **
    input1 |-> Frac input1_perm input1_state
  requires
    output0 |-> output0_state ** output1 |-> output1_state
  ensures
    exists* (o0 : chest1 I64.t 1) (o1 : chest1 I64.t 8).
      output0 |-> o0 ** output1 |-> o1 **
      pure (optimized_argmax_relation (chest1_to_seq input0_state) (chest1_to_seq input1_state)
              (chest1_to_seq o0) (chest1_to_seq o1))
{
  open Pulse.Lib.Array;
  let ninf : F32.t = neg #f32 infinity;
  let mut values = [| ninf; 256sz |];
  let mut indices = [| (-1L); 256sz |];
  let mut cvals = [| ninf; 96sz |];
  let mut cidx = [| (-1L); 96sz |];
  let s : erased (S.seq BF.t) = hide (chest1_to_seq input0_state);
  with cv0. assert (cvals |-> cv0);
  with ci0. assert (cidx |-> ci0);
  chunks_pre_empty s cv0 ci0;
  let mut c = 0sz;
  while (!c <^ 96sz)
    invariant live c ** pure (SZ.v !c <= 96)
    invariant exists* sv si cv ci. values |-> sv ** indices |-> si ** cvals |-> cv ** cidx |-> ci **
      pure (S.length sv == 256 /\ S.length si == 256 /\ S.length cv == 96 /\ S.length ci == 96 /\
            chunks_pre s cv ci (SZ.v !c))
    decreases (96 - SZ.v !c)
  {
    let vc = !c;
    process_chunk input0 values indices cvals cidx vc #input0_perm #input0_state;
    c := vc +^ 1sz;
  };
  final_stage values indices cvals cidx s;
  with sv si. assert (values |-> sv ** indices |-> si);
  let v0 = values.(0sz);
  let e = indices.(0sz);
  final_ok s v0 e;
  slot_dom s (-1) 0 425984 v0 e;
  decode_ok e;
  let tok : I64.t =
    if I64.lt e 0L then 0L
    else I64.add (I64.mul (I64.div e 4294967296L) 1600L) (I64.rem e 4294967296L);
  let mut i = 0sz;
  while (!i <^ 7sz)
    invariant live i ** pure (SZ.v !i <= 7)
    invariant exists* (o1 : chest1 I64.t 8). output1 |-> o1 **
      pure (forall (k:nat). k < 7 /\ k < SZ.v !i ==> acc1 o1 k == acc1 input1_state k)
    decreases (7 - SZ.v !i)
  {
    let vi = !i;
    let x = Kuiper.Tensor.read input1 ((vi <: szlt 7), ()) #input1_perm;
    with o1. assert (output1 |-> o1);
    copy_step o1 input1_state (SZ.v vi) x;
    Kuiper.Tensor.write output1 (((vi <: szlt 7) <: szlt 8), ()) x;
    i := vi +^ 1sz;
  };
  with o0 o1. assert (output0 |-> o0 ** output1 |-> o1);
  Kuiper.Tensor.write output1 ((7sz <: szlt 8), ()) tok;
  let t7 = Kuiper.Tensor.read output1 ((7sz <: szlt 8), ());
  post_ok input0_state input1_state o0 o1 tok t7;
  Kuiper.Tensor.write output0 ((0sz <: szlt 1), ()) t7;
}

fn kplay_run
  (input0 : array1 bf16 (l1_forward 151936) { is_global input0 })
  (input1 : array1 I64.t (l1_forward 7) { is_global input1 })
  (output0 : array1 I64.t (l1_forward 1) { is_global output0 })
  (output1 : array1 I64.t (l1_forward 8) { is_global output1 })
  (#input0_perm : perm) (#input1_perm : perm)
  (#input0_state : chest1 bf16 151936) (#input1_state : chest1 I64.t 7)
  (#output0_state : chest1 I64.t 1) (#output1_state : chest1 I64.t 8)
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac input0_perm input0_state) **
    on gpu_loc (input1 |-> Frac input1_perm input1_state)
  requires
    on gpu_loc (output0 |-> output0_state) ** on gpu_loc (output1 |-> output1_state)
  ensures
    exists* (o0 : chest1 I64.t 1) (o1 : chest1 I64.t 8).
      on gpu_loc (output0 |-> o0) ** on gpu_loc (output1 |-> o1) **
      pure (optimized_argmax_relation (chest1_to_seq input0_state) (chest1_to_seq input1_state)
              (chest1_to_seq o0) (chest1_to_seq o1))
{
  launch_kernel_1
    (fun () ->
      kf input0 input1 output0 output1
        #input0_perm #input1_perm #input0_state #input1_state #output0_state #output1_state);
}
