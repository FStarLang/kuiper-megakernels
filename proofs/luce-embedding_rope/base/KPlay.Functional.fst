module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
open Kuiper.Float.Casts
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast
module R = FStar.Real
module Trig = Kuiper.Real.Trigonometry
module Lit = Kuiper.Float32.Literals
module CF = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates

#set-options "--z3rlimit 20 --fuel 1 --ifuel 1"

(*** RoPE coefficients: native lane and its approximation chain *)


(* The kernel's 32 configuration-only RoPE frequencies (exact binary32 values
   of the CUDA hexadecimal literals, spelled as their exact decimal expansions). *)
inline_for_extraction noextract
let freq (i:sz) : f32 =
  if i = 0sz then of_literal #f32 "1.0"
  else if i = 1sz then of_literal #f32 "0.604296386241912841796875"
  else if i = 2sz then of_literal #f32 "0.3651741445064544677734375"
  else if i = 3sz then of_literal #f32 "0.22067339718341827392578125"
  else if i = 4sz then of_literal #f32 "0.13335214555263519287109375"
  else if i = 5sz then of_literal #f32 "0.08058421313762664794921875"
  else if i = 6sz then of_literal #f32 "0.0486967526376247406005859375"
  else if i = 7sz then of_literal #f32 "0.02942727319896221160888671875"
  else if i = 8sz then of_literal #f32 "0.01778279431164264678955078125"
  else if i = 9sz then of_literal #f32 "0.01074607856571674346923828125"
  else if i = 10sz then of_literal #f32 "0.0064938166178762912750244140625"
  else if i = 11sz then of_literal #f32 "0.0039241896010935306549072265625"
  else if i = 12sz then of_literal #f32 "0.00237137381918728351593017578125"
  else if i = 13sz then of_literal #f32 "0.001433012657798826694488525390625"
  else if i = 14sz then of_literal #f32 "0.000865964335389435291290283203125"
  else if i = 15sz then of_literal #f32 "0.000523299095220863819122314453125"
  else if i = 16sz then of_literal #f32 "0.000316227786242961883544921875"
  else if i = 17sz then of_literal #f32 "0.000191095299669541418552398681640625"
  else if i = 18sz then of_literal #f32 "0.000115478193038143217563629150390625"
  else if i = 19sz then of_literal #f32 "0.000069783054641447961330413818359375"
  else if i = 20sz then of_literal #f32 "0.0000421696531702764332294464111328125"
  else if i = 21sz then of_literal #f32 "0.000025482966520939953625202178955078125"
  else if i = 22sz then of_literal #f32 "0.000015399264157167635858058929443359375"
  else if i = 23sz then of_literal #f32 "0.00000930572059587575495243072509765625"
  else if i = 24sz then of_literal #f32 "0.00000562341347176698036491870880126953125"
  else if i = 25sz then of_literal #f32 "0.00000339820826411596499383449554443359375"
  else if i = 26sz then of_literal #f32 "0.00000205352489501819945871829986572265625"
  else if i = 27sz then of_literal #f32 "0.00000124093776321387849748134613037109375"
  else if i = 28sz then of_literal #f32 "0.00000074989424092564149759709835052490234375"
  else if i = 29sz then of_literal #f32 "0.000000453158378377338522113859653472900390625"
  else if i = 30sz then of_literal #f32 "0.00000027384197665014653466641902923583984375"
  else of_literal #f32 "0.00000016548170833630138076841831207275390625"

let lit_0 () : Lemma (of_literal #f32 "1.0" %~ Lit.value "1.0") =
  assert_norm (Cert.exact_normal "1.0" 8388608 (-23));
  CF.literal_approx "1.0" 8388608 (-23)

let lit_1 () : Lemma (of_literal #f32 "0.604296386241912841796875" %~ Lit.value "0.604296386241912841796875") =
  assert_norm (Cert.exact_normal "0.604296386241912841796875" 10138411 (-24));
  CF.literal_approx "0.604296386241912841796875" 10138411 (-24)

let lit_2 () : Lemma (of_literal #f32 "0.3651741445064544677734375" %~ Lit.value "0.3651741445064544677734375") =
  assert_norm (Cert.exact_normal "0.3651741445064544677734375" 12253211 (-25));
  CF.literal_approx "0.3651741445064544677734375" 12253211 (-25)

let lit_3 () : Lemma (of_literal #f32 "0.22067339718341827392578125" %~ Lit.value "0.22067339718341827392578125") =
  assert_norm (Cert.exact_normal "0.22067339718341827392578125" 14809141 (-26));
  CF.literal_approx "0.22067339718341827392578125" 14809141 (-26)

let lit_4 () : Lemma (of_literal #f32 "0.13335214555263519287109375" %~ Lit.value "0.13335214555263519287109375") =
  assert_norm (Cert.exact_normal "0.13335214555263519287109375" 8949111 (-26));
  CF.literal_approx "0.13335214555263519287109375" 8949111 (-26)

let lit_5 () : Lemma (of_literal #f32 "0.08058421313762664794921875" %~ Lit.value "0.08058421313762664794921875") =
  assert_norm (Cert.exact_normal "0.08058421313762664794921875" 10815830 (-27));
  CF.literal_approx "0.08058421313762664794921875" 10815830 (-27)

let lit_6 () : Lemma (of_literal #f32 "0.0486967526376247406005859375" %~ Lit.value "0.0486967526376247406005859375") =
  assert_norm (Cert.exact_normal "0.0486967526376247406005859375" 13071935 (-28));
  CF.literal_approx "0.0486967526376247406005859375" 13071935 (-28)

let lit_7 () : Lemma (of_literal #f32 "0.02942727319896221160888671875" %~ Lit.value "0.02942727319896221160888671875") =
  assert_norm (Cert.exact_normal "0.02942727319896221160888671875" 15798647 (-29));
  CF.literal_approx "0.02942727319896221160888671875" 15798647 (-29)

let lit_8 () : Lemma (of_literal #f32 "0.01778279431164264678955078125" %~ Lit.value "0.01778279431164264678955078125") =
  assert_norm (Cert.exact_normal "0.01778279431164264678955078125" 9547065 (-29));
  CF.literal_approx "0.01778279431164264678955078125" 9547065 (-29)

let lit_9 () : Lemma (of_literal #f32 "0.01074607856571674346923828125" %~ Lit.value "0.01074607856571674346923828125") =
  assert_norm (Cert.exact_normal "0.01074607856571674346923828125" 11538514 (-30));
  CF.literal_approx "0.01074607856571674346923828125" 11538514 (-30)

let lit_10 () : Lemma (of_literal #f32 "0.0064938166178762912750244140625" %~ Lit.value "0.0064938166178762912750244140625") =
  assert_norm (Cert.exact_normal "0.0064938166178762912750244140625" 13945365 (-31));
  CF.literal_approx "0.0064938166178762912750244140625" 13945365 (-31)

let lit_11 () : Lemma (of_literal #f32 "0.0039241896010935306549072265625" %~ Lit.value "0.0039241896010935306549072265625") =
  assert_norm (Cert.exact_normal "0.0039241896010935306549072265625" 8427133 (-31));
  CF.literal_approx "0.0039241896010935306549072265625" 8427133 (-31)

let lit_12 () : Lemma (of_literal #f32 "0.00237137381918728351593017578125" %~ Lit.value "0.00237137381918728351593017578125") =
  assert_norm (Cert.exact_normal "0.00237137381918728351593017578125" 10184973 (-32));
  CF.literal_approx "0.00237137381918728351593017578125" 10184973 (-32)

let lit_13 () : Lemma (of_literal #f32 "0.001433012657798826694488525390625" %~ Lit.value "0.001433012657798826694488525390625") =
  assert_norm (Cert.exact_normal "0.001433012657798826694488525390625" 12309485 (-33));
  CF.literal_approx "0.001433012657798826694488525390625" 12309485 (-33)

let lit_14 () : Lemma (of_literal #f32 "0.000865964335389435291290283203125" %~ Lit.value "0.000865964335389435291290283203125") =
  assert_norm (Cert.exact_normal "0.000865964335389435291290283203125" 14877154 (-34));
  CF.literal_approx "0.000865964335389435291290283203125" 14877154 (-34)

let lit_15 () : Lemma (of_literal #f32 "0.000523299095220863819122314453125" %~ Lit.value "0.000523299095220863819122314453125") =
  assert_norm (Cert.exact_normal "0.000523299095220863819122314453125" 8990210 (-34));
  CF.literal_approx "0.000523299095220863819122314453125" 8990210 (-34)

let lit_16 () : Lemma (of_literal #f32 "0.000316227786242961883544921875" %~ Lit.value "0.000316227786242961883544921875") =
  assert_norm (Cert.exact_normal "0.000316227786242961883544921875" 10865504 (-35));
  CF.literal_approx "0.000316227786242961883544921875" 10865504 (-35)

let lit_17 () : Lemma (of_literal #f32 "0.000191095299669541418552398681640625" %~ Lit.value "0.000191095299669541418552398681640625") =
  assert_norm (Cert.exact_normal "0.000191095299669541418552398681640625" 13131969 (-36));
  CF.literal_approx "0.000191095299669541418552398681640625" 13131969 (-36)

let lit_18 () : Lemma (of_literal #f32 "0.000115478193038143217563629150390625" %~ Lit.value "0.000115478193038143217563629150390625") =
  assert_norm (Cert.exact_normal "0.000115478193038143217563629150390625" 15871202 (-37));
  CF.literal_approx "0.000115478193038143217563629150390625" 15871202 (-37)

let lit_19 () : Lemma (of_literal #f32 "0.000069783054641447961330413818359375" %~ Lit.value "0.000069783054641447961330413818359375") =
  assert_norm (Cert.exact_normal "0.000069783054641447961330413818359375" 9590910 (-37));
  CF.literal_approx "0.000069783054641447961330413818359375" 9590910 (-37)

let lit_20 () : Lemma (of_literal #f32 "0.0000421696531702764332294464111328125" %~ Lit.value "0.0000421696531702764332294464111328125") =
  assert_norm (Cert.exact_normal "0.0000421696531702764332294464111328125" 11591506 (-38));
  CF.literal_approx "0.0000421696531702764332294464111328125" 11591506 (-38)

let lit_21 () : Lemma (of_literal #f32 "0.000025482966520939953625202178955078125" %~ Lit.value "0.000025482966520939953625202178955078125") =
  assert_norm (Cert.exact_normal "0.000025482966520939953625202178955078125" 14009409 (-39));
  CF.literal_approx "0.000025482966520939953625202178955078125" 14009409 (-39)

let lit_22 () : Lemma (of_literal #f32 "0.000015399264157167635858058929443359375" %~ Lit.value "0.000015399264157167635858058929443359375") =
  assert_norm (Cert.exact_normal "0.000015399264157167635858058929443359375" 8465835 (-39));
  CF.literal_approx "0.000015399264157167635858058929443359375" 8465835 (-39)

let lit_23 () : Lemma (of_literal #f32 "0.00000930572059587575495243072509765625" %~ Lit.value "0.00000930572059587575495243072509765625") =
  assert_norm (Cert.exact_normal "0.00000930572059587575495243072509765625" 10231748 (-40));
  CF.literal_approx "0.00000930572059587575495243072509765625" 10231748 (-40)

let lit_24 () : Lemma (of_literal #f32 "0.00000562341347176698036491870880126953125" %~ Lit.value "0.00000562341347176698036491870880126953125") =
  assert_norm (Cert.exact_normal "0.00000562341347176698036491870880126953125" 12366017 (-41));
  CF.literal_approx "0.00000562341347176698036491870880126953125" 12366017 (-41)

let lit_25 () : Lemma (of_literal #f32 "0.00000339820826411596499383449554443359375" %~ Lit.value "0.00000339820826411596499383449554443359375") =
  assert_norm (Cert.exact_normal "0.00000339820826411596499383449554443359375" 14945478 (-42));
  CF.literal_approx "0.00000339820826411596499383449554443359375" 14945478 (-42)

let lit_26 () : Lemma (of_literal #f32 "0.00000205352489501819945871829986572265625" %~ Lit.value "0.00000205352489501819945871829986572265625") =
  assert_norm (Cert.exact_normal "0.00000205352489501819945871829986572265625" 9031498 (-42));
  CF.literal_approx "0.00000205352489501819945871829986572265625" 9031498 (-42)

let lit_27 () : Lemma (of_literal #f32 "0.00000124093776321387849748134613037109375" %~ Lit.value "0.00000124093776321387849748134613037109375") =
  assert_norm (Cert.exact_normal "0.00000124093776321387849748134613037109375" 10915404 (-43));
  CF.literal_approx "0.00000124093776321387849748134613037109375" 10915404 (-43)

let lit_28 () : Lemma (of_literal #f32 "0.00000074989424092564149759709835052490234375" %~ Lit.value "0.00000074989424092564149759709835052490234375") =
  assert_norm (Cert.exact_normal "0.00000074989424092564149759709835052490234375" 13192279 (-44));
  CF.literal_approx "0.00000074989424092564149759709835052490234375" 13192279 (-44)

let lit_29 () : Lemma (of_literal #f32 "0.000000453158378377338522113859653472900390625" %~ Lit.value "0.000000453158378377338522113859653472900390625") =
  assert_norm (Cert.exact_normal "0.000000453158378377338522113859653472900390625" 15944093 (-45));
  CF.literal_approx "0.000000453158378377338522113859653472900390625" 15944093 (-45)

let lit_30 () : Lemma (of_literal #f32 "0.00000027384197665014653466641902923583984375" %~ Lit.value "0.00000027384197665014653466641902923583984375") =
  assert_norm (Cert.exact_normal "0.00000027384197665014653466641902923583984375" 9634958 (-45));
  CF.literal_approx "0.00000027384197665014653466641902923583984375" 9634958 (-45)

let lit_31 () : Lemma (of_literal #f32 "0.00000016548170833630138076841831207275390625" %~ Lit.value "0.00000016548170833630138076841831207275390625") =
  assert_norm (Cert.exact_normal "0.00000016548170833630138076841831207275390625" 11644740 (-46));
  CF.literal_approx "0.00000016548170833630138076841831207275390625" 11644740 (-46)

let freq_approx (i:szlt 32)
  : Lemma (freq i %~ Lit.value (base_freq_literal (SZ.v i))) =
  if i = 0sz then lit_0 ()
  else if i = 1sz then lit_1 ()
  else if i = 2sz then lit_2 ()
  else if i = 3sz then lit_3 ()
  else if i = 4sz then lit_4 ()
  else if i = 5sz then lit_5 ()
  else if i = 6sz then lit_6 ()
  else if i = 7sz then lit_7 ()
  else if i = 8sz then lit_8 ()
  else if i = 9sz then lit_9 ()
  else if i = 10sz then lit_10 ()
  else if i = 11sz then lit_11 ()
  else if i = 12sz then lit_12 ()
  else if i = 13sz then lit_13 ()
  else if i = 14sz then lit_14 ()
  else if i = 15sz then lit_15 ()
  else if i = 16sz then lit_16 ()
  else if i = 17sz then lit_17 ()
  else if i = 18sz then lit_18 ()
  else if i = 19sz then lit_19 ()
  else if i = 20sz then lit_20 ()
  else if i = 21sz then lit_21 ()
  else if i = 22sz then lit_22 ()
  else if i = 23sz then lit_23 ()
  else if i = 24sz then lit_24 ()
  else if i = 25sz then lit_25 ()
  else if i = 26sz then lit_26 ()
  else if i = 27sz then lit_27 ()
  else if i = 28sz then lit_28 ()
  else if i = 29sz then lit_29 ()
  else if i = 30sz then lit_30 ()
  else lit_31 ()

inline_for_extraction noextract
let angle (t:I64.t) (i:sz) : f32 = mul #f32 (of_int #f32 t) (freq i)

let ideal_angle (t:int) (i:nat) : real = R.of_int t *. at inv_freq_f32 i

let inv_freq_at (i:nat{i < 32})
  : Lemma (at inv_freq_f32 i == Lit.value (base_freq_literal i)) = ()

let of_int_approx (t:I64.t)
  : Lemma (of_int #f32 t %~ R.of_int (I64.v t)) =
  Kuiper.Approximates.Base.approx_of_int #f32;
  approx_apply #I64.t #I64.t #f32 #real (of_int #f32) Kuiper.Float.Realops.r_of_int t t

let angle_approx (t:I64.t) (i:szlt 32)
  : Lemma (angle t i %~ ideal_angle (I64.v t) (SZ.v i)) =
  freq_approx i;
  inv_freq_at (SZ.v i);
  of_int_approx t;
  a_mul (of_int #f32 t) (freq i) (R.of_int (I64.v t)) (Lit.value (base_freq_literal (SZ.v i)))

let cos_lane (t:I64.t) (i:sz) : bf16 = fcast #f32 #bf16 (cos #f32 (angle t i))
let sin_lane (t:I64.t) (i:sz) : bf16 = fcast #f32 #bf16 (sin #f32 (angle t i))

let lane_approx (t:I64.t) (i:szlt 32)
  : Lemma (cos_lane t i %~ S.index (rope_cos base_numerics (I64.v t)) (SZ.v i) /\
           sin_lane t i %~ S.index (rope_sin base_numerics (I64.v t)) (SZ.v i)) =
  angle_approx t i;
  Kuiper.Float32.cos_approx (angle t i) (ideal_angle (I64.v t) (SZ.v i));
  Kuiper.Float32.sin_approx (angle t i) (ideal_angle (I64.v t) (SZ.v i))


(*** Residual row: exact datum copy *)

(* Domain premise of base_embedding_rope_relation. *)
let premise (stab semb:seq bf16) (stok:seq I64.t) (t:int) : prop =
  let pP = S.length semb / 1024 in
  S.length stab == table_len /\ S.length semb == pP * 1024 /\ 1 <= pP /\
  1 <= S.length stok /\ 0 <= t /\ t < pP + S.length stok - 1 /\ t < 12 /\
  (t >= pP ==> valid_token (tok_at stok (t - pP)))

let residual_ok (stab semb:seq bf16) (stok:seq I64.t) (t:int) (r:seq bf16) : prop =
  premise stab semb stok t ==>
  (S.length r == 1024 /\
   (forall (i:nat). i < 1024 ==>
      S.index r i ==
        (let pP = S.length semb / 1024 in
         if t < pP then S.index semb (t * 1024 + i)
         else S.index stab (tok_at stok (t - pP) * 1024 + i))))

let rope_ok (t:int) (c s:seq bf16) : prop =
  c %~ rope_cos base_numerics t /\ s %~ rope_sin base_numerics t

let assemble (stab semb:seq bf16) (stok:seq I64.t) (t:int) (r c s:seq bf16)
  : Lemma (requires residual_ok stab semb stok t r /\ rope_ok t c s /\
                    S.length c == 32 /\ S.length s == 32)
          (ensures base_embedding_rope_relation stab semb stok t r c s) = ()

(* The send instance for global array contents used by launch_kernel_1. *)
instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

inline_for_extraction noextract
fn copy_row
  (#n : erased nat)
  (src : larray bf16 n)
  (off : sz)
  (dst : larray bf16 1024)
  (#p : perm)
  (#s : erased (seq bf16))
  (#d0 : erased (seq bf16))
  preserves gpu ** src |-> Frac p s
  requires dst |-> d0 ** pure (SZ.v off + 1024 <= n)
  ensures exists* (d : seq bf16).
    dst |-> d **
    pure (S.length d == 1024 /\
          (forall (k:nat). k < 1024 ==>
             SZ.v off + k < S.length s /\ S.index d k == S.index s (SZ.v off + k)))
{
  pts_to_len src;
  pts_to_len dst;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (j : sz) (d : seq bf16).
      i |-> j ** dst |-> d **
      pure (SZ.v j <= 1024 /\ S.length d == 1024 /\
            (forall (k:nat). k < SZ.v j ==>
               SZ.v off + k < S.length s /\ S.index d k == S.index s (SZ.v off + k)))
    decreases (1024 - SZ.v !i)
  {
    let j = !i;
    let v = src.(off +^ j);
    dst.(j) <- v;
    i := j +^ 1sz;
  };
}

inline_for_extraction noextract
fn fill_residual
  (tab_len emb_len tok_len : sz)
  (table : larray bf16 (SZ.v tab_len))
  (emb : larray bf16 (SZ.v emb_len))
  (tokens : larray I64.t (SZ.v tok_len))
  (t : I64.t)
  (residual : larray bf16 1024)
  (#ptab #pemb #ptok : perm)
  (#stab #semb : erased (seq bf16))
  (#stok : erased (seq I64.t))
  (#r0 : erased (seq bf16))
  preserves gpu **
    table |-> Frac ptab stab **
    emb |-> Frac pemb semb **
    tokens |-> Frac ptok stok
  requires residual |-> r0
  ensures exists* (r : seq bf16).
    residual |-> r ** pure (residual_ok stab semb stok (I64.v t) r)
{
  pts_to_len table;
  pts_to_len emb;
  pts_to_len tokens;
  let p = emb_len /^ 1024sz;
  if (I64.lte 0L t && I64.lt t 4194304L) {
    let ts = SZ.uint32_to_sizet (Cast.int64_to_uint32 t);
    if (ts <^ p) {
      copy_row emb (ts *^ 1024sz) residual;
    } else {
      let d = ts -^ p;
      if (d <^ tok_len) {
        let tok = tokens.(d);
        if (I64.lte 0L tok && I64.lt tok 248320L) {
          let tk = SZ.uint32_to_sizet (Cast.int64_to_uint32 tok);
          if (tk *^ 1024sz +^ 1024sz <=^ tab_len) {
            copy_row table (tk *^ 1024sz) residual;
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
  } else {
    ()
  }
}

inline_for_extraction noextract
fn fill_rope
  (t : I64.t)
  (cos_out : larray bf16 32)
  (sin_out : larray bf16 32)
  (#c0 #s0 : erased (seq bf16))
  preserves gpu
  requires cos_out |-> c0 ** sin_out |-> s0
  ensures exists* (c s : seq bf16).
    cos_out |-> c ** sin_out |-> s **
    pure (S.length c == 32 /\ S.length s == 32 /\ rope_ok (I64.v t) c s)
{
  pts_to_len cos_out;
  pts_to_len sin_out;
  let mut i = 0sz;
  while (!i <^ 32sz)
    invariant exists* (j : sz) (c s : seq bf16).
      i |-> j ** cos_out |-> c ** sin_out |-> s **
      pure (SZ.v j <= 32 /\ S.length c == 32 /\ S.length s == 32 /\
            (forall (k:nat). k < SZ.v j ==>
               S.index c k %~ S.index (rope_cos base_numerics (I64.v t)) k /\
               S.index s k %~ S.index (rope_sin base_numerics (I64.v t)) k))
    decreases (32 - SZ.v !i)
  {
    let j = !i;
    lane_approx t j;
    let a = angle t j;
    cos_out.(j) <- fcast #f32 #bf16 (cos #f32 a);
    sin_out.(j) <- fcast #f32 #bf16 (sin #f32 a);
    i := j +^ 1sz;
  };
}

inline_for_extraction noextract
fn kf
  (tab_len emb_len tok_len : sz)
  (table : larray bf16 (SZ.v tab_len))
  (emb : larray bf16 (SZ.v emb_len))
  (tokens : larray I64.t (SZ.v tok_len))
  (t : I64.t)
  (residual : larray bf16 1024)
  (cos_out : larray bf16 32)
  (sin_out : larray bf16 32)
  (#ptab #pemb #ptok : perm)
  (#stab #semb : erased (seq bf16))
  (#stok : erased (seq I64.t))
  (#r0 #c0 #s0 : erased (seq bf16))
  preserves gpu **
    table |-> Frac ptab stab **
    emb |-> Frac pemb semb **
    tokens |-> Frac ptok stok
  requires residual |-> r0 ** cos_out |-> c0 ** sin_out |-> s0
  ensures exists* (r c s : seq bf16).
    residual |-> r ** cos_out |-> c ** sin_out |-> s **
    pure (base_embedding_rope_relation stab semb stok (I64.v t) r c s)
{
  fill_residual tab_len emb_len tok_len table emb tokens t residual;
  fill_rope t cos_out sin_out;
  with r. assert residual |-> r;
  with c s. assert cos_out |-> c ** sin_out |-> s;
  assemble stab semb stok (I64.v t) r c s;
}

fn kplay_run
  (tab_len emb_len tok_len : sz)
  (table : larray bf16 (SZ.v tab_len) { is_global table })
  (emb : larray bf16 (SZ.v emb_len) { is_global emb })
  (tokens : larray I64.t (SZ.v tok_len) { is_global tokens })
  (t : I64.t)
  (residual : larray bf16 1024 { is_global residual })
  (cos_out : larray bf16 32 { is_global cos_out })
  (sin_out : larray bf16 32 { is_global sin_out })
  (#ptab #pemb #ptok : perm)
  (#stab #semb : erased (seq bf16))
  (#stok : erased (seq I64.t))
  (#r0 #c0 #s0 : erased (seq bf16))
  preserves
    cpu **
    on gpu_loc (table |-> Frac ptab stab) **
    on gpu_loc (emb |-> Frac pemb semb) **
    on gpu_loc (tokens |-> Frac ptok stok)
  requires
    on gpu_loc (residual |-> r0) **
    on gpu_loc (cos_out |-> c0) **
    on gpu_loc (sin_out |-> s0)
  ensures
    exists* (r c s : seq bf16).
      on gpu_loc (residual |-> r) **
      on gpu_loc (cos_out |-> c) **
      on gpu_loc (sin_out |-> s) **
      pure (base_embedding_rope_relation stab semb stok (I64.v t) r c s)
{
  launch_kernel_1
    (fun () ->
      kf tab_len emb_len tok_len table emb tokens t residual cos_out sin_out
        #ptab #pemb #ptok #stab #semb #stok #r0 #c0 #s0);
}
