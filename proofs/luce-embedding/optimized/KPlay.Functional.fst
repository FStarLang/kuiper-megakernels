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
module Trig = Kuiper.Real.Trigonometry
module Casts = Kuiper.Float.Casts
module R = FStar.Real
module Lit = Kuiper.Float32.Literals
module CF = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates

(*** RoPE inverse frequencies *)

#push-options "--z3rlimit 20 --fuel 1 --ifuel 1"

(* PyTorch's 32 float32 RoPE inverse frequencies (exact binary32 values of the
   CUDA table, spelled as their exact decimal expansions). *)
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

#pop-options

let freq_r (p:int) (i:nat) : real = R.of_int p *. at inv_freq_f32 i

let freq_ok (p:int) (i:nat{i < 32})
  : Lemma (S.index (rope_cos opt_numerics p) i == Trig.cos (freq_r p i) /\
           S.index (rope_sin opt_numerics p) i == Trig.sin (freq_r p i))
  = ()

let of_int_ok (x:I64.t)
  : Lemma (of_int #f32 x %~ R.of_int (I64.v x))
  = approx_of_int #f32;
    approx_apply (of_int #f32) RO.r_of_int x x

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
fn rope_freq (pf : f32) (pr : erased int) (i : SZ.t { SZ.v i < 32 })
  preserves gpu
  requires pure (pf %~ R.of_int pr)
  returns r : f32
  ensures pure (r %~ freq_r pr (SZ.v i))
{
  freq_approx i;
  F32.mul_rn_ftz pf (freq i)
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
  (* Runtime start index (0 whenever the relation's premise holds): keeps the
     rotary loop from being unrolled and constant-folded. *)
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
    let fr = rope_freq pf (I64.v position) vi;
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
