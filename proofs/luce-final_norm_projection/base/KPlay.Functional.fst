module KPlay.Functional

#lang-pulse

open Kuiper.TensorCore
open Kuiper.Tensor
open Kuiper.Tensor.Tiling
open Kuiper.Tensor.Layout.Alg { l2_row_major, l2_col_major }
open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF = Kuiper.BFloat16
module Casts = Kuiper.Float.Casts
module FL = Kuiper.Floating.Base
module MMA = Kuiper.TensorCore.MMA
module GEMM = Kuiper.Spec.GEMM
module Lit = Kuiper.Float32.Literals
module Cert = Kuiper.Float32.ConstantFold.Certificates
module CF = Kuiper.Float32.ConstantFold

inline_for_extraction noextract instance c1024 : concrete_sz 1024 = { x = 1024sz; }

(*** Native ghost model *)

let wlay = l2_row_major 248320 1024
let blay = l2_col_major 1024 8
let clay = l2_row_major 248320 8

let fit (#a:Type) (n:nat) (d:a) (s:S.seq a) : lseq a n =
  if S.length s = n then s else S.create n d

let wmat (s:S.seq BF.t) : chest2 BF.t 248320 1024 = from_seq wlay (fit 254279680 zero s)
let bmat (s:S.seq BF.t) : chest2 BF.t 1024 8 = from_seq blay (fit 8192 zero s)
let cmat (s:S.seq F32.t) : chest2 F32.t 248320 8 = from_seq clay (fit 1986560 zero s)

let rec acc_tile (ew : chest2 BF.t 248320 1024) (eb : chest2 BF.t 1024 8)
  (rb : natlt 15520) (p : nat{p <= 64})
  : Tot (chest2 F32.t 16 8) (decreases p)
  = if p = 0 then const (16 @| 8 @| INil) (zero #F32.t)
    else MMA.emma (acc_tile ew eb rb (p - 1))
              (ematrix_subtile ew 16 16 rb (p - 1))
              (ematrix_subtile eb 16 8 (p - 1) 0)

let rec cfin (ec : chest2 F32.t 248320 8) (ew : chest2 BF.t 248320 1024) (eb : chest2 BF.t 1024 8)
  (r : nat{r <= 15520})
  : Tot (chest2 F32.t 248320 8) (decreases r)
  = if r = 0 then ec
    else update_tile (cfin ec ew eb (r - 1)) 16 8 (r - 1) 0 (acc_tile ew eb (r - 1) 64)

let cfin' ec ew eb (r : nat) : chest2 F32.t 248320 8 =
  if r <= 15520 then cfin ec ew eb r else ec

let acc' ew eb (rb p : nat) : chest2 F32.t 16 8 =
  if rb < 15520 && p <= 64 then acc_tile ew eb rb p else const (16 @| 8 @| INil) (zero #F32.t)

inline_for_extraction noextract
let widen (x:BF.t) : F32.t = Casts.fcast #BF.t #F32.t x

inline_for_extraction noextract
let narrow (x:F32.t) : BF.t = Casts.fcast #F32.t #BF.t x

inline_for_extraction noextract
let eps_lit () : F32.t = FL.of_literal #F32.t "0.0000009999999974752427078783512115478515625"

inline_for_extraction noextract
let inv1024_lit () : F32.t = FL.of_literal #F32.t "0.0009765625"

#set-options "--z3rlimit 40"

(*** Proof support *)

#push-options "--z3rlimit 20"

(** Literal certificates *)

#push-options "--fuel 2 --ifuel 1"
noextract
let eps_fraction () : Lemma (Lit.fraction ['0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) =
  assert (Lit.fraction [] 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['5'] 999999997475242707878351211547851562 1000000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['2';'5'] 99999999747524270787835121154785156 100000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['6';'2';'5'] 9999999974752427078783512115478515 10000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['5';'6';'2';'5'] 999999997475242707878351211547851 1000000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['1';'5';'6';'2';'5'] 99999999747524270787835121154785 100000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['5';'1';'5';'6';'2';'5'] 9999999974752427078783512115478 10000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351211547 1000000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121154 100000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512115 10000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351211 1000000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835121 100000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783512 10000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878351 1000000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787835 100000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078783 10000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707878 1000000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270787 100000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427078 10000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242707 1000000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524270 100000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752427 10000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475242 1000000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747524 100000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974752 10000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997475 1000000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999747 100000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999974 10000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999997 1000000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999999 100000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999999 10000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999999 1000000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99999 100000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9999 10000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 999 1000000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 99 100000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 9 10000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1000 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 100 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 10 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  assert (Lit.fraction ['0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'] 0 1 == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000));
  ()
#pop-options
noextract
let eps_list () : Lemma (FStar.String.list_of_string "0.0000009999999974752427078783512115478515625" == ['0';'.';'0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5']) = ()
#push-options "--fuel 3 --ifuel 1"
noextract
let integral_dot (l:list FStar.Char.char{Cons? l}) : Lemma (Lit.integral ('.'::l) 0 true == Lit.fraction l 0 1) = ()
noextract
let integral_zero (l:list FStar.Char.char) : Lemma (Lit.integral ('0'::l) 0 false == Lit.integral l 0 true) = ()
noextract
let eps_integral () : Lemma (Lit.integral (FStar.String.list_of_string "0.0000009999999974752427078783512115478515625") 0 false == Some (9999999974752427078783512115478515625, 10000000000000000000000000000000000000000000)) =
  eps_list (); eps_fraction ();
  integral_zero ['.';'0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5'];
  integral_dot ['0';'0';'0';'0';'0';'0';'9';'9';'9';'9';'9';'9';'9';'9';'7';'4';'7';'5';'2';'4';'2';'7';'0';'7';'8';'7';'8';'3';'5';'1';'2';'1';'1';'5';'4';'7';'8';'5';'1';'5';'6';'2';'5']
#pop-options
#push-options "--fuel 1 --ifuel 0"
noextract
let bp43 () : Lemma (Lit.binary_power 43 == 8796093022208 /\ Lit.binary_power 33 == 8589934592) =
  assert (Lit.binary_power 0 == 1);
  assert (Lit.binary_power 1 == 2);
  assert (Lit.binary_power 2 == 4);
  assert (Lit.binary_power 3 == 8);
  assert (Lit.binary_power 4 == 16);
  assert (Lit.binary_power 5 == 32);
  assert (Lit.binary_power 6 == 64);
  assert (Lit.binary_power 7 == 128);
  assert (Lit.binary_power 8 == 256);
  assert (Lit.binary_power 9 == 512);
  assert (Lit.binary_power 10 == 1024);
  assert (Lit.binary_power 11 == 2048);
  assert (Lit.binary_power 12 == 4096);
  assert (Lit.binary_power 13 == 8192);
  assert (Lit.binary_power 14 == 16384);
  assert (Lit.binary_power 15 == 32768);
  assert (Lit.binary_power 16 == 65536);
  assert (Lit.binary_power 17 == 131072);
  assert (Lit.binary_power 18 == 262144);
  assert (Lit.binary_power 19 == 524288);
  assert (Lit.binary_power 20 == 1048576);
  assert (Lit.binary_power 21 == 2097152);
  assert (Lit.binary_power 22 == 4194304);
  assert (Lit.binary_power 23 == 8388608);
  assert (Lit.binary_power 24 == 16777216);
  assert (Lit.binary_power 25 == 33554432);
  assert (Lit.binary_power 26 == 67108864);
  assert (Lit.binary_power 27 == 134217728);
  assert (Lit.binary_power 28 == 268435456);
  assert (Lit.binary_power 29 == 536870912);
  assert (Lit.binary_power 30 == 1073741824);
  assert (Lit.binary_power 31 == 2147483648);
  assert (Lit.binary_power 32 == 4294967296);
  assert (Lit.binary_power 33 == 8589934592);
  assert (Lit.binary_power 34 == 17179869184);
  assert (Lit.binary_power 35 == 34359738368);
  assert (Lit.binary_power 36 == 68719476736);
  assert (Lit.binary_power 37 == 137438953472);
  assert (Lit.binary_power 38 == 274877906944);
  assert (Lit.binary_power 39 == 549755813888);
  assert (Lit.binary_power 40 == 1099511627776);
  assert (Lit.binary_power 41 == 2199023255552);
  assert (Lit.binary_power 42 == 4398046511104);
  assert (Lit.binary_power 43 == 8796093022208);
  ()
#pop-options
#push-options "--fuel 14 --ifuel 1"
noextract
let inv_integral () : Lemma (Lit.integral (FStar.String.list_of_string "0.0009765625") 0 false == Some (9765625, 10000000000)) = ()
#pop-options

noextract
unfold let eps_s : string = "0.0000009999999974752427078783512115478515625"
noextract
unfold let inv_s : string = "0.0009765625"
noextract
let inv_list () : Lemma (FStar.String.list_of_string "0.0009765625" == ['0';'.';'0';'0';'0';'9';'7';'6';'5';'6';'2';'5']) = ()
noextract
let eps_m1 () : Lemma (9999999974752427078783512115478515625 * 8796093022208 == 87960930000000000000000000000000000000000000000000) = ()
noextract
let eps_m2 () : Lemma (8796093 * 1 * 10000000000000000000000000000000000000000000 == 87960930000000000000000000000000000000000000000000) = ()
noextract
let inv_m1 () : Lemma (9765625 * 8589934592 == 83886080000000000) = ()
noextract
let inv_m2 () : Lemma (8388608 * 1 * 10000000000 == 83886080000000000) = ()
noextract
let exact_normal_intro (s:string) (m:nat) (e:int) (n:nat) (d:pos) (un ud:pos)
  : Lemma (requires Cert.normal_encoding m e /\
                    Lit.integral (FStar.String.list_of_string s) 0 false == Some (n, d) /\
                    Cert.unit_fraction e == (un, ud) /\ n * ud == m * un * d)
          (ensures Cert.exact_normal s m e) = ()
noextract
let eps_cert () : Lemma (Cert.exact_normal eps_s 8796093 (-43)) =
  eps_integral (); bp43 ();
  eps_m1 (); eps_m2 ();
  exact_normal_intro eps_s 8796093 (-43) 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000 1 8796093022208
noextract
let inv_cert () : Lemma (Cert.exact_normal inv_s 8388608 (-33)) =
  inv_integral (); bp43 ();
  inv_m1 (); inv_m2 ();
  exact_normal_intro inv_s 8388608 (-33) 9765625 10000000000 1 8589934592
noextract
let value_intro (s:string) (n:nat) (d:pos)
  : Lemma (requires Lit.integral (FStar.String.list_of_string s) 0 false == Some (n, d))
          (ensures Lit.value s == of_int n /. of_int d) = ()
noextract
let eps_value () : Lemma (Lit.value eps_s >. 0.0R) =
  eps_integral ();
  value_intro eps_s 9999999974752427078783512115478515625 10000000000000000000000000000000000000000000
noextract
let inv_value () : Lemma (Lit.value inv_s == 1.0R /. 1024.0R) =
  inv_integral ();
  value_intro inv_s 9765625 10000000000

(** Real sums *)

noextract
let rec rs (f : nat -> real) (n : nat) : Tot real (decreases n) =
  if n = 0 then 0.0R else rs f (n - 1) +. f (n - 1)

noextract
let rec rs_ext (f g : nat -> real) (n : nat)
  : Lemma (requires forall (k:nat). k < n ==> f k == g k)
          (ensures rs f n == rs g n) (decreases n)
  = if n = 0 then () else rs_ext f g (n - 1)

noextract
let shift (f : nat -> real) (a : nat) (k : nat) : real = f (a + k)

noextract
let rec rs_split (f : nat -> real) (a b : nat)
  : Lemma (ensures rs f (a + b) == rs f a +. rs (shift f a) b) (decreases b)
  = if b = 0 then () else rs_split f a (b - 1)

noextract
let rec rs_add (f g : nat -> real) (n : nat)
  : Lemma (ensures rs (fun k -> f k +. g k) n == rs f n +. rs g n) (decreases n)
  = if n = 0 then () else rs_add f g (n - 1)

noextract
let rec rs_nonneg (f : nat -> real) (n : nat)
  : Lemma (requires forall (k:nat). k < n ==> f k >=. 0.0R)
          (ensures rs f n >=. 0.0R) (decreases n)
  = if n = 0 then () else rs_nonneg f (n - 1)

noextract
let blk (f : nat -> real) (b : nat) (t : nat) : real = rs (shift f (b * t)) b

noextract
let rec rs_block (f : nat -> real) (b n : nat)
  : Lemma (ensures rs f (b * n) == rs (blk f b) n) (decreases n)
  = if n = 0 then ()
    else begin
      rs_block f b (n - 1);
      FStar.Math.Lemmas.distributivity_sub_right b n 1;
      assert (b * n == b * (n - 1) + b);
      rs_split f (b * (n - 1)) b
    end

noextract
let ev (g : nat -> real) (t : nat) : real = g (2 * t)
noextract
let od (g : nat -> real) (t : nat) : real = g (2 * t + 1)

noextract
let rec rs_interleave (g : nat -> real) (n : nat)
  : Lemma (ensures rs g (2 * n) == rs (ev g) n +. rs (od g) n)
          (decreases n)
  = if n = 0 then ()
    else begin
      rs_interleave g (n - 1);
      assert (rs g (2 * n) == rs g (2 * n - 1) +. g (2 * n - 1));
      assert (rs g (2 * n - 1) == rs g (2 * (n - 1)) +. g (2 * (n - 1)))
    end

noextract
let rsum_singleton (x : real)
  : Lemma (rsum (S.create 1 x) == x)
  = let Kuiper.Seq.Common.SCons hd tl = Kuiper.Seq.Common.view_seq (S.create 1 x) in
    assert (S.equal tl (S.empty #real))

noextract
let rec rsum_rs (s : S.seq real) (f : nat -> real)
  : Lemma (requires forall (i:nat). i < S.length s ==> S.index s i == f i)
          (ensures rsum s == rs f (S.length s)) (decreases S.length s)
  = let n = S.length s in
    if n = 0 then begin
      assert (S.equal s (S.empty #real));
      let Kuiper.Seq.Common.SNil = Kuiper.Seq.Common.view_seq s in ()
    end else begin
      let s1 = S.slice s 0 (n - 1) in
      rsum_rs s1 f;
      assert (S.equal s (S.append s1 (S.create 1 (S.index s (n - 1)))));
      rsum_append s1 (S.create 1 (S.index s (n - 1)));
      rsum_singleton (S.index s (n - 1))
    end

noextract
let vsum_rs (n : nat) (f : nat -> real)
  : Lemma (ensures vsum n f == rs f n)
  = let s = S.init n (fun (k:nat{k < n}) -> f k) in
    assert (vsum n f == rsum s);
    rsum_rs s f

(** Summation helpers *)

noextract
let wx (sr : S.seq BF.t) (i : nat) : F32.t =
  if i < S.length sr then widen (S.index sr i) else zero

noextract
let p0f (sr : S.seq BF.t) (c : nat) : F32.t =
  let x0 = wx sr (4 * c) in
  let x1 = wx sr (4 * c + 1) in
  let x2 = wx sr (4 * c + 2) in
  let x3 = wx sr (4 * c + 3) in
  add (add (add (add zero (mul x0 x0)) (mul x1 x1)) (mul x2 x2)) (mul x3 x3)

noextract
let rec trf (sr : S.seq BF.t) (k : nat{k <= 8}) (i : nat) : Tot F32.t (decreases k) =
  if k = 0 then p0f sr i
  else if i < pow2 (8 - k) then add (trf sr (k - 1) i) (trf sr (k - 1) (i + pow2 (8 - k)))
  else trf sr (k - 1) i

noextract
let invf (sr : S.seq BF.t) : F32.t =
  rsqrt (add (mul (trf sr 8 0) (inv1024_lit ())) (eps_lit ()))

noextract
let nrm (sn sr : S.seq BF.t) (k : nat) : BF.t =
  narrow (mul (mul (wx sr k) (invf sr)) (add one (wx sn k)))

noextract
let sqf (rx : S.seq real) (k : nat) : real = at rx k *. at rx k

noextract
let rp0 (rx : S.seq real) (c : nat) : real =
  (((0.0R +. sqf rx (4 * c)) +. sqf rx (4 * c + 1)) +. sqf rx (4 * c + 2)) +. sqf rx (4 * c + 3)

noextract
let rec rtr (rx : S.seq real) (k : nat{k <= 8}) (i : nat) : Tot real (decreases k) =
  if k = 0 then rp0 rx i
  else if i < pow2 (8 - k) then rtr rx (k - 1) i +. rtr rx (k - 1) (i + pow2 (8 - k))
  else rtr rx (k - 1) i

noextract
let wx_approx (sr : S.seq BF.t) (rx : S.seq real) (i : nat)
  : Lemma (requires sr %~ rx) (ensures wx sr i %~ at rx i)
  = if i < S.length sr then Casts.fcast_approx #BF.t #F32.t (S.index sr i) (S.index rx i)
    else a0 #F32.t

noextract
let p0f_approx (sr : S.seq BF.t) (rx : S.seq real) (c : nat)
  : Lemma (requires sr %~ rx) (ensures p0f sr c %~ rp0 rx c)
  = let x0 = wx sr (4 * c) in
    let x1 = wx sr (4 * c + 1) in
    let x2 = wx sr (4 * c + 2) in
    let x3 = wx sr (4 * c + 3) in
    let r0 = at rx (4 * c) in
    let r1 = at rx (4 * c + 1) in
    let r2 = at rx (4 * c + 2) in
    let r3 = at rx (4 * c + 3) in
    wx_approx sr rx (4 * c); wx_approx sr rx (4 * c + 1);
    wx_approx sr rx (4 * c + 2); wx_approx sr rx (4 * c + 3);
    a0 #F32.t;
    a_mul x0 x0 r0 r0; a_mul x1 x1 r1 r1; a_mul x2 x2 r2 r2; a_mul x3 x3 r3 r3;
    a_add (zero #F32.t) (mul x0 x0) 0.0R (r0 *. r0);
    a_add (add zero (mul x0 x0)) (mul x1 x1) (0.0R +. r0 *. r0) (r1 *. r1);
    a_add (add (add zero (mul x0 x0)) (mul x1 x1)) (mul x2 x2)
          ((0.0R +. r0 *. r0) +. r1 *. r1) (r2 *. r2);
    a_add (add (add (add zero (mul x0 x0)) (mul x1 x1)) (mul x2 x2)) (mul x3 x3)
          (((0.0R +. r0 *. r0) +. r1 *. r1) +. r2 *. r2) (r3 *. r3)

noextract
let rec trf_approx (sr : S.seq BF.t) (rx : S.seq real) (k : nat{k <= 8}) (i : nat)
  : Lemma (requires sr %~ rx) (ensures trf sr k i %~ rtr rx k i) (decreases k)
  = if k = 0 then p0f_approx sr rx i
    else begin
      let o = pow2 (8 - k) in
      trf_approx sr rx (k - 1) i;
      if i < o then begin
        trf_approx sr rx (k - 1) (i + o);
        a_add (trf sr (k - 1) i) (trf sr (k - 1) (i + o)) (rtr rx (k - 1) i) (rtr rx (k - 1) (i + o))
      end
    end

(** Tree sum equals the squared sum *)

noextract
let strd (f : nat -> real) (i w : nat) (t : nat) : real = f (i + t * w)

noextract
let ev_eq (f : nat -> real) (i o : nat) (t : nat)
  : Lemma (ev (strd f i o) t == strd f i (2 * o) t)
  = FStar.Math.Lemmas.paren_mul_right 2 t o;
    FStar.Math.Lemmas.swap_mul 2 t

noextract
let od_eq (f : nat -> real) (i o : nat) (t : nat)
  : Lemma (od (strd f i o) t == strd f (i + o) (2 * o) t)
  = FStar.Math.Lemmas.distributivity_add_left (2 * t) 1 o;
    FStar.Math.Lemmas.paren_mul_right 2 t o;
    FStar.Math.Lemmas.swap_mul 2 t

#push-options "--fuel 2 --ifuel 1 --z3rlimit 40"
noextract
let rec rtr_sum (rx : S.seq real) (k : nat{k <= 8}) (i : nat)
  : Lemma (requires i < pow2 (8 - k))
          (ensures rtr rx k i == rs (strd (rp0 rx) i (pow2 (8 - k))) (pow2 k))
          (decreases k)
  = if k = 0 then ()
    else begin
      let o = pow2 (8 - k) in
      let n = pow2 (k - 1) in
      assert (pow2 (9 - k) == 2 * o);
      assert (pow2 k == 2 * n);
      rtr_sum rx (k - 1) i;
      rtr_sum rx (k - 1) (i + o);
      rs_interleave (strd (rp0 rx) i o) n;
      FStar.Classical.forall_intro (ev_eq (rp0 rx) i o);
      FStar.Classical.forall_intro (od_eq (rp0 rx) i o);
      rs_ext (ev (strd (rp0 rx) i o)) (strd (rp0 rx) i (2 * o)) n;
      rs_ext (od (strd (rp0 rx) i o)) (strd (rp0 rx) (i + o) (2 * o)) n
    end
#pop-options

#push-options "--fuel 9 --ifuel 1"
noextract
let pow2_8 () : Lemma (pow2 8 == 256 /\ pow2 0 == 1) = ()
#pop-options

#push-options "--fuel 5 --ifuel 1"
noextract
let rp0_blk (rx : S.seq real) (c : nat) : Lemma (rp0 rx c == blk (sqf rx) 4 c) = ()
#pop-options

noextract
let sq_nonneg (x : real) : Lemma (x *. x >=. 0.0R) = ()

noextract
let vsum_rs_ext (n : nat) (g : nat -> real) (f : nat -> real)
  : Lemma ((forall (k:nat). k < n ==> f k == g k) ==> vsum n f == rs g n)
  = FStar.Classical.move_requires (rs_ext f g) n;
    vsum_rs n f

noextract
let vsum_rs_all (n : nat) (g : nat -> real)
  : Lemma (forall (f : nat -> real). {:pattern (vsum n f)}
             (forall (k:nat). k < n ==> f k == g k) ==> vsum n f == rs g n)
  = FStar.Classical.forall_intro (vsum_rs_ext n g)

noextract
let sq_sum_rs (rx : S.seq real)
  : Lemma (sq_sum rx 0 1024 == rs (sqf rx) 1024)
  = vsum_rs_all 1024 (sqf rx)

noextract
let tree_rs (rx : S.seq real)
  : Lemma (rtr rx 8 0 == rs (sqf rx) 1024)
  = pow2_8 ();
    rtr_sum rx 8 0;
    assert (rtr rx 8 0 == rs (strd (rp0 rx) 0 1) 256);
    rs_ext (strd (rp0 rx) 0 1) (rp0 rx) 256;
    FStar.Classical.forall_intro (rp0_blk rx);
    rs_ext (rp0 rx) (blk (sqf rx) 4) 256;
    rs_block (sqf rx) 4 256

noextract
let tree_sq_sum (rx : S.seq real)
  : Lemma (rtr rx 8 0 == sq_sum rx 0 1024 /\ rtr rx 8 0 >=. 0.0R)
  = tree_rs rx;
    sq_sum_rs rx;
    FStar.Classical.forall_intro sq_nonneg;
    rs_nonneg (sqf rx) 1024

(** Normalized vector *)

noextract
let lit_facts ()
  : Lemma (eps_lit () %~ eps_f32 /\ inv1024_lit () %~ (1.0R /. 1024.0R) /\ eps_f32 >. 0.0R)
  = eps_cert (); inv_cert (); eps_value (); inv_value ();
    CF.literal_approx eps_s 8796093 (-43);
    CF.literal_approx inv_s 8388608 (-33)

noextract
let rinv (rx : S.seq real) : real =
  rsqrt_t (rdiv (sq_sum rx 0 1024) (of_int 1024) +. eps_f32)

noextract
let invf_approx (sr : S.seq BF.t) (rx : S.seq real)
  : Lemma (requires sr %~ rx) (ensures invf sr %~ rinv rx)
  = lit_facts ();
    tree_sq_sum rx;
    trf_approx sr rx 8 0;
    let s = rtr rx 8 0 in
    let t = trf sr 8 0 in
    a_mul t (inv1024_lit ()) s (1.0R /. 1024.0R);
    a_add (mul t (inv1024_lit ())) (eps_lit ()) (s *. (1.0R /. 1024.0R)) eps_f32;
    let arg : (x:real{x >. 0.0R}) = s *. (1.0R /. 1024.0R) +. eps_f32 in
    assert (rdiv (sq_sum rx 0 1024) (of_int 1024) == s *. (1.0R /. 1024.0R));
    rsqrt_models_pat F32.t;
    approx_apply_dom #F32.t #real #F32.t #real #(fun x -> x >. 0.0R)
      (rsqrt #F32.t) FStar.Math.Sqrt.rsqrt (add (mul t (inv1024_lit ())) (eps_lit ())) arg

noextract
let nrm_approx (sn sr : S.seq BF.t) (rn rx : S.seq real) (k : nat)
  : Lemma (requires sr %~ rx /\ sn %~ rn /\ S.length rx == 1024 /\ k < 1024)
          (ensures nrm sn sr k %~ S.index (rmsnorm1p eps_f32 rx 1024 rn 0) k)
  = invf_approx sr rx;
    wx_approx sr rx k;
    wx_approx sn rn k;
    a1 #F32.t;
    a_mul (wx sr k) (invf sr) (at rx k) (rinv rx);
    a_add one (wx sn k) 1.0R (at rn k);
    a_mul (mul (wx sr k) (invf sr)) (add one (wx sn k)) (at rx k *. rinv rx) (1.0R +. at rn k);
    Casts.fcast_approx #F32.t #BF.t (mul (mul (wx sr k) (invf sr)) (add one (wx sn k)))
      ((at rx k *. rinv rx) *. (1.0R +. at rn k))

(** Matrix models *)

noextract
let rtm (rt : S.seq real) : chest2 real 248320 1024 = mk2 (fun i j -> at rt (i * 1024 + j))
noextract
let rbm (hn : S.seq real) : chest2 real 1024 8 = mk2 (fun k n -> at hn k)

noextract
let wlay_f (i : natlt 248320) (j : natlt 1024) : Lemma (wlay.imap.f (idx2 i j) == i * 1024 + j) = ()
noextract
let blay_f (k : natlt 1024) (n : natlt 8) : Lemma (blay.imap.f (idx2 k n) == n * 1024 + k) = ()
noextract
let clay_f (m : natlt 248320) (j : natlt 8) : Lemma (clay.imap.f (idx2 m j) == m * 8 + j) = ()

noextract
let chest2_approx_intro (#et:Type0) {| scalar et, real_like et |} (#r #c : nat)
  (m1 : chest2 et r c) (m2 : chest2 real r c)
  : Lemma (requires forall (i:natlt r) (j:natlt c). acc2 m1 i j %~ acc2 m2 i j)
          (ensures m1 %~ m2)
  = introduce forall (ix : abs (r @| c @| INil)). acc m1 ix %~ acc m2 ix
    with (let (i, (j, ())) = ix in assert (acc2 m1 i j %~ acc2 m2 i j))

noextract
let chest2_approx_elim (#et:Type0) {| scalar et, real_like et |} (#r #c : nat)
  (m1 : chest2 et r c) (m2 : chest2 real r c) (i : natlt r) (j : natlt c)
  : Lemma (requires m1 %~ m2) (ensures acc2 m1 i j %~ acc2 m2 i j)
  = ()

noextract
let wmat_el (st : S.seq BF.t) (rt : S.seq real) (i : natlt 248320) (j : natlt 1024)
  : Lemma (requires st %~ rt /\ S.length st == table_len)
          (ensures acc2 (wmat st) i j %~ acc2 (rtm rt) i j)
  = wlay_f i j;
    FStar.Math.Lemmas.lemma_mult_le_right 1024 i 248319

noextract
let wmat_approx (st : S.seq BF.t) (rt : S.seq real)
  : Lemma (requires st %~ rt /\ S.length st == table_len)
          (ensures wmat st %~ rtm rt)
  = introduce forall (i:natlt 248320) (j:natlt 1024). acc2 (wmat st) i j %~ acc2 (rtm rt) i j
    with wmat_el st rt i j;
    chest2_approx_intro (wmat st) (rtm rt)

noextract
let bmat_el (sb : S.seq BF.t) (hn : S.seq real) (k : natlt 1024) (n : natlt 8)
  : Lemma (requires S.length sb == 8192 /\
                    (forall (n k : nat). n < 8 /\ k < 1024 ==> S.index sb (n * 1024 + k) %~ at hn k))
          (ensures acc2 (bmat sb) k n %~ acc2 (rbm hn) k n)
  = blay_f k n

noextract
let bmat_approx (sb : S.seq BF.t) (hn : S.seq real)
  : Lemma (requires S.length sb == 8192 /\
                    (forall (n k : nat). n < 8 /\ k < 1024 ==> S.index sb (n * 1024 + k) %~ at hn k))
          (ensures bmat sb %~ rbm hn)
  = introduce forall (k:natlt 1024) (n:natlt 8). acc2 (bmat sb) k n %~ acc2 (rbm hn) k n
    with bmat_el sb hn k n;
    chest2_approx_intro (bmat sb) (rbm hn)

noextract
let wsub_approx (ew : chest2 BF.t 248320 1024) (rw : chest2 real 248320 1024)
  (rb : natlt 15520) (q : natlt 64)
  : Lemma (requires ew %~ rw)
          (ensures ematrix_subtile ew 16 16 rb q %~ ematrix_subtile rw 16 16 rb q)
  = introduce forall (i:natlt 16) (j:natlt 16).
      acc2 (ematrix_subtile ew 16 16 rb q) i j %~ acc2 (ematrix_subtile rw 16 16 rb q) i j
    with chest2_approx_elim ew rw (rb * 16 + i) (q * 16 + j);
    chest2_approx_intro (ematrix_subtile ew 16 16 rb q) (ematrix_subtile rw 16 16 rb q)

noextract
let bsub_approx (eb : chest2 BF.t 1024 8) (rbb : chest2 real 1024 8) (q : natlt 64)
  : Lemma (requires eb %~ rbb)
          (ensures ematrix_subtile eb 16 8 q 0 %~ ematrix_subtile rbb 16 8 q 0)
  = introduce forall (i:natlt 16) (j:natlt 8).
      acc2 (ematrix_subtile eb 16 8 q 0) i j %~ acc2 (ematrix_subtile rbb 16 8 q 0) i j
    with chest2_approx_elim eb rbb (q * 16 + i) j;
    chest2_approx_intro (ematrix_subtile eb 16 8 q 0) (ematrix_subtile rbb 16 8 q 0)

noextract
let rec racc (rt hn : S.seq real) (rb : natlt 15520) (p : nat{p <= 64})
  : Tot (chest2 real 16 8) (decreases p)
  = if p = 0 then const (16 @| 8 @| INil) 0.0R
    else GEMM.matplus (racc rt hn rb (p - 1))
                 (GEMM.matmul (ematrix_subtile (rtm rt) 16 16 rb (p - 1))
                         (ematrix_subtile (rbm hn) 16 8 (p - 1) 0))

noextract
let rec acc_approx (ew : chest2 BF.t 248320 1024) (eb : chest2 BF.t 1024 8)
  (rt hn : S.seq real) (rb : natlt 15520) (p : nat{p <= 64})
  : Lemma (requires ew %~ rtm rt /\ eb %~ rbm hn)
          (ensures acc_tile ew eb rb p %~ racc rt hn rb p) (decreases p)
  = if p = 0 then begin
      a0 #F32.t;
      chest2_approx_intro (const (16 @| 8 @| INil) (zero #F32.t)) (const (16 @| 8 @| INil) 0.0R)
    end else begin
      acc_approx ew eb rt hn rb (p - 1);
      wsub_approx ew (rtm rt) rb (p - 1);
      bsub_approx eb (rbm hn) (p - 1);
      MMA.emma_approx_lemma (acc_tile ew eb rb (p - 1))
        (ematrix_subtile ew 16 16 rb (p - 1)) (ematrix_subtile eb 16 8 (p - 1) 0)
        (racc rt hn rb (p - 1))
        (ematrix_subtile (rtm rt) 16 16 rb (p - 1)) (ematrix_subtile (rbm hn) 16 8 (p - 1) 0)
    end

noextract
let gfun (rt hn : S.seq real) (row : nat) (k : nat) : real = at rt (row * 1024 + k) *. at hn k

noextract
let rec mm_rs (rt hn : S.seq real) (rb : natlt 15520) (q : natlt 64) (i : natlt 16) (t : nat{t <= 16})
  : Lemma (ensures GEMM.__matmul_single (ematrix_subtile (rtm rt) 16 16 rb q)
                                   (ematrix_subtile (rbm hn) 16 8 q 0) i 0 t
                   == rs (shift (gfun rt hn (16 * rb + i)) (16 * q)) t)
          (decreases t)
  = let sa = ematrix_subtile (rtm rt) 16 16 rb q in
    let sb = ematrix_subtile (rbm hn) 16 8 q 0 in
    if t = 0 then GEMM.matmul_zero_lemma sa sb i 0
    else begin
      mm_rs rt hn rb q i (t - 1);
      GEMM.matmul_single_lemma sa sb i 0 t
    end

noextract
let rec racc_val (rt hn : S.seq real) (rb : natlt 15520) (p : nat{p <= 64}) (i : natlt 16)
  : Lemma (ensures acc2 (racc rt hn rb p) i 0 == rs (gfun rt hn (16 * rb + i)) (16 * p))
          (decreases p)
  = if p = 0 then ()
    else begin
      racc_val rt hn rb (p - 1) i;
      let sa = ematrix_subtile (rtm rt) 16 16 rb (p - 1) in
      let sb = ematrix_subtile (rbm hn) 16 8 (p - 1) 0 in
      GEMM.lemma_matmul_index sa sb i 0;
      mm_rs rt hn rb (p - 1) i 16;
      rs_split (gfun rt hn (16 * rb + i)) (16 * (p - 1)) 16
    end

noextract
let rec cfin_val (ec : chest2 F32.t 248320 8) (ew : chest2 BF.t 248320 1024) (eb : chest2 BF.t 1024 8)
  (r : nat{r <= 15520}) (m : natlt 248320)
  : Lemma (ensures acc2 (cfin ec ew eb r) m 0 ==
                   (if m / 16 < r then acc2 (acc_tile ew eb (m / 16) 64) (m % 16) 0 else acc2 ec m 0))
          (decreases r)
  = if r = 0 then () else cfin_val ec ew eb (r - 1) m

noextract
let cseq_val (c : chest2 F32.t 248320 8) (m : natlt 248320)
  : Lemma (S.index (to_seq clay c) (m * 8) == acc2 c m 0)
  = clay_f m 0;
    Kuiper.Injection.lem_pat clay.imap (idx2 m 0)

(** Final relation *)

noextract
let divmod16 (m : natlt 248320)
  : Lemma (0 <= m / 16 /\ m / 16 < 15520 /\ 0 <= m % 16 /\ m % 16 < 16 /\ 16 * (m / 16) + m % 16 == m)
  = FStar.Math.Lemmas.euclidean_division_definition m 16

noextract
let logit_el (st sn sr l sb : S.seq BF.t) (scf : S.seq F32.t) (rt rn rx : S.seq real) (m : natlt 248320)
  : Lemma (requires S.length st == table_len /\ S.length sn == 1024 /\ S.length sr == 1024 /\
                    S.length sb == 8192 /\
                    (forall (n k : nat). n < 8 /\ k < 1024 ==> S.index sb (n * 1024 + k) == nrm sn sr k) /\
                    S.length l == 248320 /\
                    (forall (m:nat). m < 248320 ==>
                      S.index l m == narrow (S.index (to_seq clay (cfin (cmat scf) (wmat st) (bmat sb) 15520)) (m * 8))) /\
                    st %~ rt /\ sn %~ rn /\ sr %~ rx)
          (ensures S.index l m %~ S.index (logits base_numerics rt 0 rn 0 rx) m)
  = divmod16 m;
    let hn = rmsnorm1p eps_f32 rx 1024 rn 0 in
    let c = cfin (cmat scf) (wmat st) (bmat sb) 15520 in
    introduce forall (k:nat). k < 1024 ==> nrm sn sr k %~ at hn k
    with introduce _ ==> _ with nrm_approx sn sr rn rx k;
    wmat_approx st rt;
    bmat_approx sb hn;
    cseq_val c m;
    cfin_val (cmat scf) (wmat st) (bmat sb) 15520 m;
    acc_approx (wmat st) (bmat sb) rt hn (m / 16) 64;
    chest2_approx_elim (acc_tile (wmat st) (bmat sb) (m / 16) 64) (racc rt hn (m / 16) 64) (m % 16) 0;
    racc_val rt hn (m / 16) 64 (m % 16);
    vsum_rs_all 1024 (gfun rt hn m);
    assert (S.index (logits base_numerics rt 0 rn 0 rx) m == rs (gfun rt hn m) 1024);
    Casts.fcast_approx #F32.t #BF.t (acc2 c m 0) (rs (gfun rt hn m) 1024)

noextract
let final_lemma (st sn sr h l sb : S.seq BF.t) (scf : S.seq F32.t)
  : Lemma (requires S.length st == table_len /\ S.length sn == 1024 /\ S.length sr == 1024 /\
                    S.length h == 1024 /\ (forall (i:nat). i < 1024 ==> S.index h i == S.index sr i) /\
                    S.length sb == 8192 /\
                    (forall (n k : nat). n < 8 /\ k < 1024 ==> S.index sb (n * 1024 + k) == nrm sn sr k) /\
                    S.length l == 248320 /\
                    (forall (m:nat). m < 248320 ==>
                      S.index l m == narrow (S.index (to_seq clay (cfin (cmat scf) (wmat st) (bmat sb) 15520)) (m * 8))))
          (ensures base_final_norm_projection_relation st sn sr h l)
  = introduce forall (rt rn rx : S.seq real).
      st %~ rt /\ sn %~ rn /\ sr %~ rx ==> l %~ logits base_numerics rt 0 rn 0 rx
    with introduce _ ==> _ with
      introduce forall (m:nat). m < 248320 ==> S.index l m %~ S.index (logits base_numerics rt 0 rn 0 rx) m
      with introduce _ ==> _ with logit_el st sn sr l sb scf rt rn rx m

#pop-options

noextract
let isoff (vo : nat) : prop =
  vo == 128 \/ vo == 64 \/ vo == 32 \/ vo == 16 \/ vo == 8 \/ vo == 4 \/ vo == 2 \/ vo == 1 \/ vo == 0

noextract
let klev (vo : nat) : (k:nat{k <= 8}) =
  if vo = 128 then 0 else if vo = 64 then 1 else if vo = 32 then 2 else if vo = 16 then 3
  else if vo = 8 then 4 else if vo = 4 then 5 else if vo = 2 then 6 else if vo = 1 then 7 else 8

#push-options "--fuel 9 --ifuel 1"
noextract
let klev_step (vo : nat)
  : Lemma (requires isoff vo /\ vo > 0)
          (ensures klev vo < 8 /\ pow2 (7 - klev vo) == vo /\
                   klev (vo / 2) == klev vo + 1 /\ isoff (vo / 2))
  = ()
#pop-options

noextract
let trf_unfold (sr : S.seq BF.t) (k : nat{k < 8}) (i : nat)
  : Lemma (trf sr (k + 1) i ==
           (if i < pow2 (7 - k) then add (trf sr k i) (trf sr k (i + pow2 (7 - k))) else trf sr k i))
  = ()

noextract
let k1facts (sn sr h b : S.seq BF.t) : prop =
  S.length sr == 1024 /\ S.length h == 1024 /\ (forall (i:nat). i < 1024 ==> S.index h i == S.index sr i) /\
  S.length b == 8192 /\ (forall (n k : nat). n < 8 /\ k < 1024 ==> S.index b (n * 1024 + k) == nrm sn sr k)

noextract
let k3facts (sc : S.seq F32.t) (l : S.seq BF.t) : prop =
  S.length l == 248320 /\
  (forall (m:nat). m < 248320 ==> m * 8 < S.length sc /\ S.index l m == narrow (S.index sc (m * 8)))

noextract
let trf_level (sr : S.seq BF.t) (k : nat{k < 8}) (p : S.seq F32.t)
  : Lemma (requires S.length p == 256 /\
             (forall (q:nat). q < 256 ==> S.index p q ==
               (if q < pow2 (7 - k) then trf sr (k + 1) q else trf sr k q)))
          (ensures forall (q:nat). q < 256 ==> S.index p q == trf sr (k + 1) q)
  = introduce forall (q:nat). q < 256 ==> S.index p q == trf sr (k + 1) q
    with introduce _ ==> _ with trf_unfold sr k q

noextract
let bidx_fits (n k : nat)
  : Lemma (requires n < 8 /\ k < 1024)
          (ensures FStar.SizeT.fits (n * 1024) /\ FStar.SizeT.fits (n * 1024 + k) /\ n * 1024 + k < 8192)
  = FStar.UInt.pow2_values 16;
    FStar.SizeT.fits_at_least_16 (n * 1024);
    FStar.SizeT.fits_at_least_16 (n * 1024 + k)

noextract
let kinv (sn sr b : S.seq BF.t) (vk : nat) : prop =
  vk <= 1024 /\ S.length b == 8192 /\
  (forall (n' q : nat). n' < 8 /\ q < vk ==> S.index b (n' * 1024 + q) == nrm sn sr q)

noextract
let bidx_inv (sn sr b : S.seq BF.t) (vn vk : nat) : prop =
  vn <= 8 /\ vk < 1024 /\ kinv sn sr b vk /\
  (forall (n' : nat). n' < vn ==> S.index b (n' * 1024 + vk) == nrm sn sr vk)

noextract
let bidx_close (sn sr b : S.seq BF.t) (vk : nat)
  : Lemma (requires bidx_inv sn sr b 8 vk)
          (ensures kinv sn sr b (vk + 1))
  = introduce forall (n q : nat). n < 8 /\ q < vk + 1 ==> S.index b (n * 1024 + q) == nrm sn sr q
    with introduce _ ==> _ with (if q < vk then () else ())

noextract
let bidx_step (sn sr b : S.seq BF.t) (vn vk : nat) (v : BF.t)
  : Lemma (requires vn < 8 /\ vk < 1024 /\ bidx_inv sn sr b vn vk /\ v == nrm sn sr vk)
          (ensures bidx_inv sn sr (S.upd b (vn * 1024 + vk) v) (vn + 1) vk)
  = let b' = S.upd b (vn * 1024 + vk) v in
    let aux (n' q : nat)
      : Lemma (requires n' < 8 /\ q < 1024 /\ n' * 1024 + q <> vn * 1024 + vk)
              (ensures S.index b' (n' * 1024 + q) == S.index b (n' * 1024 + q))
      = ()
    in
    let aux2 (n' q : nat)
      : Lemma (requires n' < 8 /\ q < 1024 /\ (n' <> vn \/ q <> vk))
              (ensures n' * 1024 + q <> vn * 1024 + vk)
      = if n' < vn then FStar.Math.Lemmas.lemma_mult_le_right 1024 (n' + 1) vn
        else if n' > vn then FStar.Math.Lemmas.lemma_mult_le_right 1024 (vn + 1) n'
        else ()
    in
    introduce forall (n' q : nat). n' < 8 /\ q < vk ==> S.index b' (n' * 1024 + q) == nrm sn sr q
    with introduce _ ==> _ with (aux2 n' q; aux n' q);
    introduce forall (n' : nat). n' < vn + 1 ==> S.index b' (n' * 1024 + vk) == nrm sn sr vk
    with introduce _ ==> _ with (if n' < vn then (aux2 n' vk; aux n' vk) else ())

noextract
let final_rel (st sn sr h l sb : S.seq BF.t) (scf : S.seq F32.t)
  : Lemma (requires k1facts sn sr h sb /\ k3facts (to_seq clay (cfin (cmat scf) (wmat st) (bmat sb) 15520)) l)
          (ensures base_final_norm_projection_relation st sn sr h l)
  = introduce (S.length st == table_len /\ S.length sn == 1024 /\ S.length sr == 1024) ==>
              base_final_norm_projection_relation st sn sr h l
    with final_lemma st sn sr h l sb scf

(*** Kernel 1: copy, RMS reduction, normalization (single thread) *)

inline_for_extraction noextract
fn k1
  (normw residual hidden : larray BF.t 1024)
  (part : larray F32.t 256)
  (bsc : larray BF.t 8192)
  (#fw #fr : perm) (#sn #sr #h0 #b0 : erased (S.seq BF.t)) (#p0 : erased (S.seq F32.t))
  preserves gpu ** normw |-> Frac fw sn ** residual |-> Frac fr sr
  requires hidden |-> h0 ** part |-> p0 ** bsc |-> b0
  ensures exists* (h b : S.seq BF.t) (p : S.seq F32.t).
    hidden |-> h ** part |-> p ** bsc |-> b ** pure (k1facts sn sr h b)
{
  pts_to_len normw;
  pts_to_len residual;
  pts_to_len hidden;
  pts_to_len part;
  pts_to_len bsc;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (vi : SZ.t) (h : S.seq BF.t).
      i |-> vi ** hidden |-> h ** pure (SZ.v vi <= 1024 /\ S.length h == 1024 /\ S.length sr == 1024 /\
        (forall (q:nat). q < SZ.v vi ==> S.index h q == S.index sr q))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    let x = residual.(vi);
    hidden.(vi) <- x;
    i := vi +^ 1sz;
  };
  let mut c = 0sz;
  while (!c <^ 256sz)
    invariant exists* (vc : SZ.t) (p : S.seq F32.t).
      c |-> vc ** part |-> p ** pure (SZ.v vc <= 256 /\ S.length p == 256 /\
        (forall (q:nat). q < SZ.v vc ==> S.index p q == p0f sr q))
    decreases (256 - SZ.v !c)
  {
    let vc = !c;
    let b = 4sz *^ vc;
    let x0 = widen (residual.(b));
    let x1 = widen (residual.(b +^ 1sz));
    let x2 = widen (residual.(b +^ 2sz));
    let x3 = widen (residual.(b +^ 3sz));
    let s = add (add (add (add zero (mul x0 x0)) (mul x1 x1)) (mul x2 x2)) (mul x3 x3);
    assert (pure (s == p0f sr (SZ.v vc)));
    part.(vc) <- s;
    c := vc +^ 1sz;
  };
  let mut off = 128sz;
  while (0sz <^ !off)
    invariant exists* (vo : SZ.t) (p : S.seq F32.t).
      off |-> vo ** part |-> p ** pure (SZ.v vo <= 128 /\ S.length p == 256 /\ isoff (SZ.v vo) /\
        (forall (q:nat). q < 256 ==> S.index p q == trf sr (klev (SZ.v vo)) q))
    decreases (SZ.v !off)
  {
    let vo = !off;
    klev_step (SZ.v vo);
    let mut j = 0sz;
    while (!j <^ vo)
      invariant exists* (vj : SZ.t) (p : S.seq F32.t).
        j |-> vj ** part |-> p ** pure (SZ.v vj <= SZ.v vo /\ S.length p == 256 /\
          (forall (q:nat). q < 256 ==> S.index p q ==
            (if q < SZ.v vj then trf sr (klev (SZ.v vo) + 1) q else trf sr (klev (SZ.v vo)) q)))
      decreases (SZ.v vo - SZ.v !j)
    {
      let vj = !j;
      let a = part.(vj);
      let b = part.(vj +^ vo);
      trf_unfold sr (klev (SZ.v vo)) (SZ.v vj);
      part.(vj) <- add a b;
      j := vj +^ 1sz;
    };
    with p1. assert (part |-> p1);
    trf_level sr (klev (SZ.v vo)) p1;
    off := vo /^ 2sz;
  };
  let s = part.(0sz);
  let inv = rsqrt (add (mul s (inv1024_lit ())) (eps_lit ()));
  assert (pure (inv == invf sr));
  let mut k = 0sz;
  while (!k <^ 1024sz)
    invariant exists* (vk : SZ.t) (b : S.seq BF.t).
      k |-> vk ** bsc |-> b ** pure (SZ.v vk <= 1024 /\ kinv sn sr b (SZ.v vk))
    decreases (1024 - SZ.v !k)
  {
    let vk = !k;
    let r = widen (residual.(vk));
    let w = widen (normw.(vk));
    let nv = narrow (mul (mul r inv) (add one w));
    assert (pure (nv == nrm sn sr (SZ.v vk)));
    let mut n = 0sz;
    while (!n <^ 8sz)
      invariant exists* (vn : SZ.t) (b : S.seq BF.t).
        n |-> vn ** bsc |-> b ** pure (SZ.v vn <= 8 /\ SZ.v vk < 1024 /\ bidx_inv sn sr b (SZ.v vn) (SZ.v vk))
      decreases (8 - SZ.v !n)
    {
      let vn = !n;
      bidx_fits (SZ.v vn) (SZ.v vk);
      with b. assert (bsc |-> b);
      bidx_step sn sr b (SZ.v vn) (SZ.v vk) nv;
      bsc.(vn *^ 1024sz +^ vk) <- nv;
      n := vn +^ 1sz;
    };
    with b2. assert (bsc |-> b2);
    bidx_close sn sr b2 (SZ.v vk);
    k := vk +^ 1sz;
  };
  with h1. assert (hidden |-> h1);
  with b1. assert (bsc |-> b1);
  assert (pure (k1facts sn sr h1 b1));
}

(*** Kernel 2: WMMA projection (one warp) *)

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn k2_lane
  (gw : array2 BF.t wlay)
  (gb : array2 BF.t blay)
  (gc : array2 F32.t clay)
  (#fw #fb : perm)
  (#ew : chest2 BF.t 248320 1024) (#eb : chest2 BF.t 1024 8) (#ec : chest2 F32.t 248320 8)
  preserves gpu ** gw |-> Frac fw ew ** gb |-> Frac fb eb
  requires gc |-> Frac (1.0R /. warp_size) ec
  ensures gc |-> Frac (1.0R /. warp_size) (cfin ec ew eb 15520)
{
  let fc = MMA.alloc_fragment ();
  let mut rb = 0sz;
  while (!rb <^ 15520sz)
    invariant exists* (vrb : SZ.t) vc.
      rb |-> vrb ** fc |-> vc **
      gc |-> Frac (1.0R /. warp_size) (cfin' ec ew eb (SZ.v vrb)) **
      pure (SZ.v vrb <= 15520)
    decreases (15520 - SZ.v !rb)
  {
    let vrb = !rb;
    MMA.fill fc zero;
    let mut p = 0sz;
    while (!p <^ 64sz)
      invariant exists* (vp : SZ.t).
        p |-> vp **
        fc |-> acc' ew eb (SZ.v vrb) (SZ.v vp) **
        pure (SZ.v vp <= 64)
      decreases (64 - SZ.v !p)
    {
      let vp = !p;
      let ta = array2_extract_tile_ro' gw 16 16 (SZ.v vrb) (SZ.v vp);
      let tb = array2_extract_tile_ro' gb 16 8 (SZ.v vp) 0;
      MMA.mma_sync ta tb fc;
      with eta. assert (tensor_pts_to ta #fw eta);
      Pulse.Lib.Trade.elim_trade (ta |-> Frac fw eta) (gw |-> Frac fw ew);
      with etb. assert (tensor_pts_to tb #fb etb);
      Pulse.Lib.Trade.elim_trade (tb |-> Frac fb etb) (gb |-> Frac fb eb);
      p := vp +^ 1sz;
    };
    let tc = array2_extract_tile_st gc 16 8 (SZ.v vrb) 0;
    MMA.store fc tc;
    with x3. assert (tensor_pts_to tc #(1.0R /. warp_size) x3);
    Pulse.Lib.Forall.elim_forall x3;
    Pulse.Lib.Trade.elim_trade (tensor_pts_to tc #(1.0R /. warp_size) x3) _;
    rb := vrb +^ 1sz;
  };
  with x. assert fc |-> x; drop_ (fc |-> x);
}
#pop-options

(*** Kernel 3: logits (single thread) *)

inline_for_extraction noextract
fn k3
  (csc : larray F32.t 1986560)
  (logits : larray BF.t 248320)
  (#sc : erased (S.seq F32.t)) (#l0 : erased (S.seq BF.t))
  preserves gpu ** csc |-> sc
  requires logits |-> l0
  ensures exists* (l : S.seq BF.t). logits |-> l ** pure (k3facts sc l)
{
  pts_to_len csc;
  pts_to_len logits;
  let mut m = 0sz;
  while (!m <^ (SZ.uint32_to_sizet 248320ul))
    invariant exists* (vm : SZ.t) (l : S.seq BF.t).
      m |-> vm ** logits |-> l ** pure (SZ.v vm <= 248320 /\ S.length l == 248320 /\
        (forall (q:nat). q < SZ.v vm ==> q * 8 < S.length sc /\ S.index l q == narrow (S.index sc (q * 8))))
    decreases (248320 - SZ.v !m)
  {
    let vm = !m;
    let x = csc.(vm *^ 8sz);
    logits.(vm) <- narrow x;
    m := vm +^ 1sz;
  };
}

(*** Host *)

instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

let k2post (table : larray BF.t table_len) (bsc : larray BF.t 8192) (csc : larray F32.t 1986560)
  (f : perm) (st sb : S.seq BF.t) (scf : S.seq F32.t) (tid : nat) : slprop =
  from_array wlay table |-> Frac (f /. 32) (wmat st) **
  from_array blay bsc |-> Frac (1.0R /. 32) (bmat sb) **
  from_array clay csc |-> Frac (1.0R /. 32) (cfin (cmat scf) (wmat st) (bmat sb) 15520)

let k2pre' (table : larray BF.t table_len) (bsc : larray BF.t 8192) (csc : larray F32.t 1986560)
  (f : perm) (st sb : S.seq BF.t) (scf : S.seq F32.t) (tid : nat) : slprop =
  from_array wlay table |-> Frac (f /. 32) (wmat st) **
  from_array blay bsc |-> Frac (1.0R /. 32) (bmat sb) **
  from_array clay csc |-> Frac (1.0R /. 32) (cmat scf)

let k2full_pre (table : larray BF.t table_len) (bsc : larray BF.t 8192) (csc : larray F32.t 1986560)
  (f : perm) (st sb : S.seq BF.t) (scf : S.seq F32.t) : slprop =
  table |-> Frac f st ** bsc |-> sb ** csc |-> scf

let k2full_post (table : larray BF.t table_len) (bsc : larray BF.t 8192) (csc : larray F32.t 1986560)
  (f : perm) (st sb : S.seq BF.t) (scf : S.seq F32.t) : slprop =
  table |-> Frac f st ** bsc |-> sb **
  csc |-> to_seq clay (cfin (cmat scf) (wmat st) (bmat sb) 15520)

let k2frame (st sb : S.seq BF.t) (scf : S.seq F32.t) : slprop =
  pure (S.length st == table_len /\ S.length sb == 8192 /\ S.length scf == 1986560)

ghost
fn k2_setup
  (table : larray BF.t table_len) (bsc : larray BF.t 8192) (csc : larray F32.t 1986560)
  (f : perm) (st sb : S.seq BF.t) (scf : S.seq F32.t) ()
  requires k2full_pre table bsc csc f st sb scf
  ensures (forall+ (tid : natlt (SZ.v 32sz)). k2pre' table bsc csc f st sb scf tid) ** k2frame st sb scf
{
  unfold k2full_pre;
  pts_to_len table;
  pts_to_len bsc;
  pts_to_len csc;
  fold (k2frame st sb scf);
  tensor_abs' wlay table;
  tensor_abs' blay bsc;
  tensor_abs' clay csc;
  rewrite each from_seq wlay st as wmat st;
  rewrite each from_seq blay sb as bmat sb;
  rewrite each from_seq clay scf as cmat scf;
  tensor_share_n (from_array wlay table) 32;
  tensor_share_n (from_array blay bsc) 32;
  tensor_share_n (from_array clay csc) 32;
  forevery_zip3 #(natlt 32)
    (fun _ -> from_array wlay table |-> Frac (f /. 32) (wmat st))
    (fun _ -> from_array blay bsc |-> Frac (1.0R /. 32) (bmat sb))
    (fun _ -> from_array clay csc |-> Frac (1.0R /. 32) (cmat scf));
  forevery_ext #(natlt 32) _ (k2pre' table bsc csc f st sb scf);
  forevery_rw_size 32 (SZ.v 32sz);
}

ghost
fn k2_teardown
  (table : larray BF.t table_len) (bsc : larray BF.t 8192) (csc : larray F32.t 1986560)
  (f : perm) (st sb : S.seq BF.t) (scf : S.seq F32.t) ()
  requires (forall+ (tid : natlt (SZ.v 32sz)). k2post table bsc csc f st sb scf tid) ** k2frame st sb scf
  ensures k2full_post table bsc csc f st sb scf
{
  unfold k2frame;
  forevery_rw_size (SZ.v 32sz) 32;
  forevery_ext #(natlt 32) _
    (fun _ -> from_array wlay table |-> Frac (f /. 32) (wmat st) **
      from_array blay bsc |-> Frac (1.0R /. 32) (bmat sb) **
      from_array clay csc |-> Frac (1.0R /. 32) (cfin (cmat scf) (wmat st) (bmat sb) 15520));
  forevery_unzip3 #(natlt 32)
    (fun _ -> from_array wlay table |-> Frac (f /. 32) (wmat st))
    (fun _ -> from_array blay bsc |-> Frac (1.0R /. 32) (bmat sb))
    (fun _ -> from_array clay csc |-> Frac (1.0R /. 32) (cfin (cmat scf) (wmat st) (bmat sb) 15520));
  tensor_gather_n (from_array wlay table) 32;
  tensor_gather_n (from_array blay bsc) 32;
  tensor_gather_n (from_array clay csc) 32;
  tensor_concr (from_array wlay table);
  tensor_concr (from_array blay bsc);
  tensor_concr (from_array clay csc);
  rewrite each core (from_array wlay table) as table;
  rewrite each core (from_array blay bsc) as bsc;
  rewrite each core (from_array clay csc) as csc;
  rewrite each to_seq wlay (wmat st) as st;
  rewrite each to_seq blay (bmat sb) as sb;
  fold (k2full_post table bsc csc f st sb scf);
}

inline_for_extraction noextract
fn k2_thread
  (table : larray BF.t table_len) (bsc : larray BF.t 8192) (csc : larray F32.t 1986560)
  (f : perm) (st sb : erased (S.seq BF.t)) (scf : erased (S.seq F32.t))
  (tid : szlt 32sz) ()
  requires gpu ** k2pre' table bsc csc f st sb scf tid ** thread_id 32sz tid
  ensures gpu ** k2post table bsc csc f st sb scf tid ** thread_id 32sz tid
{
  unfold k2pre';
  rewrite each (1.0R /. 32) as (1.0R /. warp_size);
  k2_lane (from_array wlay table) (from_array blay bsc) (from_array clay csc);
  rewrite each (1.0R /. warp_size) as (1.0R /. 32);
  fold (k2post table bsc csc f st sb scf tid);
}

inline_for_extraction noextract
let k2desc
  (table : larray BF.t table_len { is_global table })
  (bsc : larray BF.t 8192 { is_global bsc })
  (csc : larray F32.t 1986560 { is_global csc })
  (f : perm) (st sb : erased (S.seq BF.t)) (scf : erased (S.seq F32.t))
  : kernel_desc (k2full_pre table bsc csc f st sb scf) (k2full_post table bsc csc f st sb scf)
= {
  f = k2_thread table bsc csc f st sb scf;
  nthr = 32sz;
  frame = k2frame st sb scf;
  kpre = k2pre' table bsc csc f st sb scf;
  kpost = k2post table bsc csc f st sb scf;
  block_setup = k2_setup table bsc csc f st sb scf;
  block_teardown = k2_teardown table bsc csc f st sb scf;
  kpost_sendable = solve;
  kpre_sendable = solve;
  full_post_sendable = solve;
  full_pre_sendable = solve;
} <: kernel_desc_1_n _ _

fn kplay_run
  (table : larray BF.t table_len { is_global table })
  (normw : larray BF.t 1024 { is_global normw })
  (residual : larray BF.t 1024 { is_global residual })
  (hidden : larray BF.t 1024 { is_global hidden })
  (logits : larray BF.t 248320 { is_global logits })
  (#ft #fw #fr : perm)
  (#st #sn #sr : erased (S.seq BF.t))
  (#h0 #l0 : erased (S.seq BF.t))
  preserves cpu
  preserves on gpu_loc (table |-> Frac ft st)
  preserves on gpu_loc (normw |-> Frac fw sn)
  preserves on gpu_loc (residual |-> Frac fr sr)
  requires on gpu_loc (hidden |-> h0)
  requires on gpu_loc (logits |-> l0)
  ensures exists* (h l : S.seq BF.t).
    on gpu_loc (hidden |-> h) **
    on gpu_loc (logits |-> l) **
    pure (base_final_norm_projection_relation st sn sr h l)
{
  let part0 = gpu_array_alloc #F32.t 256sz;
  let bsc0 = gpu_array_alloc #BF.t 8192sz;
  let csc0 = gpu_array_alloc #F32.t (SZ.uint32_to_sizet 1986560ul);
  let part : (x : larray F32.t 256 { is_global x }) = part0;
  let bsc : (x : larray BF.t 8192 { is_global x }) = bsc0;
  let csc : (x : larray F32.t 1986560 { is_global x }) = csc0;
  rewrite each part0 as part;
  rewrite each bsc0 as bsc;
  rewrite each csc0 as csc;
  with p0. assert on gpu_loc (part |-> p0);
  with b0. assert on gpu_loc (bsc |-> b0);
  launch_kernel_1 #(normw |-> Frac fw sn ** residual |-> Frac fr sr ** hidden |-> h0 ** part |-> p0 ** bsc |-> b0)
    (fun () -> k1 normw residual hidden part bsc #fw #fr #sn #sr #h0 #b0 #p0);
  with h1. assert on gpu_loc (hidden |-> h1);
  with sb. assert on gpu_loc (bsc |-> sb);
  on_pure_elim gpu_loc (k1facts sn sr h1 sb);
  with scf. assert on gpu_loc (csc |-> scf);
  on_star_intro (bsc |-> sb) (csc |-> scf);
  on_star_intro (table |-> Frac ft st) (bsc |-> sb ** csc |-> scf);
  rewrite on gpu_loc (table |-> Frac ft st ** bsc |-> sb ** csc |-> scf)
       as on gpu_loc (k2full_pre table bsc csc ft st sb scf);
  launch_sync (k2desc table bsc csc ft st sb scf);
  rewrite on gpu_loc (k2full_post table bsc csc ft st sb scf)
       as on gpu_loc (table |-> Frac ft st ** bsc |-> sb **
            csc |-> to_seq clay (cfin (cmat scf) (wmat st) (bmat sb) 15520));
  on_star_elim (table |-> Frac ft st)
    (bsc |-> sb ** csc |-> to_seq clay (cfin (cmat scf) (wmat st) (bmat sb) 15520));
  on_star_elim (bsc |-> sb) (csc |-> to_seq clay (cfin (cmat scf) (wmat st) (bmat sb) 15520));
  with sc. assert on gpu_loc (csc |-> sc);
  launch_kernel_1 #(csc |-> sc ** logits |-> l0)
    (fun () -> k3 csc logits #sc #l0);
  with l1. assert on gpu_loc (logits |-> l1);
  on_pure_elim gpu_loc (k3facts sc l1);
  final_rel st sn sr h1 l1 sb scf;
  gpu_array_free part;
  gpu_array_free bsc;
  gpu_array_free csc;
}
