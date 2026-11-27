module KPlay.Functional

#lang-pulse

open Kuiper
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF = Kuiper.BFloat16
module A = Pulse.Lib.Array
module I64 = FStar.Int64

fn kplay_run
  (w : A.array BF.t{is_global w}) (x : A.array F32.t{is_global x})
  (rc : A.array F32.t{is_global rc}) (cv : A.array F32.t{is_global cv})
  (xo : A.array F32.t{is_global xo}) (rco : A.array F32.t{is_global rco})
  (cvo : A.array F32.t{is_global cvo})
  (wn : SZ.t) (base dn : I64.t)
  (#ws : erased (Seq.seq BF.t)) (#xs #rs #cs : erased (Seq.seq F32.t))
  preserves cpu ** on gpu_loc (w |-> ws) ** on gpu_loc (x |-> xs) ** on gpu_loc (rc |-> rs) ** on gpu_loc (cv |-> cs)
  requires (exists* a. on gpu_loc (xo |-> a) ** pure (Seq.length a == 1024)) **
           (exists* b. on gpu_loc (rco |-> b) ** pure (Seq.length b == 4718592)) **
           (exists* c. on gpu_loc (cvo |-> c) ** pure (Seq.length c == 442368)) **
           pure (Seq.length ws == SZ.v wn /\ Seq.length xs == 1024 /\ Seq.length rs == 4718592 /\ Seq.length cs == 442368)
  ensures exists* a b c. on gpu_loc (xo |-> a) ** on gpu_loc (rco |-> b) ** on gpu_loc (cvo |-> c) **
    pure (KPlay.SharedSpec.optimized_recurrent_decoder_relation ws xs rs cs (I64.v base) (I64.v dn) a b c)
