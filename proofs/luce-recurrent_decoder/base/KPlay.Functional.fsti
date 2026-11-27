module KPlay.Functional

#lang-pulse

module S = FStar.Seq
module SZ = Kuiper.SizeT
module M = Kuiper.TensorCore.MMA
module RO = Kuiper.Float.Realops
module C = Kuiper.Float.Casts
module CF = Kuiper.Float32.ConstantFold
module Cert = Kuiper.Float32.ConstantFold.Certificates
module Lit = Kuiper.Float32.Literals
module Exp = FStar.Math.Exp
module Sqrt = FStar.Math.Sqrt
module L = FStar.Math.Lemmas
module SP = Luce.Softplus
module FM = Kuiper.Float32.FastMath
module A = Pulse.Lib.Array

open Kuiper
open KPlay.SharedSpec

fn kplay_run
  (w : larray bf16 21555360 { is_global w })
  (x : larray bf16 1024 { is_global x })
  (conv : larray bf16 24576 { is_global conv })
  (st : larray float 262144 { is_global st })
  (out0 : larray bf16 1024 { is_global out0 })
  (out1 : larray bf16 24576 { is_global out1 })
  (out2 : larray float 262144 { is_global out2 })
  (#pw #px #pc #ps : perm)
  (#sw #sx #sc : erased (S.seq bf16)) (#ss : erased (S.seq float))
  (#o0 #o1 : erased (S.seq bf16)) (#o2 : erased (S.seq float))
  preserves cpu ** on gpu_loc (w |-> Frac pw sw) ** on gpu_loc (x |-> Frac px sx) **
    on gpu_loc (conv |-> Frac pc sc) ** on gpu_loc (st |-> Frac ps ss)
  requires on gpu_loc (out0 |-> o0) ** on gpu_loc (out1 |-> o1) ** on gpu_loc (out2 |-> o2)
  ensures exists* (a b : S.seq bf16) (c : S.seq float).
    on gpu_loc (out0 |-> a) ** on gpu_loc (out1 |-> b) ** on gpu_loc (out2 |-> c) **
    pure (base_recurrent_decoder_relation sw sx sc ss a b c)
