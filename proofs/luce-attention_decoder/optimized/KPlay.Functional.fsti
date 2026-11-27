module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64

fn kplay_run
  (w : array bf16{is_global w}) (x : larray f32 1024{is_global x})
  (k : larray bf16 6291456{is_global k}) (v : larray bf16 6291456{is_global v})
  (cs : larray f32 32{is_global cs}) (sn : larray f32 32{is_global sn})
  (xo : larray f32 1024{is_global xo})
  (ko : larray bf16 6291456{is_global ko}) (vo : larray bf16 6291456{is_global vo})
  (wn : SZ.t) (base fa pos : I64.t)
  (#sw : erased (S.seq bf16)) (#sx : erased (S.seq f32))
  (#sk #sv : erased (S.seq bf16)) (#sc #ss : erased (S.seq f32))
  preserves cpu ** on gpu_loc (w |-> sw) ** on gpu_loc (x |-> sx) **
    on gpu_loc (k |-> sk) ** on gpu_loc (v |-> sv) **
    on gpu_loc (cs |-> sc) ** on gpu_loc (sn |-> ss)
  requires pure (SZ.v wn == S.length sw /\ S.length sx == 1024 /\ S.length sk == 6291456 /\ S.length sv == 6291456 /\ S.length sc == 32 /\ S.length ss == 32) **
    (exists* s. on gpu_loc (xo |-> s)) ** (exists* s. on gpu_loc (ko |-> s)) ** (exists* s. on gpu_loc (vo |-> s))
  ensures exists* sx' sk' sv'.
    on gpu_loc (xo |-> sx') ** on gpu_loc (ko |-> sk') ** on gpu_loc (vo |-> sv') **
    pure (optimized_attention_decoder_relation sw sx sk sv sc ss (I64.v base) (I64.v fa) (I64.v pos) sx' sk' sv')
