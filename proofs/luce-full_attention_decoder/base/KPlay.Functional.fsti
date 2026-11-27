module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module BF = Kuiper.BFloat16

(* ABI: nkv = numel(input2); input0 weights, input1 residual, input2 KV state,
   input3 cos, input4 sin, input5 position t; output0 residual', output1 KV'.
   Postcondition: base_full_attention_decoder_relation. *)
fn kplay_run
  (nkv : sz)
  (w : larray bf16 18352640 { is_global w })
  (x : larray bf16 1024 { is_global x })
  (kv : larray bf16 (SZ.v nkv) { is_global kv })
  (cs : larray bf16 32 { is_global cs })
  (sn : larray bf16 32 { is_global sn })
  (t : i64)
  (xo : larray bf16 1024 { is_global xo })
  (kvo : larray bf16 (SZ.v nkv) { is_global kvo })
  (#pw #px #pkv #pc #ps : perm)
  (#sw #sx #skv #sc #ss : erased (S.seq bf16))
  (#ox #okv : erased (S.seq bf16))
  preserves
    cpu **
    on gpu_loc (w |-> Frac pw sw) **
    on gpu_loc (x |-> Frac px sx) **
    on gpu_loc (kv |-> Frac pkv skv) **
    on gpu_loc (cs |-> Frac pc sc) **
    on gpu_loc (sn |-> Frac ps ss)
  requires
    on gpu_loc (xo |-> ox) **
    on gpu_loc (kvo |-> okv)
  ensures
    exists* (x' kv' : S.seq bf16).
      on gpu_loc (xo |-> x') **
      on gpu_loc (kvo |-> kv') **
      pure (base_full_attention_decoder_relation sw sx skv sc ss (I64.v t) x' kv')
