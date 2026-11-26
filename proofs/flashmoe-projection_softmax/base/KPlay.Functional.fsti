module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec

(* Physical interface of the base projection_softmax family: flat row-major
   buffers x bf16[256*512], w bf16[512*8], softmax f32[256*8], prefix i64[8],
   token_indices i64[8*32], probabilities f32[8*32], survival i32[0*8]. *)

inline_for_extraction noextract
let x_len : nat = 131072

inline_for_extraction noextract
let w_len : nat = 4096

inline_for_extraction noextract
let sm_len : nat = 2048

inline_for_extraction noextract
let prefix_len : nat = 8

inline_for_extraction noextract
let slot_len : nat = 256

inline_for_extraction noextract
let surv_len : nat = 0

fn kplay_run
  (x : larray bf16 x_len{is_global x})
  (w : larray bf16 w_len{is_global w})
  (softmax : larray f32 sm_len{is_global softmax})
  (prefix : larray i64 prefix_len{is_global prefix})
  (token_indices : larray i64 slot_len{is_global token_indices})
  (probabilities : larray f32 slot_len{is_global probabilities})
  (survival : larray i32 surv_len{is_global survival})
  (#vx : erased (lseq bf16 x_len))
  (#vw : erased (lseq bf16 w_len))
  (#vsurv : erased (lseq i32 surv_len))
  preserves cpu ** on gpu_loc (x |-> vx) ** on gpu_loc (w |-> vw) **
            on gpu_loc (survival |-> vsurv)
  requires exists* (s0 : seq f32) (p0 : seq i64) (i0 : seq i64) (q0 : seq f32).
    on gpu_loc (softmax |-> s0) ** on gpu_loc (prefix |-> p0) **
    on gpu_loc (token_indices |-> i0) ** on gpu_loc (probabilities |-> q0)
  ensures exists* (s : seq f32) (p : seq i64) (i : seq i64) (q : seq f32).
    on gpu_loc (softmax |-> s) ** on gpu_loc (prefix |-> p) **
    on gpu_loc (token_indices |-> i) ** on gpu_loc (probabilities |-> q) **
    pure (base_projection_softmax_relation vx vw s p i q vsurv)
