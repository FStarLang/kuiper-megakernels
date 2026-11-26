module KPlay.Functional

#lang-pulse

open Kuiper
open Kuiper.Tensor
open Kuiper.Tensor.Layout.Alg { l1_forward, l2_row_major }
open KPlay.SharedSpec

(* ABI order: tokens [256,512] bf16, weights [512,8] bf16 (inputs);
   scores [256,8] f32, counts [8] i32, token_indices [8,32] i64 (flat 256),
   probabilities [8,32] f32 (flat 256), survival [256,8] i32 (flat 2048). *)
fn kplay_run
  (x : array2 bf16 (l2_row_major 256 512) { is_global x })
  (w : array2 bf16 (l2_row_major 512 8) { is_global w })
  (sc : array2 float (l2_row_major 256 8) { is_global sc })
  (cn : array1 i32 (l1_forward 8) { is_global cn })
  (ix : array1 i64 (l1_forward 256) { is_global ix })
  (pr : array1 f32 (l1_forward 256) { is_global pr })
  (sv : array1 i32 (l1_forward 2048) { is_global sv })
  (#fx #fw : perm) (#xs : chest2 bf16 256 512) (#ws : chest2 bf16 512 8)
  (#s0 : chest2 float 256 8)
  (#cn0 : chest1 i32 8) (#ix0 : chest1 i64 256) (#pr0 : chest1 f32 256) (#sv0 : chest1 i32 2048)
  preserves cpu ** on gpu_loc (x |-> Frac fx xs) ** on gpu_loc (w |-> Frac fw ws)
  requires on gpu_loc (sc |-> s0) ** on gpu_loc (cn |-> cn0) ** on gpu_loc (ix |-> ix0) **
           on gpu_loc (pr |-> pr0) ** on gpu_loc (sv |-> sv0)
  ensures exists* (scs : chest2 float 256 8) (c : chest1 i32 8) (i : chest1 i64 256)
                  (p : chest1 f32 256) (s : chest1 i32 2048).
    on gpu_loc (sc |-> scs) ** on gpu_loc (cn |-> c) ** on gpu_loc (ix |-> i) **
    on gpu_loc (pr |-> p) ** on gpu_loc (sv |-> s) **
    pure (optimized_projection_relation
            (to_seq (l2_row_major 256 512) xs) (to_seq (l2_row_major 512 8) ws)
            (to_seq (l2_row_major 256 8) scs)
            (to_seq (l1_forward 8) c) (to_seq (l1_forward 256) i)
            (to_seq (l1_forward 256) p) (to_seq (l1_forward 2048) s))
