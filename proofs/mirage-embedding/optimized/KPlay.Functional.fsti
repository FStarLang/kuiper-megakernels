module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module I64 = FStar.Int64
module A = Pulse.Lib.Array
module SS = KPlay.SharedSpec

(* Forces traversal of this interface for inlining (FOOTGUNS). *)
inline_for_extraction let () = ()

(* optimized/embedding: input0 = embedding table bf16[151936*1024], input1 = prompt
   tokens i64[7], input2 = RMSNorm weight bf16[1024]; outputs residual bf16[8*1024],
   normalized bf16[8*1024], rotary cos/sin bf16[4096*128]. Only ownership and shapes
   are required; the numerical semantics is the shared family relation. *)
fn kplay_run
  (a0 : A.larray bf16 155582464{is_global a0})
  (a1 : A.larray I64.t 7{is_global a1})
  (a2 : A.larray bf16 1024{is_global a2})
  (o0 : A.larray bf16 8192{is_global o0})
  (o1 : A.larray bf16 8192{is_global o1})
  (o2 : A.larray bf16 524288{is_global o2})
  (o3 : A.larray bf16 524288{is_global o3})
  (#p0 #p1 #p2 : perm)
  (#s0 : erased (S.seq bf16))
  (#s1 : erased (S.seq I64.t))
  (#s2 : erased (S.seq bf16))
  preserves cpu ** on gpu_loc (A.pts_to a0 #p0 s0) ** on gpu_loc (A.pts_to a1 #p1 s1) **
            on gpu_loc (A.pts_to a2 #p2 s2)
  requires exists* b0 b1 b2 b3.
    on gpu_loc (A.pts_to o0 b0) ** on gpu_loc (A.pts_to o1 b1) **
    on gpu_loc (A.pts_to o2 b2) ** on gpu_loc (A.pts_to o3 b3)
  ensures exists* r0 r1 r2 r3.
    on gpu_loc (A.pts_to o0 r0) ** on gpu_loc (A.pts_to o1 r1) **
    on gpu_loc (A.pts_to o2 r2) ** on gpu_loc (A.pts_to o3 r3) **
    pure (SS.optimized_embedding_relation s0 s1 s2 r0 r1 r2 r3)
