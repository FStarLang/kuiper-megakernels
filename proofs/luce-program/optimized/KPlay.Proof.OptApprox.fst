module KPlay.Proof.OptApprox

(* Approximation facts used by the composition: zero buffers approximate
   zero reals, and a weight view approximates the matching real slice. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Optimized
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32

#set-options "--fuel 0 --ifuel 0 --z3rlimit 50"

let zero_approx_f32 (s:S.seq F32.t)
  : Lemma (requires all_zero s) (ensures s %~ S.create (S.length s) 0.0R)
  = let a0 : squash (v_approximates #F32.t (zero #F32.t) 0.0R) = a0 #F32.t in
    ()

let zero_approx_bf (s:S.seq BF.t)
  : Lemma (requires all_zero s) (ensures s %~ S.create (S.length s) 0.0R)
  = let a0 : squash (v_approximates #BF.t (zero #BF.t) 0.0R) = a0 #BF.t in
    ()

let view_approx (w:S.seq BF.t) (rw:S.seq real) (off len:nat)
  : Lemma (requires w %~ rw /\ off + len <= S.length w)
          (ensures view w off len %~ S.slice rw off (off + len))
  = ()
