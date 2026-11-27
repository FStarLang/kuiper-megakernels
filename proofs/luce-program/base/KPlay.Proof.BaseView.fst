module KPlay.Proof.BaseView

(* Views of approximated sequences, zero-initialised buffers, footprint
   agreement of real views. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base
open KPlay.Proof.BaseCongr
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

let zero_bf () : Lemma ((zero #BF.t) %~ 0.0R) = a0 #BF.t
let zero_f32 () : Lemma ((zero #F32.t) %~ 0.0R) = a0 #F32.t

let all_zero_bf (s:S.seq BF.t)
  : Lemma (requires all_zero s) (ensures s %~ S.create (S.length s) 0.0R)
  = zero_bf ()

let all_zero_f32 (s:S.seq F32.t)
  : Lemma (requires all_zero s) (ensures s %~ S.create (S.length s) 0.0R)
  = zero_f32 ()

let view_index (#a:Type) (s:S.seq a) (o n:int) (i:nat)
  : Lemma (requires 0 <= o /\ 0 <= n /\ o + n <= S.length s /\ i < n)
          (ensures S.length (view s o n) == n /\ S.index (view s o n) i == S.index s (o + i))
  = ()

let view_len (#a:Type) (s:S.seq a) (o n:int)
  : Lemma (requires 0 <= o /\ 0 <= n /\ o + n <= S.length s)
          (ensures S.length (view s o n) == n)
  = ()

let view_approx_bf (s:S.seq BF.t) (r:S.seq real) (o n:int)
  : Lemma (requires s %~ r /\ 0 <= o /\ 0 <= n /\ o + n <= S.length s)
          (ensures view s o n %~ view r o n /\ S.length (view s o n) == n)
  = let aux (i:nat{i < n}) : Lemma (S.index (view s o n) i %~ S.index (view r o n) i) =
      view_index s o n i; view_index r o n i
    in
    Classical.forall_intro aux

let view_wagree (r:S.seq real) (o n:int)
  : Lemma (requires 0 <= o /\ 0 <= n /\ o + n <= S.length r)
          (ensures wagree (view r o n) 0 r o n)
  = let aux (i:int) : Lemma (0 <= i /\ i < n ==> at (view r o n) i == at r (o + i)) =
      if 0 <= i && i < n then view_index r o n i
    in
    Classical.forall_intro aux
