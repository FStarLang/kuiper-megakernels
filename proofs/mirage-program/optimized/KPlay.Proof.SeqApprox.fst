module KPlay.Proof.SeqApprox

(* Size-generic structural facts about windows and %~ on BF16 sequences. *)

open FStar.Real
open Kuiper.Approximates
open KPlay.SharedSpec
module S = FStar.Seq
module BF = Kuiper.BFloat16
module W = KPlay.Proof.Wiring.Optimized

let view_window (#a:Type) (s:S.seq a) (o n:nat)
  : Lemma (W.view s o n == window s o n)
  = ()

let approx_index (s:S.seq BF.t) (r:S.seq real) (i:nat)
  : Lemma (requires s %~ r /\ i < S.length s)
          (ensures S.index s i %~ S.index r i)
  = ()

(* A window of an approximated sequence approximates the window of the reals. *)
let window_approx (s:S.seq BF.t) (r:S.seq real) (o n:int)
  : Lemma (requires s %~ r)
          (ensures window s o n %~ window r o n)
  = if 0 <= o && 0 <= n && o + n <= S.length s then begin
      let ws = window s o n in
      let wr = window r o n in
      let aux (i:nat{i < S.length ws}) : Lemma (S.index ws i %~ S.index wr i)
        = approx_index s r (o + i)
      in
      Classical.forall_intro aux
    end

let view_approx (s:S.seq BF.t) (r:S.seq real) (o n:nat)
  : Lemma (requires s %~ r)
          (ensures W.view s o n %~ window r o n)
  = window_approx s r o n

(* A sequence whose prefix of length n approximates h approximates some
   real sequence with prefix h (the tail is approximated by its own reals). *)
let extend_real (s:S.seq BF.t) (n:nat{n <= S.length s}) (h:S.seq real)
  : GTot (S.seq real)
  = S.append h (to_real_seq (window s n (S.length s - n)))

let extend_approx (s:S.seq BF.t) (n:nat) (h:S.seq real)
  : Lemma (requires n <= S.length s /\ window s 0 n %~ h)
          (ensures s %~ extend_real s n h /\ window (extend_real s n h) 0 n == h)
  = let t = window s n (S.length s - n) in
    let rt = to_real_seq t in
    to_real_seq_is_approx t;
    let e = S.append h rt in
    let aux (i:nat{i < S.length s}) : Lemma (S.index s i %~ S.index e i)
      = if i < n then approx_index (window s 0 n) h i
        else approx_index t rt (i - n)
    in
    Classical.forall_intro aux;
    S.lemma_eq_intro (window e 0 n) h

let window_extend (s:S.seq BF.t) (n:nat{n <= S.length s}) (h:S.seq real) (o m:nat)
  : Lemma (requires S.length h == n /\ o + m <= n)
          (ensures window (extend_real s n h) o m == window h o m)
  = S.lemma_eq_intro (window (extend_real s n h) o m) (window h o m)

(* The logit row restricted to the first n of m columns. *)
let logits_row_prefix (n m:nat) (we x:S.seq real)
  : Lemma (requires n <= m)
          (ensures window (logits_row m we x) 0 n == logits_row n we x)
  = reveal_opaque (`%logits_row) logits_row;
    S.lemma_eq_intro (window (logits_row m we x) 0 n) (logits_row n we x)
