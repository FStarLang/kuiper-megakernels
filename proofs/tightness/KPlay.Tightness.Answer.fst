module KPlay.Tightness.Answer

(* A program's complete answer, written out flat: the exact part (tokens,
   counts, ...) and the list of all floating-point output values. *)
noeq type answer (e:Type0) (n:Type0) (count:nat) = {
  exact : e;
  numbers : s:FStar.Seq.seq n{FStar.Seq.length s == count};
}
