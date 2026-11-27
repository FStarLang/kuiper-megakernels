module KPlay.Tightness.Answer

(* A program's complete answer, written out flat: the exact part (tokens,
   counts, ...) and the list of all floating-point output values. How many
   values there are may depend on the exact part (e.g. tokens generated). *)
noeq type answer (e:Type0) (n:Type0) (count:e -> nat) = {
  exact : e;
  numbers : s:FStar.Seq.seq n{FStar.Seq.length s == count exact};
}
