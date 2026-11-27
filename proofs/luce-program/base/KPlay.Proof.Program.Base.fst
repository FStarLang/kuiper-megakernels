module KPlay.Proof.Program.Base

(* Composition theorem for the base program. Structure: BaseShape (lengths,
   tokens, history; induction over positions), BaseChain (numerics; layer
   chain and position chain), here only the final assembly per output row. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Base
open KPlay.Proof.BaseAccess
open KPlay.Proof.BaseShape
open KPlay.Proof.BaseChain
module S = FStar.Seq
module BF = Kuiper.BFloat16
module I64 = FStar.Int64

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

let tok_lens (r:run)
  : Lemma (requires tok_inv r 11)
          (ensures S.length (utok r 11) == r.input2 /\ S.length (uhist r 11) == r.input2 * 1024)
  = reveal_opaque (`%tok_inv) (tok_inv r 11)

let hist_at (r:run) (k:nat)
  : Lemma (requires tok_inv r 11 /\ k < r.input2 * 1024 /\ pP r - 1 + k / 1024 <= 11)
          (ensures S.length (uhist r 11) == r.input2 * 1024 /\ S.length (hrow r (pP r - 1 + k / 1024)) == 1024 /\
                   S.index (uhist r 11) k == S.index (hrow r (pP r - 1 + k / 1024)) (k % 1024))
  = reveal_opaque (`%tok_inv) (tok_inv r 11)

let row_pos (r:run) (j:nat)
  : Lemma (requires premise r /\ j < r.input2)
          (ensures 0 <= pP r - 1 + j /\ pP r - 1 + j < cap r /\ cap r <= 12)
  = fam_lengths r

let hid_entry (r:run) (j:nat) (i:nat)
  : Lemma (requires premise r /\ j < r.input2 /\ i < 1024 /\ tok_inv r 11)
          (ensures S.length (uhist r 11) == r.input2 * 1024 /\ 0 <= j * 1024 /\ j * 1024 + i < r.input2 * 1024 /\
                   S.length (lx r (pP r - 1 + j) 24) == 1024 /\
                   S.index (uhist r 11) (j * 1024 + i) == S.index (lx r (pP r - 1 + j) 24) i)
  = row_pos r j;
    mul_bound j r.input2 1024;
    dm1024 j i;
    let t : nat = pP r - 1 + j in
    pos_shape r t;
    hist_at r (j * 1024 + i)

let hid_row (r:run) (j:nat)
  : Lemma (requires premise r /\ j < r.input2)
          (ensures bf_row (output1 r) j == lx r (pP r - 1 + j) 24)
  = final_tokens r;
    outputs_ok r;
    tok_lens r;
    row_pos r j;
    pos_shape r (pP r - 1 + j);
    mul_bound j r.input2 1024;
    let h = output1 r in
    let row = S.slice h (j * 1024) (j * 1024 + 1024) in
    let aux (i:nat{i < 1024}) : Lemma (S.index row i == S.index (lx r (pP r - 1 + j) 24) i) =
      hid_entry r j i
    in
    Classical.forall_intro aux;
    S.lemma_eq_intro row (lx r (pP r - 1 + j) 24)

let tok_out (r:run) (j:nat)
  : Lemma (requires premise r /\ j < r.input2)
          (ensures valid_token (tok_at (output0 r) j) /\ tok_at (output0 r) j == sel r (pP r - 1 + j))
  = final_tokens r;
    outputs_ok r;
    row_pos r j;
    tok_inv_at r 11 j

let sel_ok (r:run) (rw:S.seq real) (j:nat)
  : Lemma (requires ctx r rw /\ j < r.input2)
          (ensures selected_ok base_numerics rw (ref_hidden base_numerics rw (toks r) (pP r - 1 + j)) (tok_at (output0 r) j))
  = row_pos r j;
    let t : nat = pP r - 1 + j in
    pos_ok r rw t;
    pos_shape r t;
    argmax_facts r t;
    tok_out r j;
    let lg = ulog r t in
    assert (lg %~ ref_logits base_numerics rw (ref_hidden base_numerics rw (toks r) t) /\
            allowed_choice lg (tok_at (output0 r) j))

let row_ok (r:run) (rw:S.seq real) (j:nat)
  : Lemma (requires ctx r rw /\ j < r.input2)
          (ensures bf_row (output1 r) j %~ ref_hidden base_numerics rw (toks r) (pP r - 1 + j) /\
                   selected_ok base_numerics rw (ref_hidden base_numerics rw (toks r) (pP r - 1 + j)) (tok_at (output0 r) j))
  = row_pos r j;
    pos_ok r rw (pP r - 1 + j);
    hid_row r j;
    sel_ok r rw j

let rows_ok (r:run)
  : Lemma (requires premise r)
          (ensures forall (rw:S.seq real). r.input0 %~ rw ==>
                     (forall (j:nat). j < r.input2 ==>
                        bf_row (output1 r) j %~ ref_hidden base_numerics rw (toks r) (pP r - 1 + j) /\
                        selected_ok base_numerics rw (ref_hidden base_numerics rw (toks r) (pP r - 1 + j)) (tok_at (output0 r) j)))
  = let aux (rw:S.seq real) (j:nat)
      : Lemma (r.input0 %~ rw /\ j < r.input2 ==>
                 bf_row (output1 r) j %~ ref_hidden base_numerics rw (toks r) (pP r - 1 + j) /\
                 selected_ok base_numerics rw (ref_hidden base_numerics rw (toks r) (pP r - 1 + j)) (tok_at (output0 r) j)) =
      Classical.move_requires (row_ok r rw) j
    in
    Classical.forall_intro_2 aux

let out_tokens (r:run)
  : Lemma (requires premise r)
          (ensures S.length (output0 r) == r.input2 /\ S.length (output1 r) == r.input2 * 1024 /\ valid_tokens (output0 r))
  = final_tokens r;
    outputs_ok r;
    tok_lens r;
    let aux (i:nat) : Lemma (i < S.length (output0 r) ==> valid_token (I64.v (S.index (output0 r) i))) =
      if i < S.length (output0 r) then tok_out r i
    in
    Classical.forall_intro aux

let body (r:run) (nm:numerics) : prop =
  numerics_ok nm /\
  (1 <= S.length r.input1 ==>
   (forall (rw:S.seq real). r.input0 %~ rw ==>
     (forall (j:nat). j < r.input2 ==>
        bf_row (output1 r) j %~ ref_hidden nm rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j) /\
        selected_ok nm rw (ref_hidden nm rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j))
                    (tok_at (output0 r) j))))

let post_core (r:run)
  : Lemma (requires premise r)
          (ensures S.length (output0 r) == r.input2 /\ S.length (output1 r) == r.input2 * 1024 /\ valid_tokens (output0 r) /\
                   (exists (nm:numerics). numerics_ok nm /\
                     (forall (rw:S.seq real). r.input0 %~ rw ==>
                       (forall (j:nat). j < r.input2 ==>
                          bf_row (output1 r) j %~ ref_hidden nm rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j) /\
                          selected_ok nm rw (ref_hidden nm rw (prog_tok r.input1 (output0 r)) (S.length r.input1 - 1 + j))
                                      (tok_at (output0 r) j)))))
  = out_tokens r;
    rows_ok r;
    reveal_opaque (`%toks) (toks r);
    assert (pP r == S.length r.input1);
    assert (body r base_numerics);
    Classical.exists_intro (body r) base_numerics

let post_pp (r:run)
  : Lemma (requires premise r)
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
  = post_core r

let refines (r:run)
  : Lemma (requires input_lengths r /\ family_ok r)
          (ensures program_postcondition r.input0 r.input1 r.input2 (output0 r) (output1 r))
  = reveal_opaque (`%fam) (fam r);
    Classical.move_requires post_pp r
