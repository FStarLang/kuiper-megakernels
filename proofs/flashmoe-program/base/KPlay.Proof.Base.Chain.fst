module KPlay.Proof.Base.Chain

(* ONE size-generic chain lemma for the base routing loop (G4): a trace of
   per-token boundary values (argmax outputs and routing state after each
   token), with every token step related by the base family relations,
   yields the closed-form routing state after n tokens, by induction on n. *)

open Kuiper
open KPlay.SharedSpec
open KPlay.Proof.Base.Select
module S = FStar.Seq
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

#set-options "--z3rlimit 20 --fuel 0 --ifuel 1"

noeq type trace = {
  rows : S.seq F32.t;             (* softmax rows consumed by argmax, [256, 8] *)
  selv : nat -> S.seq I32.t;      (* argmax index of token t *)
  spv  : nat -> S.seq F32.t;      (* argmax probability of token t *)
  cnt  : nat -> S.seq I64.t;      (* routing state after t tokens *)
  idx  : nat -> S.seq I64.t;
  prb  : nat -> S.seq F32.t;
  srv  : nat -> S.seq I32.t;
}

let step_ok (tr:trace) (t:nat) : prop =
  base_argmax_relation (row_of tr.rows t) (tr.selv t) (tr.spv t) /\
  base_capacity_routing_relation (tr.selv t) (tr.spv t)
    (tr.cnt t) (tr.idx t) (tr.prb t) (tr.srv t) t
    (tr.cnt (t + 1)) (tr.idx (t + 1)) (tr.prb (t + 1)) (tr.srv (t + 1))

let trace_ok (tr:trace) : prop =
  S.length tr.rows == tokens * experts /\
  initial_routing_state (tr.cnt 0) (tr.idx 0) (tr.prb 0) /\
  S.length (tr.srv 0) == 0 /\
  (forall (t:nat). t < tokens ==> step_ok tr t)

let sidx (#a:Type) (s:S.seq a) (i:int) (d:a) : a =
  if 0 <= i && i < S.length s then S.index s i else d

let sel_at (tr:trace) (t:nat) : int = I32.v (sidx (tr.selv t) 0 0l)

let sel_of (tr:trace) : S.seq int =
  S.init tokens (fun (t:nat{t < tokens}) -> sel_at tr t)

(* Combine weight of token t: what the routing step wrote at its slot. *)
let comb_at (tr:trace) (t:nat) : F32.t =
  let s = sel_at tr t in
  sidx (tr.prb (t + 1)) (capacity * s + I64.v (sidx (tr.cnt t) s 0L)) zero

let comb_of (tr:trace) : S.seq F32.t =
  S.init tokens (fun (t:nat{t < tokens}) -> comb_at tr t)

let sel_len (tr:trace)
  : Lemma (S.length (sel_of tr) == tokens) [SMTPat (sel_of tr)] = ()

let comb_len (tr:trace)
  : Lemma (S.length (comb_of tr) == tokens) [SMTPat (comb_of tr)] = ()

let inv (tr:trace) (n:nat{n <= tokens}) : prop =
  routing_state (sel_of tr) (comb_of tr) n (tr.cnt n) (tr.idx n) (tr.prb n) /\
  S.length (tr.srv n) == experts * n /\
  survival_rows (sel_of tr) n (tr.srv n)

let count_v_i64 (x:I64.t) : Lemma (count_v x == I64.v x) = ()

#push-options "--fuel 1"
let rec count_upto_le (sel:S.seq int) (e:int) (n:nat{n <= S.length sel})
  : Lemma (ensures count_upto sel e n <= n) (decreases n)
  = if n = 0 then () else count_upto_le sel e (n - 1)
#pop-options

let step_at (tr:trace) (t:nat{t < tokens})
  : Lemma (requires trace_ok tr) (ensures step_ok tr t)
  = ()

let row_len (rows:S.seq F32.t) (t:nat{t < tokens})
  : Lemma (requires S.length rows == tokens * experts)
          (ensures S.length (row_of rows t) == experts)
  = ()

(* Per-token selection facts: the argmax output is the base scan of the row. *)
let sel_facts (tr:trace) (t:nat{t < tokens})
  : Lemma (requires trace_ok tr)
          (ensures (let row = row_of tr.rows t in
                    S.length row == experts /\
                    S.length (tr.selv t) == 1 /\ S.length (tr.spv t) == 1 /\
                    0 <= sel_at tr t /\ sel_at tr t < experts /\
                    S.index (sel_of tr) t == sel_at tr t /\
                    sel_at tr t == fst (base_scan row experts) /\
                    S.index (tr.spv t) 0 == S.index row (sel_at tr t) /\
                    allowed_choice row (sel_at tr t)))
  = step_at tr t;
    row_len tr.rows t;
    base_scan_allowed (row_of tr.rows t)

let slot_split (k s r:int)
  : Lemma (requires 0 <= k /\ k < slots /\ 0 <= s /\ s < experts /\ 0 <= r /\ r < capacity)
          (ensures (k == capacity * s + r) <==> (k / capacity == s /\ k % capacity == r))
  = FStar.Math.Lemmas.euclidean_division_definition k capacity;
    if k = capacity * s + r then begin
      FStar.Math.Lemmas.lemma_div_plus r s capacity;
      FStar.Math.Lemmas.lemma_mod_plus r s capacity
    end

#push-options "--fuel 1"
let slot_token_step (sel:S.seq int) (e c:int) (n:nat{n < S.length sel})
  : Lemma (slot_token sel e c (n + 1) ==
           (if S.index sel n = e && count_upto sel e n = c && c < capacity
            then n else slot_token sel e c n))
  = ()

let count_upto_step (sel:S.seq int) (e:int) (n:nat{n < S.length sel})
  : Lemma (count_upto sel e (n + 1) == count_upto sel e n + (if S.index sel n = e then 1 else 0))
  = ()
#pop-options

let survival_step (sel:S.seq int) (n:nat{n < S.length sel}) (s r:int) (sv sv':S.seq I32.t)
  : Lemma (requires survival_rows sel n sv /\ S.length sv == experts * n /\
                    S.length sv' == experts * n + experts /\
                    S.index sel n == s /\ (kept sel n <==> r < capacity) /\
                    (forall (k:nat). k < experts * n ==> S.index sv' k == S.index sv k) /\
                    (forall (e:nat). e < experts ==>
                       I32.v (S.index sv' (experts * n + e)) == (if e = s && r < capacity then 1 else 0)))
          (ensures S.length sv' == experts * (n + 1) /\ survival_rows sel (n + 1) sv')
  = introduce forall (t:nat) (e:nat). t < n + 1 /\ e < experts ==>
        I32.v (S.index sv' (experts * t + e)) ==
          (if S.index sel t = e && kept sel t then 1 else 0)
    with introduce _ ==> _ with begin
      if t < n then assert (experts * t + e < experts * n)
    end

#push-options "--z3rlimit 100 --fuel 0 --ifuel 1"
let step_inv (tr:trace) (n:nat{n < tokens})
  : Lemma (requires trace_ok tr /\ inv tr n) (ensures inv tr (n + 1))
  = let sel = sel_of tr in
    let comb = comb_of tr in
    sel_facts tr n;
    step_at tr n;
    let s = sel_at tr n in
    assert (S.index sel n == s);
    let c = tr.cnt n in
    let c' = tr.cnt (n + 1) in
    introduce forall (e:nat). e < experts ==>
        0 <= I64.v (S.index c e) /\ I64.v (S.index c e) <= n
    with introduce _ ==> _ with begin
      count_v_i64 (S.index c e);
      count_upto_le sel e n
    end;
    let r = I64.v (S.index c s) in
    count_v_i64 (S.index c s);
    assert (r == count_upto sel s n);
    let slot = capacity * s + r in
    assert (comb_at tr n == sidx (tr.prb (n + 1)) slot zero);
    assert (S.index comb n == comb_at tr n);
    (* counts *)
    introduce forall (e:nat). e < experts ==>
        count_v (S.index c' e) == count_upto sel e (n + 1)
    with introduce _ ==> _ with begin
      count_v_i64 (S.index c' e);
      count_v_i64 (S.index c e);
      count_upto_step sel e n
    end;
    (* slots *)
    introduce forall (k:nat). k < slots ==>
        I64.v (S.index (tr.idx (n + 1)) k) ==
          slot_token sel (k / capacity) (k % capacity) (n + 1) /\
        S.index (tr.prb (n + 1)) k ==
          (let t = slot_token sel (k / capacity) (k % capacity) (n + 1) in
           if t >= 0 then S.index comb t else zero)
    with introduce _ ==> _ with begin
      slot_token_step sel (k / capacity) (k % capacity) n;
      FStar.Math.Lemmas.euclidean_division_definition k capacity;
      if r < capacity then slot_split k s r
    end;
    (* survival *)
    assert (kept sel n <==> r < capacity);
    survival_step sel n s r (tr.srv n) (tr.srv (n + 1))
#pop-options

#push-options "--fuel 1"
let rec chain (tr:trace) (n:nat{n <= tokens})
  : Lemma (requires trace_ok tr) (ensures inv tr n) (decreases n)
  = if n = 0 then begin
      introduce forall (k:nat). k < slots ==>
          I64.v (S.index (tr.idx 0) k) == slot_token (sel_of tr) (k / capacity) (k % capacity) 0
      with ();
      introduce forall (e:nat). e < experts ==>
          count_v (S.index (tr.cnt 0) e) == count_upto (sel_of tr) e 0
      with ()
    end else begin
      chain tr (n - 1);
      step_inv tr (n - 1)
    end
#pop-options

(* The combine weight of a kept token approximates the reference p / p. *)
#push-options "--z3rlimit 50 --fuel 0 --ifuel 1"
let comb_ok (tr:trace) (t:nat{t < tokens})
  : Lemma (requires trace_ok tr /\ kept (sel_of tr) t)
          (ensures selected_combine_ok (row_of tr.rows t) (S.index (sel_of tr) t)
                                      (S.index (comb_of tr) t))
  = chain tr t;
    sel_facts tr t;
    step_at tr t;
    let sel = sel_of tr in
    let s = sel_at tr t in
    let c = tr.cnt t in
    introduce forall (e:nat). e < experts ==>
        0 <= I64.v (S.index c e) /\ I64.v (S.index c e) <= t
    with introduce _ ==> _ with begin
      count_v_i64 (S.index c e);
      count_upto_le sel e t
    end;
    count_v_i64 (S.index c s);
    let r = I64.v (S.index c s) in
    assert (r == rank sel t);
    let slot = capacity * s + r in
    assert (S.index (comb_of tr) t == comb_at tr t);
    assert (comb_at tr t == S.index (tr.prb (t + 1)) slot)
#pop-options

let trace_post (tr:trace) : prop =
  S.length (sel_of tr) == tokens /\ S.length (comb_of tr) == tokens /\
  (forall (t:nat). t < tokens ==> allowed_choice (row_of tr.rows t) (S.index (sel_of tr) t)) /\
  (forall (t:nat). t < tokens /\ kept (sel_of tr) t ==>
     selected_combine_ok (row_of tr.rows t) (S.index (sel_of tr) t) (S.index (comb_of tr) t)) /\
  routing_state (sel_of tr) (comb_of tr) tokens (tr.cnt tokens) (tr.idx tokens) (tr.prb tokens) /\
  S.length (tr.srv tokens) == tokens * experts /\
  survival_rows (sel_of tr) tokens (tr.srv tokens)

let comb_ok' (tr:trace) (t:nat)
  : Lemma (requires trace_ok tr /\ t < tokens /\ kept (sel_of tr) t)
          (ensures selected_combine_ok (row_of tr.rows t) (S.index (sel_of tr) t)
                                      (S.index (comb_of tr) t))
  = comb_ok tr t

let sel_ok' (tr:trace) (t:nat)
  : Lemma (requires trace_ok tr /\ t < tokens)
          (ensures allowed_choice (row_of tr.rows t) (S.index (sel_of tr) t))
  = sel_facts tr t

let sel_all (tr:trace)
  : Lemma (requires trace_ok tr)
          (ensures forall (t:nat). t < tokens ==>
                     allowed_choice (row_of tr.rows t) (S.index (sel_of tr) t))
  = introduce forall (t:nat). t < tokens ==>
        allowed_choice (row_of tr.rows t) (S.index (sel_of tr) t)
    with introduce _ ==> _ with sel_ok' tr t

let comb_all (tr:trace)
  : Lemma (requires trace_ok tr)
          (ensures forall (t:nat). t < tokens /\ kept (sel_of tr) t ==>
                     selected_combine_ok (row_of tr.rows t) (S.index (sel_of tr) t)
                                         (S.index (comb_of tr) t))
  = introduce forall (t:nat). t < tokens /\ kept (sel_of tr) t ==>
        selected_combine_ok (row_of tr.rows t) (S.index (sel_of tr) t) (S.index (comb_of tr) t)
    with introduce _ ==> _ with comb_ok' tr t

let trace_refines (tr:trace)
  : Lemma (requires trace_ok tr) (ensures trace_post tr)
  = sel_all tr;
    comb_all tr;
    chain tr tokens;
    FStar.Math.Lemmas.swap_mul experts tokens
