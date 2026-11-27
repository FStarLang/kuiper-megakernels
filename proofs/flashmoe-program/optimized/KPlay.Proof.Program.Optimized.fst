module KPlay.Proof.Program.Optimized

open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
open KPlay.Proof.Wiring.Optimized
open KPlay.Proof.OptScan
module S = FStar.Seq
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64

#set-options "--z3rlimit 50 --fuel 1 --ifuel 1"

(* ------------------------------------------------------------------ *)
(* Uniform per-token view of the execution record: token 0 is the fixed
   calls, token t >= 1 is loop iteration t - 1. st_* n is the routing state
   after n tokens (st_* 0 = projection initialization). *)

let soft (r:run) (t:nat) : S.seq F32.t =
  if t = 0 then r.sample_0000_token_000_softmax_0 else r.loop_sample_0000_token_i_softmax_0 (t - 1)

let asel (r:run) (t:nat) : S.seq I32.t =
  if t = 0 then r.sample_0000_token_000_argmax_0 else r.loop_sample_0000_token_i_argmax_0 (t - 1)

let aprob (r:run) (t:nat) : S.seq F32.t =
  if t = 0 then r.sample_0000_token_000_argmax_1 else r.loop_sample_0000_token_i_argmax_1 (t - 1)

let st_cnt (r:run) (n:nat) : S.seq I32.t =
  if n = 0 then r.sample_0000_projection_1
  else if n = 1 then r.sample_0000_token_000_routing_0
  else r.loop_sample_0000_token_i_routing_0 (n - 2)

let st_idx (r:run) (n:nat) : S.seq I64.t =
  if n = 0 then r.sample_0000_projection_2
  else if n = 1 then r.sample_0000_token_000_routing_1
  else r.loop_sample_0000_token_i_routing_1 (n - 2)

let st_prb (r:run) (n:nat) : S.seq F32.t =
  if n = 0 then r.sample_0000_projection_3
  else if n = 1 then r.sample_0000_token_000_routing_2
  else r.loop_sample_0000_token_i_routing_2 (n - 2)

let st_srv (r:run) (n:nat) : S.seq I32.t =
  if n = 0 then r.sample_0000_projection_4
  else if n = 1 then r.sample_0000_token_000_routing_3
  else r.loop_sample_0000_token_i_routing_3 (n - 2)

let scores (r:run) : S.seq F32.t = r.sample_0000_projection_0

let token_ok (r:run) (t:nat) : prop =
  optimized_softmax_relation (scores r) t (soft r t) /\
  optimized_argmax_relation (soft r t) (asel r t) (aprob r t) /\
  optimized_routing_relation (soft r t) (asel r t) (aprob r t)
    (st_cnt r t) (st_idx r t) (st_prb r t) (st_srv r t) t
    (st_cnt r (t + 1)) (st_idx r (t + 1)) (st_prb r (t + 1)) (st_srv r (t + 1))

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let token_ok_of_family (r:run) (t:nat)
  : Lemma (requires family_ok r /\ t < tokens) (ensures token_ok r t)
  = if t = 0 then ()
    else assert (loop_ok r (t - 1))
#pop-options

(* ------------------------------------------------------------------ *)
(* Witnesses of the program postcondition *)

let sel_at (r:run) (t:nat) : int =
  let s = asel r t in
  if S.length s >= 1 then I32.v (S.index s 0) else 0

let sel_seq (r:run) : S.seq int = S.init tokens (fun t -> sel_at r t)

let rows_seq (r:run) : S.seq F32.t =
  S.init (tokens * experts) (fun k ->
    let row = soft r (k / experts) in
    if S.length row = experts then S.index row (k % experts) else zero)

let comb_at (r:run) (t:nat) : F32.t =
  let s = sel_at r t in
  let c = st_cnt r t in
  let p = st_prb r (t + 1) in
  if 0 <= s && s < experts && S.length c = experts then
    let slot = capacity * s + I32.v (S.index c s) in
    if 0 <= slot && slot < S.length p then S.index p slot else zero
  else zero

let comb_seq (r:run) : S.seq F32.t = S.init tokens (fun t -> comb_at r t)

let rows_row (r:run) (t:nat{t < tokens})
  : Lemma (requires S.length (soft r t) == experts)
          (ensures row_of (rows_seq r) t == soft r t)
  = let a = row_of (rows_seq r) t in
    let aux (e:nat{e < experts}) : Lemma (S.index a e == S.index (soft r t) e) =
      assert ((experts * t + e) / experts == t);
      assert ((experts * t + e) % experts == e)
    in
    FStar.Classical.forall_intro aux;
    S.lemma_eq_intro a (soft r t)

let rec count_upto_le (sel:S.seq int) (e:int) (n:nat{n <= S.length sel})
  : Lemma (ensures count_upto sel e n <= n) (decreases n)
  = if n = 0 then () else count_upto_le sel e (n - 1)

(* ------------------------------------------------------------------ *)
(* Chain invariant after n tokens *)

let surv_inv (sel:S.seq int{S.length sel == tokens}) (n:nat{n <= tokens}) (surv:S.seq I32.t) : prop =
  S.length surv == tokens * experts /\
  (forall (k:nat). k < tokens * experts ==>
     I32.v (S.index surv k) ==
       (if k / experts < n && S.index sel (k / experts) = k % experts && kept sel (k / experts)
        then 1 else 0))

let token_post (r:run) (t:nat{t < tokens}) : prop =
  row_of (rows_seq r) t == soft r t /\
  allowed_choice (row_of (rows_seq r) t) (S.index (sel_seq r) t) /\
  (kept (sel_seq r) t ==>
     selected_combine_ok (row_of (rows_seq r) t) (S.index (sel_seq r) t) (S.index (comb_seq r) t))

(* Opaque wrappers keep the chain contexts small (L9, L13). *)
[@@"opaque_to_smt"]
let rs_ok (r:run) (n:nat{n <= tokens}) : prop =
  routing_state (sel_seq r) (comb_seq r) n (st_cnt r n) (st_idx r n) (st_prb r n)

[@@"opaque_to_smt"]
let sv_ok (r:run) (n:nat{n <= tokens}) : prop = surv_inv (sel_seq r) n (st_srv r n)

[@@"opaque_to_smt"]
let tp (r:run) (t:nat{t < tokens}) : prop = token_post r t

let rs_reveal (r:run) (n:nat{n <= tokens})
  : Lemma (rs_ok r n <==>
           routing_state (sel_seq r) (comb_seq r) n (st_cnt r n) (st_idx r n) (st_prb r n))
  = reveal_opaque (`%rs_ok) (rs_ok r n)

let sv_reveal (r:run) (n:nat{n <= tokens})
  : Lemma (sv_ok r n <==> surv_inv (sel_seq r) n (st_srv r n))
  = reveal_opaque (`%sv_ok) (sv_ok r n)

let tp_reveal (r:run) (t:nat{t < tokens}) : Lemma (tp r t <==> token_post r t)
  = reveal_opaque (`%tp) (tp r t)

let inv (r:run) (n:nat{n <= tokens}) : prop =
  rs_ok r n /\ sv_ok r n /\ (forall (t:nat). t < n ==> tp r t)

let count_v_i32 (x:I32.t) : Lemma (count_v x == I32.v x) = ()

#push-options "--z3rlimit 50 --fuel 1 --ifuel 1"
let inv_zero (r:run)
  : Lemma (requires initial_routing_state (st_cnt r 0) (st_idx r 0) (st_prb r 0) /\
                    S.length (st_srv r 0) == tokens * experts /\
                    (forall (k:nat). k < tokens * experts ==> I32.v (S.index (st_srv r 0) k) == 0))
          (ensures inv r 0)
  = rs_reveal r 0; sv_reveal r 0
#pop-options

let slot_iff (k s rk:int)
  : Lemma (requires 0 <= k /\ k < slots /\ 0 <= s /\ s < experts /\ 0 <= rk)
          (ensures (rk < capacity && k = capacity * s + rk) <==>
                   (k / capacity = s /\ k % capacity = rk /\ rk < capacity))
  = ()

let surv_iff (k n s:int)
  : Lemma (requires 0 <= k /\ k < tokens * experts /\ 0 <= s /\ s < experts /\ 0 <= n)
          (ensures (k = experts * n + s) <==> (k / experts = n /\ k % experts = s))
  = ()

(* The context of one routing step: premises of the routing relation hold. *)
let step_ctx (r:run) (n:nat{n < tokens}) : prop =
  S.length (scores r) == tokens * experts /\ inv r n /\ token_ok r n /\
  S.length (soft r n) == experts /\
  S.length (asel r n) == 1 /\ S.length (aprob r n) == 1 /\
  0 <= sel_at r n /\ sel_at r n < experts /\
  S.index (sel_seq r) n == sel_at r n /\
  (forall (e:nat). e < experts ==> count_upto (sel_seq r) e n <= n)

#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
let step_ctx_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires S.length (scores r) == tokens * experts /\ inv r n /\ token_ok r n)
          (ensures step_ctx r n)
  = let row = soft r n in
    assert (S.length row == experts);
    opt_scan_allowed row;
    let aux_le (e:nat{e < experts}) : Lemma (count_upto (sel_seq r) e n <= n) =
      count_upto_le (sel_seq r) e n in
    FStar.Classical.forall_intro aux_le
#pop-options

(* The routing relation's conclusions at step n, with named terms. *)
let rk_at (r:run) (n:nat) : int =
  let c = st_cnt r n in
  let s = sel_at r n in
  if 0 <= s && s < S.length c then I32.v (S.index c s) else 0

let slot_at (r:run) (n:nat) : int = capacity * sel_at r n + rk_at r n

let ro_base (r:run) (n:nat{n < tokens}) : prop =
  S.length (st_cnt r n) == experts /\ S.length (st_idx r n) == slots /\
  S.length (st_prb r n) == slots /\ S.length (st_srv r n) == tokens * experts /\
  S.length (soft r n) == experts /\ S.length (aprob r n) == 1 /\
  0 <= sel_at r n /\ sel_at r n < experts /\
  rk_at r n == count_upto (sel_seq r) (sel_at r n) n /\ 0 <= rk_at r n /\ rk_at r n <= n /\
  S.length (st_cnt r (n + 1)) == experts /\ S.length (st_idx r (n + 1)) == slots /\
  S.length (st_prb r (n + 1)) == slots /\ S.length (st_srv r (n + 1)) == tokens * experts

let ro_cnt (r:run) (n:nat{n < tokens}) : prop =
  S.length (st_cnt r n) == experts /\ S.length (st_cnt r (n + 1)) == experts /\
  (forall (e:nat). e < experts ==>
     I32.v (S.index (st_cnt r (n + 1)) e) ==
       I32.v (S.index (st_cnt r n) e) + (if e = sel_at r n then 1 else 0))

let ro_idx (r:run) (n:nat{n < tokens}) : prop =
  S.length (st_idx r n) == slots /\ S.length (st_idx r (n + 1)) == slots /\
  (forall (k:nat). k < slots ==>
     (if rk_at r n < capacity && k = slot_at r n
      then I64.v (S.index (st_idx r (n + 1)) k) == n
      else S.index (st_idx r (n + 1)) k == S.index (st_idx r n) k))

let ro_prb (r:run) (n:nat{n < tokens}) : prop =
  S.length (st_prb r n) == slots /\ S.length (st_prb r (n + 1)) == slots /\
  (forall (k:nat). k < slots ==>
     (if rk_at r n < capacity && k = slot_at r n then True
      else S.index (st_prb r (n + 1)) k == S.index (st_prb r n) k))

let ro_comb (r:run) (n:nat{n < tokens}) : prop =
  S.length (soft r n) == experts /\ S.length (aprob r n) == 1 /\
  0 <= sel_at r n /\ sel_at r n < experts /\
  S.length (st_prb r (n + 1)) == slots /\
  (rk_at r n < capacity ==>
     0 <= slot_at r n /\ slot_at r n < slots /\
     optimized_combine_ok (S.index (soft r n) (sel_at r n)) (S.index (aprob r n) 0)
       (S.index (st_prb r (n + 1)) (slot_at r n)))

let ro_srv (r:run) (n:nat{n < tokens}) : prop =
  S.length (st_srv r n) == tokens * experts /\ S.length (st_srv r (n + 1)) == tokens * experts /\
  (forall (k:nat). k < tokens * experts ==>
     (if rk_at r n < capacity && k = experts * n + sel_at r n
      then I32.v (S.index (st_srv r (n + 1)) k) == 1
      else S.index (st_srv r (n + 1)) k == S.index (st_srv r n) k))

let routing_out (r:run) (n:nat{n < tokens}) : prop =
  ro_base r n /\ ro_cnt r n /\ ro_idx r n /\ ro_prb r n /\ ro_comb r n /\ ro_srv r n

(* The premises of the routing relation hold in the step context. *)
#push-options "--z3rlimit 100 --fuel 1 --ifuel 1"
let routing_rel (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n)
          (ensures ro_base r n /\
                   (let row = soft r n in let sel = asel r n in let sp = aprob r n in
                    let counts = st_cnt r n in
                    S.length row == experts /\ S.length sel == 1 /\ S.length sp == 1 /\
                    S.length counts == experts /\ S.length (st_idx r n) == slots /\
                    S.length (st_prb r n) == slots /\
                    S.length (st_srv r n) == tokens * experts /\ 0 <= n /\ n < tokens /\
                    0 <= I32.v (S.index sel 0) /\ I32.v (S.index sel 0) < experts /\
                    I32.v (S.index sel 0) == sel_at r n /\
                    I32.v (S.index counts (sel_at r n)) == rk_at r n /\
                    (forall (e:nat). e < experts ==>
                       0 <= I32.v (S.index counts e) /\ I32.v (S.index counts e) <= n)))
  = let sel = sel_seq r in
    let c = st_cnt r n in
    rs_reveal r n; sv_reveal r n;
    assert (routing_state sel (comb_seq r) n c (st_idx r n) (st_prb r n));
    let aux (e:nat{e < experts}) : Lemma (0 <= I32.v (S.index c e) /\ I32.v (S.index c e) <= n /\
                                          I32.v (S.index c e) == count_upto sel e n) =
      count_v_i32 (S.index c e) in
    FStar.Classical.forall_intro aux;
    assert (optimized_routing_relation (soft r n) (asel r n) (aprob r n)
              c (st_idx r n) (st_prb r n) (st_srv r n) n
              (st_cnt r (n + 1)) (st_idx r (n + 1)) (st_prb r (n + 1)) (st_srv r (n + 1)))
#pop-options

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let ro_cnt_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n) (ensures ro_cnt r n)
  = routing_rel r n

let ro_idx_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n) (ensures ro_idx r n)
  = routing_rel r n

let ro_prb_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n) (ensures ro_prb r n)
  = routing_rel r n

let ro_comb_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n) (ensures ro_comb r n)
  = routing_rel r n;
    if rk_at r n < capacity then assert (slot_at r n < slots)

let ro_srv_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n) (ensures ro_srv r n)
  = routing_rel r n

let routing_out_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n) (ensures routing_out r n)
  = routing_rel r n; ro_cnt_intro r n; ro_idx_intro r n; ro_prb_intro r n;
    ro_comb_intro r n; ro_srv_intro r n
#pop-options

(* ------------------------------------------------------------------ *)
(* One routing step preserves the invariant *)

let step_hyp (r:run) (n:nat{n < tokens}) : prop = step_ctx r n /\ routing_out r n

#push-options "--z3rlimit 50 --fuel 1 --ifuel 1"
let step_cnt (r:run) (n:nat{n < tokens})
  : Lemma (requires step_hyp r n)
          (ensures forall (e:nat). e < experts ==>
                     count_v (S.index (st_cnt r (n + 1)) e) == count_upto (sel_seq r) e (n + 1))
  = rs_reveal r n;
    let aux (e:nat{e < experts})
      : Lemma (count_v (S.index (st_cnt r (n + 1)) e) == count_upto (sel_seq r) e (n + 1))
      = count_v_i32 (S.index (st_cnt r (n + 1)) e);
        count_v_i32 (S.index (st_cnt r n) e)
    in
    FStar.Classical.forall_intro aux

let step_idx (r:run) (n:nat{n < tokens})
  : Lemma (requires step_hyp r n)
          (ensures forall (k:nat). k < slots ==>
                     I64.v (S.index (st_idx r (n + 1)) k) ==
                       slot_token (sel_seq r) (k / capacity) (k % capacity) (n + 1))
  = rs_reveal r n;
    let aux (k:nat{k < slots})
      : Lemma (I64.v (S.index (st_idx r (n + 1)) k) ==
                 slot_token (sel_seq r) (k / capacity) (k % capacity) (n + 1))
      = slot_iff k (sel_at r n) (rk_at r n)
    in
    FStar.Classical.forall_intro aux

let comb_at_slot (r:run) (n:nat{n < tokens})
  : Lemma (requires step_hyp r n /\ rk_at r n < capacity)
          (ensures S.index (comb_seq r) n == S.index (st_prb r (n + 1)) (slot_at r n))
  = assert (ro_base r n /\ ro_comb r n);
    assert (S.index (comb_seq r) n == comb_at r n);
    assert (rk_at r n == I32.v (S.index (st_cnt r n) (sel_at r n)));
    assert (0 <= slot_at r n /\ slot_at r n < slots)

let step_prb (r:run) (n:nat{n < tokens})
  : Lemma (requires step_hyp r n)
          (ensures forall (k:nat). k < slots ==>
                     S.index (st_prb r (n + 1)) k ==
                       (let t = slot_token (sel_seq r) (k / capacity) (k % capacity) (n + 1) in
                        if t >= 0 then S.index (comb_seq r) t else zero))
  = rs_reveal r n;
    let aux (k:nat{k < slots})
      : Lemma (S.index (st_prb r (n + 1)) k ==
                 (let t = slot_token (sel_seq r) (k / capacity) (k % capacity) (n + 1) in
                  if t >= 0 then S.index (comb_seq r) t else zero))
      = slot_iff k (sel_at r n) (rk_at r n);
        if rk_at r n < capacity then comb_at_slot r n
    in
    FStar.Classical.forall_intro aux

let lt_succ_ne (a n:int) : Lemma (requires a <> n) (ensures (a < n + 1) == (a < n)) = ()

let srv_step_pure (sel:S.seq int{S.length sel == tokens}) (n:nat{n < tokens}) (s rk:int)
  (old nw:S.seq I32.t)
  : Lemma (requires S.length old == tokens * experts /\ S.length nw == tokens * experts /\
                    rank sel n == rk /\ S.index sel n == s /\ 0 <= s /\ s < experts /\
                    surv_inv sel n old /\
                    (forall (k:nat). k < tokens * experts ==>
                       (if rk < capacity && k = experts * n + s
                        then I32.v (S.index nw k) == 1
                        else S.index nw k == S.index old k)))
          (ensures surv_inv sel (n + 1) nw)
  = let aux (k:nat{k < tokens * experts})
      : Lemma (I32.v (S.index nw k) ==
                 (if k / experts < n + 1 && S.index sel (k / experts) = k % experts &&
                     kept sel (k / experts) then 1 else 0))
      = surv_iff k n s;
        assert (I32.v (S.index old k) ==
                 (if k / experts < n && S.index sel (k / experts) = k % experts &&
                     kept sel (k / experts) then 1 else 0));
        if k / experts = n then begin
          assert (kept sel n == (rk < capacity));
          assert ((k = experts * n + s) == (k % experts = s))
        end else begin
          assert (k <> experts * n + s);
          assert (S.index nw k == S.index old k);
          lt_succ_ne (k / experts) n
        end
    in
    FStar.Classical.forall_intro aux

let step_srv (r:run) (n:nat{n < tokens})
  : Lemma (requires step_hyp r n)
          (ensures surv_inv (sel_seq r) (n + 1) (st_srv r (n + 1)))
  = sv_reveal r n; sv_reveal r (n + 1);
    assert (ro_base r n /\ ro_srv r n);
    assert (rank (sel_seq r) n == rk_at r n);
    srv_step_pure (sel_seq r) n (sel_at r n) (rk_at r n) (st_srv r n) (st_srv r (n + 1))
#pop-options

let combine_from_opt (row:S.seq F32.t) (s:int) (sp c:F32.t)
  : Lemma (requires S.length row == experts /\ 0 <= s /\ s < experts /\
                    sp == S.index row s /\
                    optimized_combine_ok (S.index row s) sp c)
          (ensures selected_combine_ok row s c)
  = let aux (rp:real)
      : Lemma (requires S.index row s %~ rp /\ rp =!= 0.0R)
              (ensures rp =!= 0.0R ==> c %~ ref_combine rp)
      = assert (0.0R +. rp == rp)
    in
    FStar.Classical.forall_intro (FStar.Classical.move_requires aux)

#push-options "--z3rlimit 50 --fuel 1 --ifuel 1"
let step_post (r:run) (n:nat{n < tokens})
  : Lemma (requires step_hyp r n) (ensures token_post r n)
  = let row = soft r n in
    assert (ro_base r n /\ ro_comb r n);
    rows_row r n;
    opt_scan_allowed row;
    assert (optimized_argmax_relation row (asel r n) (aprob r n));
    assert (sel_at r n == fst (base_scan row experts));
    assert (S.index (aprob r n) 0 == snd (base_scan row experts));
    if kept (sel_seq r) n then begin
      assert (rank (sel_seq r) n == rk_at r n);
      comb_at_slot r n;
      combine_from_opt row (sel_at r n) (S.index (aprob r n) 0) (S.index (comb_seq r) n)
    end
#pop-options

let routing_state_intro (sel:S.seq int) (comb:S.seq F32.t)
  (n:nat{n <= S.length sel /\ n <= S.length comb})
  (counts:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  : Lemma (requires S.length counts == experts /\ S.length idx == slots /\ S.length probs == slots /\
                    (forall (e:nat). e < experts ==> count_v (S.index counts e) == count_upto sel e n) /\
                    (forall (k:nat). k < slots ==>
                       I64.v (S.index idx k) == slot_token sel (k / capacity) (k % capacity) n) /\
                    (forall (k:nat). k < slots ==>
                       S.index probs k ==
                         (let t = slot_token sel (k / capacity) (k % capacity) n in
                          if t >= 0 then S.index comb t else zero)))
          (ensures routing_state sel comb n counts idx probs)
  = ()

#push-options "--fuel 0 --ifuel 0"
let inv_intro (r:run) (n:nat{n < tokens})
  : Lemma (requires rs_ok r (n + 1) /\ sv_ok r (n + 1) /\
                    (forall (t:nat). t < n ==> tp r t) /\ tp r n)
          (ensures inv r (n + 1))
  = let aux (t:nat) : Lemma (t < n + 1 ==> tp r t) =
      if t < n then () else () in
    FStar.Classical.forall_intro aux
#pop-options

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let step (r:run) (n:nat{n < tokens})
  : Lemma (requires step_ctx r n) (ensures inv r (n + 1))
  = routing_out_intro r n;
    assert (ro_base r n);
    step_cnt r n; step_idx r n; step_prb r n; step_srv r n; step_post r n;
    routing_state_intro (sel_seq r) (comb_seq r) (n + 1)
      (st_cnt r (n + 1)) (st_idx r (n + 1)) (st_prb r (n + 1));
    rs_reveal r (n + 1); sv_reveal r (n + 1); tp_reveal r n;
    inv_intro r n
#pop-options

(* ------------------------------------------------------------------ *)
(* G4: the one chain lemma over the token sequence, size-generic in n. *)

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let proj_facts (r:run)
  : Lemma (requires input_lengths r /\ family_ok r)
          (ensures S.length (scores r) == tokens * experts /\
                   scores_ok r.input0 r.input1 (scores r) /\
                   initial_routing_state (st_cnt r 0) (st_idx r 0) (st_prb r 0) /\
                   S.length (st_srv r 0) == tokens * experts /\
                   (forall (k:nat). k < tokens * experts ==> I32.v (S.index (st_srv r 0) k) == 0))
  = assert (optimized_projection_relation r.input0 r.input1 r.sample_0000_projection_0
              r.sample_0000_projection_1 r.sample_0000_projection_2 r.sample_0000_projection_3
              r.sample_0000_projection_4)

let rec chain (r:run) (n:nat{n <= tokens})
  : Lemma (requires input_lengths r /\ family_ok r) (ensures inv r n) (decreases n)
  = proj_facts r;
    if n = 0 then inv_zero r
    else begin
      chain r (n - 1);
      token_ok_of_family r (n - 1);
      step_ctx_intro r (n - 1);
      step r (n - 1)
    end
#pop-options

(* ------------------------------------------------------------------ *)
(* The composition theorem *)

let surv_final (sel:S.seq int{S.length sel == tokens}) (surv:S.seq I32.t)
  : Lemma (requires surv_inv sel tokens surv) (ensures survival_rows sel tokens surv)
  = let aux (t:nat) (e:nat)
      : Lemma (t < tokens /\ e < experts ==>
               I32.v (S.index surv (experts * t + e)) ==
                 (if S.index sel t = e && kept sel t then 1 else 0))
      = if t < tokens && e < experts then
          surv_iff (experts * t + e) t e
    in
    FStar.Classical.forall_intro_2 aux

let post_intro (x w:S.seq BF.t) (demand:S.seq I32.t) (idx:S.seq I64.t) (probs:S.seq F32.t)
  (surv:S.seq I32.t) (rows:S.seq F32.t) (sel:S.seq int) (comb:S.seq F32.t)
  : Lemma (requires
      S.length demand == experts /\ S.length idx == slots /\
      S.length probs == slots /\ S.length surv == tokens * experts /\
      S.length rows == tokens * experts /\
      S.length sel == tokens /\ S.length comb == tokens /\
      softmax_rows_ok x w rows /\
      (forall (t:nat). t < tokens ==> allowed_choice (row_of rows t) (S.index sel t)) /\
      (forall (t:nat). t < tokens /\ kept sel t ==>
         selected_combine_ok (row_of rows t) (S.index sel t) (S.index comb t)) /\
      routing_state sel comb tokens demand idx probs /\
      survival_rows sel tokens surv)
    (ensures program_postcondition x w demand idx probs surv)
  = ()

#push-options "--z3rlimit 50 --fuel 0 --ifuel 0"
let token_final (r:run) (t:nat{t < tokens})
  : Lemma (requires input_lengths r /\ family_ok r /\ tp r t)
          (ensures softmax_row_ok r.input0 r.input1 (rows_seq r) t /\
                   allowed_choice (row_of (rows_seq r) t) (S.index (sel_seq r) t) /\
                   (kept (sel_seq r) t ==>
                      selected_combine_ok (row_of (rows_seq r) t) (S.index (sel_seq r) t)
                        (S.index (comb_seq r) t)))
  = tp_reveal r t;
    proj_facts r;
    token_ok_of_family r t;
    optimized_softmax_chain r.input0 r.input1 (scores r) (rows_seq r) (soft r t) t

let refines (r:run)
  : Lemma (requires input_lengths r /\ family_ok r)
          (ensures program_postcondition r.input0 r.input1 (output0 r) (output1 r) (output2 r) (output3 r))
  = chain r tokens;
    rs_reveal r tokens; sv_reveal r tokens;
    let aux (t:nat{t < tokens})
      : Lemma (softmax_row_ok r.input0 r.input1 (rows_seq r) t /\
               allowed_choice (row_of (rows_seq r) t) (S.index (sel_seq r) t) /\
               (kept (sel_seq r) t ==>
                  selected_combine_ok (row_of (rows_seq r) t) (S.index (sel_seq r) t)
                    (S.index (comb_seq r) t)))
      = token_final r t
    in
    FStar.Classical.forall_intro aux;
    surv_final (sel_seq r) (st_srv r tokens);
    assert (st_cnt r tokens == output0 r);
    assert (st_idx r tokens == output1 r);
    assert (st_prb r tokens == output2 r);
    assert (st_srv r tokens == output3 r);
    post_intro r.input0 r.input1 (output0 r) (output1 r) (output2 r) (output3 r)
      (rows_seq r) (sel_seq r) (comb_seq r)
#pop-options
