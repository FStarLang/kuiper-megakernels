module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = FStar.SizeT
module I64 = FStar.Int64
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast
module F = Kuiper.Floating
module ML = FStar.Math.Lemmas

#set-options "--z3rlimit 40"

(*** Ghost bookkeeping *)

let row_of (q j:nat)
  : Lemma (requires j < 1024)
          (ensures (q * 1024 + j) / 1024 == q /\ (q * 1024 + j) % 1024 == j)
  = ML.lemma_div_plus j q 1024;
    ML.small_div j 1024;
    ML.lemma_mod_plus j q 1024;
    ML.small_mod j 1024

(* Prefix [0,n) of the embedding buffer holds the rows of valid prompt tokens. *)
let emb_ok (ts:S.seq bf16) (ps:S.seq i64) (s:S.seq bf16) (n:nat) : prop =
  S.length ts == table_len /\ n <= S.length s /\
  (forall (k:nat). k < n /\ valid_token (tok_at ps (k / 1024)) ==>
     S.index s k == S.index ts (tok_at ps (k / 1024) * 1024 + k % 1024))

let emb_ok_step (ts:S.seq bf16) (ps:S.seq i64) (s:S.seq bf16)
                (q i:nat) (tk:int)
  : Lemma (requires emb_ok ts ps s (q * 1024 + i) /\ i < 1024 /\ q * 1024 + i < S.length s /\
                    tok_at ps q == tk /\ valid_token tk)
          (ensures emb_ok ts ps (S.upd s (q * 1024 + i) (S.index ts (tk * 1024 + i))) (q * 1024 + i + 1))
  = row_of q i

let emb_ok_skip (ts:S.seq bf16) (ps:S.seq i64) (s:S.seq bf16) (q:nat)
  : Lemma (requires emb_ok ts ps s (q * 1024) /\ (q + 1) * 1024 <= S.length s /\
                    ~(valid_token (tok_at ps q)))
          (ensures emb_ok ts ps s ((q + 1) * 1024))
  = let aux (k:nat) : Lemma (k < (q + 1) * 1024 /\ valid_token (tok_at ps (k / 1024)) ==>
                               S.index s k == S.index ts (tok_at ps (k / 1024) * 1024 + k % 1024)) =
      if k >= q * 1024 && k < (q + 1) * 1024 then row_of q (k - q * 1024)
    in
    FStar.Classical.forall_intro aux

let emb_final (ts:S.seq bf16) (ps:S.seq i64) (s:S.seq bf16) (pP:nat)
  : Lemma (requires emb_ok ts ps s (pP * 1024) /\ S.length s == pP * 1024 /\
                    S.length ps == pP /\ valid_tokens ps)
          (ensures forall (i:nat). i < pP * 1024 ==>
                     S.index s i == S.index ts (tok_at ps (i / 1024) * 1024 + i % 1024))
  = let aux (i:nat) : Lemma (i < pP * 1024 ==>
                               S.index s i == S.index ts (tok_at ps (i / 1024) * 1024 + i % 1024)) =
      if i < pP * 1024 then begin
        ML.euclidean_division_definition i 1024;
        assert (i / 1024 < pP)
      end
    in
    FStar.Classical.forall_intro aux

let zeros_upto (#t:Type) {| F.floating t |} (s:S.seq t) (n:nat) : prop =
  n <= S.length s /\ (forall (k:nat). k < n ==> S.index s k == zero)

let toks_upto (s:S.seq i64) (n:nat) : prop =
  n <= S.length s /\ (forall (k:nat). k < n ==> I64.v (S.index s k) == 0)

(*** Device code *)

inline_for_extraction noextract
fn zero_bf16
  (a : array bf16)
  (n : sz)
  (#s0 : erased (S.seq bf16))
  preserves gpu
  requires a |-> s0 ** pure (S.length s0 == SZ.v n)
  ensures exists* (s : S.seq bf16). a |-> s ** pure (S.length s == SZ.v n /\ zeros_upto s (SZ.v n))
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (vi : sz) (s : S.seq bf16).
      i |-> vi ** a |-> s **
      pure (SZ.v vi <= SZ.v n /\ S.length s == SZ.v n /\ zeros_upto s (SZ.v vi))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    pts_to_len a;
    a.(vi) <- zero;
    i := vi +^ 1sz;
  };
}

inline_for_extraction noextract
fn zero_f32
  (a : array f32)
  (n : sz)
  (#s0 : erased (S.seq f32))
  preserves gpu
  requires a |-> s0 ** pure (S.length s0 == SZ.v n)
  ensures exists* (s : S.seq f32). a |-> s ** pure (S.length s == SZ.v n /\ zeros_upto s (SZ.v n))
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (vi : sz) (s : S.seq f32).
      i |-> vi ** a |-> s **
      pure (SZ.v vi <= SZ.v n /\ S.length s == SZ.v n /\ zeros_upto s (SZ.v vi))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    pts_to_len a;
    a.(vi) <- zero;
    i := vi +^ 1sz;
  };
}

inline_for_extraction noextract
fn zero_i64
  (a : array i64)
  (n : sz)
  (#s0 : erased (S.seq i64))
  preserves gpu
  requires a |-> s0 ** pure (S.length s0 == SZ.v n)
  ensures exists* (s : S.seq i64). a |-> s ** pure (S.length s == SZ.v n /\ toks_upto s (SZ.v n))
{
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (vi : sz) (s : S.seq i64).
      i |-> vi ** a |-> s **
      pure (SZ.v vi <= SZ.v n /\ S.length s == SZ.v n /\ toks_upto s (SZ.v vi))
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    pts_to_len a;
    a.(vi) <- 0L;
    i := vi +^ 1sz;
  };
}

let cast_small (x:i64) (bound:nat)
  : Lemma (requires 0 <= I64.v x /\ I64.v x < bound /\ bound <= 4294967296)
          (ensures U32.v (Cast.int64_to_uint32 x) == I64.v x)
  = FStar.UInt.pow2_values 32;
    ML.small_mod (I64.v x) (pow2 32)

inline_for_extraction noextract
fn copy_row
  (table : array bf16)
  (emb : array bf16)
  (p : sz)
  (q : sz)
  (tk : sz)
  (#ft : perm)
  (#ts : erased (S.seq bf16))
  (#ps : erased (S.seq i64))
  (vtk : erased int)
  (#e0 : erased (S.seq bf16))
  preserves gpu ** table |-> Frac ft ts
  requires emb |-> e0 **
    pure (S.length ts == table_len /\ SZ.v q < SZ.v p /\ SZ.v p <= 12 /\
          S.length e0 == SZ.v p * 1024 /\ emb_ok ts ps e0 (SZ.v q * 1024) /\
          SZ.v tk == reveal vtk /\ tok_at ps (SZ.v q) == reveal vtk /\ valid_token vtk)
  ensures exists* (s : S.seq bf16). emb |-> s **
    pure (S.length s == SZ.v p * 1024 /\ emb_ok ts ps s ((SZ.v q + 1) * 1024))
{
  let base = tk *^ 1024sz;
  let row = q *^ 1024sz;
  let mut i = 0sz;
  while (!i <^ 1024sz)
    invariant exists* (vi : sz) (s : S.seq bf16).
      i |-> vi ** emb |-> s **
      pure (SZ.v vi <= 1024 /\ S.length s == SZ.v p * 1024 /\
            emb_ok ts ps s (SZ.v q * 1024 + SZ.v vi))
    decreases (1024 - SZ.v !i)
  {
    let vi = !i;
    pts_to_len table;
    pts_to_len emb;
    let x = table.(base +^ vi);
    with s. assert (emb |-> s);
    emb_ok_step ts ps s (SZ.v q) (SZ.v vi) vtk;
    emb.(row +^ vi) <- x;
    i := vi +^ 1sz;
  };
}

inline_for_extraction noextract
fn row_step
  (table : array bf16)
  (emb : array bf16)
  (p : sz)
  (q : sz)
  (tok : i64)
  (#ft : perm)
  (#ts : erased (S.seq bf16))
  (#ps : erased (S.seq i64))
  (#e0 : erased (S.seq bf16))
  preserves gpu ** table |-> Frac ft ts
  requires emb |-> e0 **
    pure (S.length ts == table_len /\ SZ.v q < SZ.v p /\ SZ.v p <= 12 /\ SZ.v p == S.length ps /\
          S.length e0 == SZ.v p * 1024 /\ emb_ok ts ps e0 (SZ.v q * 1024) /\
          tok == S.index ps (SZ.v q))
  ensures exists* (s : S.seq bf16). emb |-> s **
    pure (S.length s == SZ.v p * 1024 /\ emb_ok ts ps s ((SZ.v q + 1) * 1024))
{
  if (I64.gte tok 0L && I64.lt tok 248320L) {
    cast_small tok 248320;
    copy_row table emb p q (SZ.uint32_to_sizet (Cast.int64_to_uint32 tok)) #ft #ts #ps (I64.v tok);
  } else {
    emb_ok_skip ts ps e0 (SZ.v q);
    ()
  }
}

inline_for_extraction noextract
fn copy_emb
  (table : array bf16)
  (prompt : array i64)
  (p : sz)
  (emb : array bf16)
  (#ft #fp : perm)
  (#ts : erased (S.seq bf16))
  (#ps : erased (S.seq i64))
  (#e0 : erased (S.seq bf16))
  preserves gpu ** table |-> Frac ft ts ** prompt |-> Frac fp ps
  requires emb |-> e0 **
    pure (S.length ts == table_len /\ S.length ps == SZ.v p /\ SZ.v p <= 12 /\
          S.length e0 == SZ.v p * 1024)
  ensures exists* (s : S.seq bf16). emb |-> s **
    pure (S.length s == SZ.v p * 1024 /\ emb_ok ts ps s (SZ.v p * 1024))
{
  let mut q = 0sz;
  while (!q <^ p)
    invariant exists* (vq : sz) (s : S.seq bf16).
      q |-> vq ** emb |-> s **
      pure (SZ.v vq <= SZ.v p /\ S.length s == SZ.v p * 1024 /\ emb_ok ts ps s (SZ.v vq * 1024))
    decreases (SZ.v p - SZ.v !q)
  {
    let vq = !q;
    pts_to_len prompt;
    let tok = prompt.(vq);
    row_step table emb p vq tok #ft #ts #ps;
    q := vq +^ 1sz;
  };
}

let rel_intro (ts:S.seq bf16) (ps:S.seq i64) (steps:int)
              (e c:S.seq bf16) (r:S.seq f32) (t:S.seq i64) (h:S.seq bf16)
  : Lemma (requires S.length ts == table_len /\ 1 <= steps /\
                    S.length e == S.length ps * 1024 /\ emb_ok ts ps e (S.length ps * 1024) /\
                    S.length c == 442368 + (S.length ps + steps - 1) * 6144 /\ zeros_upto c (S.length c) /\
                    S.length r == opt_rec_len /\ zeros_upto r (S.length r) /\
                    S.length t == steps /\ toks_upto t steps /\
                    S.length h == steps * 1024 /\ zeros_upto h (S.length h))
          (ensures base_initialize_relation ts ps steps e c r t h)
  = introduce valid_tokens ps ==>
      (forall (i:nat). i < S.length ps * 1024 ==>
         S.index e i == S.index ts (tok_at ps (i / 1024) * 1024 + i % 1024))
    with emb_final ts ps e (S.length ps)

let rel_vacuous (ts:S.seq bf16) (ps:S.seq i64) (steps:int)
                (e c:S.seq bf16) (r:S.seq f32) (t:S.seq i64) (h:S.seq bf16)
  : Lemma (requires ~(1 <= S.length ps /\ 1 <= steps /\ S.length ps + steps - 1 <= 12))
          (ensures base_initialize_relation ts ps steps e c r t h)
  = ()

inline_for_extraction noextract
fn kernel
  (table : array bf16)
  (prompt : array i64)
  (p : sz)
  (steps : i64)
  (emb : array bf16)
  (cache : array bf16)
  (recurrent : array f32)
  (tokens : array i64)
  (history : array bf16)
  (#ft #fp : perm)
  (#ts : erased (S.seq bf16))
  (#ps : erased (S.seq i64))
  (#e0 #c0 #h0 : erased (S.seq bf16))
  (#r0 : erased (S.seq f32))
  (#t0 : erased (S.seq i64))
  (_ : squash (S.length ts == table_len /\
          S.length ps == SZ.v p /\
          S.length e0 == SZ.v p * 1024 /\
          S.length c0 == cache_extent (SZ.v p) (I64.v steps) /\
          S.length r0 == opt_rec_len /\
          S.length t0 == snat (I64.v steps) /\
          S.length h0 == snat (I64.v steps) * 1024))
  preserves gpu ** table |-> Frac ft ts ** prompt |-> Frac fp ps
  requires
    emb |-> e0 ** cache |-> c0 ** recurrent |-> r0 ** tokens |-> t0 ** history |-> h0
  ensures
    exists* (e c h : S.seq bf16) (r : S.seq f32) (t : S.seq i64).
      emb |-> e ** cache |-> c ** recurrent |-> r ** tokens |-> t ** history |-> h **
      pure (base_initialize_relation ts ps (I64.v steps) e c r t h)
{
  if (p >=^ 1sz && p <=^ 12sz && I64.gte steps 1L && I64.lte steps 12L) {
    cast_small steps 13;
    let s = SZ.uint32_to_sizet (Cast.int64_to_uint32 steps);
    if (p +^ s <=^ 13sz) {
      copy_emb table prompt p emb;
      zero_bf16 cache (SZ.uint32_to_sizet 442368ul +^ (p +^ s -^ 1sz) *^ 6144sz);
      zero_f32 recurrent (SZ.uint32_to_sizet 4718592ul);
      zero_i64 tokens s;
      zero_bf16 history (s *^ 1024sz);
      with e. assert (emb |-> e);
      with c. assert (cache |-> c);
      with r. assert (recurrent |-> r);
      with t. assert (tokens |-> t);
      with h. assert (history |-> h);
      rel_intro ts ps (I64.v steps) e c r t h;
    } else {
      rel_vacuous ts ps (I64.v steps) e0 c0 r0 t0 h0;
    }
  } else {
    rel_vacuous ts ps (I64.v steps) e0 c0 r0 t0 h0;
  }
}

instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (table : array bf16 { is_global table })
  (prompt : array i64 { is_global prompt })
  (p : sz)
  (steps : i64)
  (emb : array bf16 { is_global emb })
  (cache : array bf16 { is_global cache })
  (recurrent : array f32 { is_global recurrent })
  (tokens : array i64 { is_global tokens })
  (history : array bf16 { is_global history })
  (#ft #fp : perm)
  (#ts : erased (S.seq bf16))
  (#ps : erased (S.seq i64))
  (#e0 #c0 #h0 : erased (S.seq bf16))
  (#r0 : erased (S.seq f32))
  (#t0 : erased (S.seq i64))
  preserves
    cpu **
    on gpu_loc (table |-> Frac ft ts) **
    on gpu_loc (prompt |-> Frac fp ps)
  requires
    on gpu_loc (emb |-> e0) **
    on gpu_loc (cache |-> c0) **
    on gpu_loc (recurrent |-> r0) **
    on gpu_loc (tokens |-> t0) **
    on gpu_loc (history |-> h0) **
    pure (S.length ts == table_len /\
          S.length ps == SZ.v p /\
          S.length e0 == SZ.v p * 1024 /\
          S.length c0 == cache_extent (SZ.v p) (I64.v steps) /\
          S.length r0 == opt_rec_len /\
          S.length t0 == snat (I64.v steps) /\
          S.length h0 == snat (I64.v steps) * 1024)
  ensures
    exists* (e c h : S.seq bf16) (r : S.seq f32) (t : S.seq i64).
      on gpu_loc (emb |-> e) **
      on gpu_loc (cache |-> c) **
      on gpu_loc (recurrent |-> r) **
      on gpu_loc (tokens |-> t) **
      on gpu_loc (history |-> h) **
      pure (base_initialize_relation ts ps (I64.v steps) e c r t h)
{
  launch_kernel_1
    (fun () ->
      kernel table prompt p steps emb cache recurrent tokens history
        #ft #fp #ts #ps #e0 #c0 #h0 #r0 #t0 ());
}
