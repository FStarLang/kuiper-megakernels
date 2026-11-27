module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT
module I64 = FStar.Int64
module U32 = FStar.UInt32
module BF = Kuiper.BFloat16
module Cast = FStar.Int.Cast

#push-options "--z3rlimit 50"

(* Domain of the reference call (premises of the relation, minus shapes). *)
noextract
let in_domain (pP t : int) : bool = 1 <= pP && pP <= 12 && 0 <= t && t < 12

(* Destination row t - P + 1; outside the domain no row is a destination. *)
inline_for_extraction noextract
let dest_of (pP pos : i64)
  : r:i64 { I64.v r == (if in_domain (I64.v pP) (I64.v pos) then I64.v pos - I64.v pP + 1 else -1) }
  = if I64.(pP >=^ 1L && pP <=^ 12L && pos >=^ 0L && pos <^ 12L)
    then I64.(pos -^ pP +^ 1L)
    else (-1L)

inline_for_extraction noextract
let to_i64 (x : sz) : r:i64 { I64.v r == SZ.v x } =
  Cast.uint32_to_int64 (SZ.sizet_to_u32 x)

(* Full-array ownership of a global array may cross to the GPU. *)
instance is_send_global_pts_to (#a:Type0) (x:array a { is_global x }) (#f:perm) (v:S.seq a)
  : is_send_across gpu_of (pts_to x #f v)
  = let i : is_send_across (visibility_of x) (pts_to x #f v) = solve in i

inline_for_extraction noextract
let pick (#a:Type0) (c:bool) (x y:a) : r:a { r == (if c then x else y) } = if c then x else y

let tokens_ok (s0 s3 : S.seq i64) (d : int) (k : nat) (o : S.seq i64) : prop =
  S.length o == S.length s0 /\ k <= S.length s0 /\ S.length s3 == 1 /\
  (forall (i:nat). i < k ==> S.index o i == (if i = d then S.index s3 0 else S.index s0 i))

let history_ok (s1 s2 : S.seq bf16) (d : int) (k : nat) (o : S.seq bf16) : prop =
  S.length o == S.length s1 /\ k <= S.length s1 /\ S.length s2 == 1024 /\
  (forall (i:nat). i < k ==> S.index o i == (if i / 1024 = d then S.index s2 (i % 1024) else S.index s1 i))

inline_for_extraction noextract
fn kf
  (n : sz { SZ.fits (SZ.v n * 1024) })
  (old_tokens : larray i64 (SZ.v n) { is_global old_tokens })
  (old_history : larray bf16 (SZ.v n * 1024) { is_global old_history })
  (hidden : larray bf16 1024 { is_global hidden })
  (selected : larray i64 1 { is_global selected })
  (pP : i64 { is_global selected })
  (t : i64)
  (tokens : larray i64 (SZ.v n) { is_global tokens })
  (history : larray bf16 (SZ.v n * 1024) { is_global history })
  (#p0 #p1 #p2 #p3 : perm)
  (#s0 : erased (S.seq i64))
  (#s1 #s2 : erased (S.seq bf16))
  (#s3 : erased (S.seq i64))
  (#o0 : erased (S.seq i64))
  (#o1 : erased (S.seq bf16))
  preserves
    gpu **
    (old_tokens |-> Frac p0 s0) **
    (old_history |-> Frac p1 s1) **
    (hidden |-> Frac p2 s2) **
    (selected |-> Frac p3 s3)
  requires
    (tokens |-> o0) **
    (history |-> o1)
  ensures
    exists* (tokens' : S.seq i64) (history' : S.seq bf16).
      (tokens |-> tokens') **
      (history |-> history') **
      pure (base_token_history_update_relation s0 s1 s2 s3 (I64.v pP) (I64.v t) tokens' history')
{
  pts_to_len old_tokens;
  pts_to_len old_history;
  pts_to_len hidden;
  pts_to_len selected;
  pts_to_len tokens;
  pts_to_len history;
  let d = dest_of pP t;
  let best = selected.(0sz);
  let mut i : sz = 0sz;
  while (!i <^ n)
    invariant exists* (vi : sz) (o : S.seq i64).
      (i |-> vi) ** (tokens |-> o) ** pure (SZ.v vi <= SZ.v n /\ tokens_ok s0 s3 (I64.v d) (SZ.v vi) o)
    decreases (SZ.v n - SZ.v !i)
  {
    let vi = !i;
    let ot = old_tokens.(vi);
    tokens.(vi) <- pick (I64.eq (to_i64 vi) d) best ot;
    i := vi +^ 1sz;
  };
  let nh = n *^ row;
  let mut k : sz = 0sz;
  while (!k <^ nh)
    invariant exists* (vk : sz) (o : S.seq bf16).
      (k |-> vk) ** (history |-> o) ** pure (SZ.v vk <= SZ.v nh /\ history_ok s1 s2 (I64.v d) (SZ.v vk) o)
    decreases (SZ.v nh - SZ.v !k)
  {
    let vk = !k;
    let hv = hidden.(vk %^ row);
    let ov = old_history.(vk);
    history.(vk) <- pick (I64.eq (to_i64 (vk /^ row)) d) hv ov;
    k := vk +^ 1sz;
  };
  ()
}

fn kplay_run
  (n : sz { SZ.fits (SZ.v n * 1024) })
  (old_tokens : larray i64 (SZ.v n) { is_global old_tokens })
  (old_history : larray bf16 (SZ.v n * 1024) { is_global old_history })
  (hidden : larray bf16 1024 { is_global hidden })
  (selected : larray i64 1 { is_global selected })
  (pP : i64 { is_global selected })
  (t : i64)
  (tokens : larray i64 (SZ.v n) { is_global tokens })
  (history : larray bf16 (SZ.v n * 1024) { is_global history })
  (#p0 #p1 #p2 #p3 : perm)
  (#s0 : erased (S.seq i64))
  (#s1 #s2 : erased (S.seq bf16))
  (#s3 : erased (S.seq i64))
  (#o0 : erased (S.seq i64))
  (#o1 : erased (S.seq bf16))
  preserves
    cpu **
    on gpu_loc (old_tokens |-> Frac p0 s0) **
    on gpu_loc (old_history |-> Frac p1 s1) **
    on gpu_loc (hidden |-> Frac p2 s2) **
    on gpu_loc (selected |-> Frac p3 s3)
  requires
    on gpu_loc (tokens |-> o0) **
    on gpu_loc (history |-> o1)
  ensures
    exists* (tokens' : S.seq i64) (history' : S.seq bf16).
      on gpu_loc (tokens |-> tokens') **
      on gpu_loc (history |-> history') **
      pure (base_token_history_update_relation s0 s1 s2 s3 (I64.v pP) (I64.v t) tokens' history')
{
  launch_kernel_1
    (fun () ->
      kf n old_tokens old_history hidden selected pP t tokens history
        #p0 #p1 #p2 #p3 #s0 #s1 #s2 #s3 #o0 #o1);
}

#pop-options
