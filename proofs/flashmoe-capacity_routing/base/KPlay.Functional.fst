module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module A = Pulse.Lib.Array
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module I32 = FStar.Int32
module I64 = FStar.Int64
module U32 = FStar.UInt32
module Cast = FStar.Int.Cast

(* ------------------------------------------------------------------ *)
(* Pure facts                                                           *)

(* __fdiv_rn(p, p) realizes the reference p / p for every nonzero real
   that p approximates. *)
let combine_lemma (p : F32.t) : Lemma (base_combine_ok p (div p p)) =
  approx_div #F32.t

let small_i64_to_sz (x : I64.t)
  : Lemma (requires 0 <= I64.v x /\ I64.v x < 4096)
          (ensures U32.v (Cast.int64_to_uint32 x) == I64.v x)
  = FStar.UInt.pow2_values 32

let small_i32_to_sz (x : I32.t)
  : Lemma (requires 0 <= I32.v x /\ I32.v x < 4096)
          (ensures U32.v (Cast.int32_to_uint32 x) == I32.v x)
  = FStar.UInt.pow2_values 32

inline_for_extraction noextract
let max_i64 : I64.t = 9223372036854775807L

(* Post-state of one update, as closed forms over the inputs. *)
let step_post
  (sel : int) (rank : I64.t) (c : F32.t) (t : int)
  (s2 s3 : S.seq I64.t) (s4 : S.seq F32.t) (s5 : S.seq I32.t)
  (o0 o1 : S.seq I64.t) (o2 : S.seq F32.t) (o3 : S.seq I32.t) : prop =
  0 <= t /\ S.length s2 == 8 /\ S.length s3 == 256 /\ S.length s4 == 256 /\
  S.length s5 == 8 * t /\
  S.length o0 == 8 /\ S.length o1 == 256 /\ S.length o2 == 256 /\
  S.length o3 == 8 * t + 8 /\
  (forall (e:nat). e < 8 ==>
     S.index o0 e == (if e = sel then (if I64.v rank < I64.v max_i64 then I64.add rank 1L else rank)
                      else S.index s2 e)) /\
  (forall (k:nat). k < 256 ==>
     (if 0 <= I64.v rank && I64.v rank < 32 && k = 32 * sel + I64.v rank
      then I64.v (S.index o1 k) == t /\ S.index o2 k == c
      else S.index o1 k == S.index s3 k /\ S.index o2 k == S.index s4 k)) /\
  (forall (k:nat). k < 8 * t ==> S.index o3 k == S.index s5 k) /\
  (forall (e:nat). e < 8 ==>
     I32.v (S.index o3 (8 * t + e)) == (if e = sel && I64.v rank < 32 then 1 else 0))

#push-options "--z3rlimit 50"
let step_relation
  (s0 : S.seq I32.t) (s1 : S.seq F32.t) (s2 s3 : S.seq I64.t) (s4 : S.seq F32.t)
  (s5 : S.seq I32.t) (t : int)
  (o0 o1 : S.seq I64.t) (o2 : S.seq F32.t) (o3 : S.seq I32.t)
  : Lemma
      (requires
        S.length s0 == 1 /\ S.length s1 == 1 /\ S.length s2 == 8 /\
        0 <= I32.v (S.index s0 0) /\ I32.v (S.index s0 0) < 8 /\
        step_post (I32.v (S.index s0 0)) (S.index s2 (I32.v (S.index s0 0)))
          (div (S.index s1 0) (S.index s1 0)) t s2 s3 s4 s5 o0 o1 o2 o3)
      (ensures base_capacity_routing_relation s0 s1 s2 s3 s4 s5 t o0 o1 o2 o3)
  = combine_lemma (S.index s1 0)
#pop-options

 (* Outside the relation's domain any final state satisfies it. *)
let out_of_domain
  (s0 : S.seq I32.t) (s1 : S.seq F32.t) (s2 s3 : S.seq I64.t) (s4 : S.seq F32.t)
  (s5 : S.seq I32.t) (t : int)
  (o0 o1 : S.seq I64.t) (o2 : S.seq F32.t) (o3 : S.seq I32.t)
  : Lemma
      (requires
        S.length s0 == 1 /\
        ~(0 <= t /\ t < 256 /\ 0 <= I32.v (S.index s0 0) /\ I32.v (S.index s0 0) < 8))
      (ensures base_capacity_routing_relation s0 s1 s2 s3 s4 s5 t o0 o1 o2 o3)
  = ()

(* ------------------------------------------------------------------ *)
(* Device code                                                          *)

inline_for_extraction noextract
fn copy_prefix
  (#a : Type0)
  (src : A.array a) (dst : A.array a) (n : SZ.t)
  (#p : perm) (#s : erased (S.seq a)) (#d : erased (S.seq a))
  preserves gpu ** src |-> Frac p s
  requires dst |-> d ** pure (SZ.v n <= S.length s /\ SZ.v n <= S.length d)
  ensures exists* (v : S.seq a).
    dst |-> v **
    pure (S.length v == S.length d /\ SZ.v n <= S.length v /\ SZ.v n <= S.length s /\
          (forall (k:nat). k < SZ.v n ==> S.index v k == S.index s k))
{
  A.pts_to_len src;
  A.pts_to_len dst;
  let mut i = 0sz;
  while (!i <^ n)
    invariant exists* (j : SZ.t) (v : S.seq a).
      i |-> j ** dst |-> v **
      pure (SZ.v j <= SZ.v n /\ S.length v == S.length d /\
            SZ.v n <= S.length v /\ SZ.v n <= S.length s /\
            (forall (k:nat). k < SZ.v j ==> S.index v k == S.index s k))
    decreases (SZ.v n - SZ.v !i)
  {
    let j = !i;
    let x = src.(j);
    dst.(j) <- x;
    i := j +^ 1sz;
  };
}

inline_for_extraction noextract
fn write_if
  (#a : Type0)
  (dst : A.array a) (cond : bool) (idx : SZ.t) (x : a)
  (#d : erased (S.seq a))
  preserves gpu
  requires dst |-> d ** pure (cond ==> SZ.v idx < S.length d)
  ensures exists* (v : S.seq a).
    dst |-> v **
    pure (S.length v == S.length d /\
          (forall (k:nat). k < S.length v ==>
             S.index v k == (if cond && k = SZ.v idx then x else S.index d k)))
{
  A.pts_to_len dst;
  if cond {
    dst.(idx) <- x;
  }
}

inline_for_extraction noextract
fn write_survival
  (output3 : A.array I32.t) (base : SZ.t) (ssz : SZ.t) (keep : bool)
  (#d : erased (S.seq I32.t))
  preserves gpu
  requires output3 |-> d ** pure (SZ.v base + 8 == S.length d)
  ensures exists* (v : S.seq I32.t).
    output3 |-> v **
    pure (S.length v == S.length d /\ S.length v == SZ.v base + 8 /\
          (forall (k:nat). k < SZ.v base ==> S.index v k == S.index d k) /\
          (forall (e:nat). e < 8 ==>
             I32.v (S.index v (SZ.v base + e)) == (if e = SZ.v ssz && keep then 1 else 0)))
{
  A.pts_to_len output3;
  let mut i = 0sz;
  while (!i <^ 8sz)
    invariant exists* (j : SZ.t) (v : S.seq I32.t).
      i |-> j ** output3 |-> v **
      pure (SZ.v j <= 8 /\ S.length v == S.length d /\ S.length v == SZ.v base + 8 /\
            (forall (k:nat). k < SZ.v base ==> S.index v k == S.index d k) /\
            (forall (e:nat). e < SZ.v j ==>
               I32.v (S.index v (SZ.v base + e)) == (if e = SZ.v ssz && keep then 1 else 0)))
    decreases (8 - SZ.v !i)
  {
    let j = !i;
    let flag : I32.t = if ((j <=^ ssz) && (ssz <=^ j) && keep) then 1l else 0l;
    output3.(base +^ j) <- flag;
    i := j +^ 1sz;
  };
}

inline_for_extraction noextract
let slot_of (ssz : SZ.t{SZ.v ssz < 8}) (rank : I64.t) (ok : bool)
  : r:SZ.t{ok /\ 0 <= I64.v rank /\ I64.v rank < 32 ==> SZ.v r == 32 * SZ.v ssz + I64.v rank}
  = if ok && I64.lte 0L rank && I64.lt rank 32L then begin
      small_i64_to_sz rank;
      32sz *^ ssz +^ SZ.uint32_to_sizet (Cast.int64_to_uint32 rank)
    end
    else 0sz

#push-options "--z3rlimit 40"
inline_for_extraction noextract
fn kernel
  (input0 : A.array I32.t) (input1 : A.array F32.t)
  (input2 : A.array I64.t) (input3 : A.array I64.t) (input4 : A.array F32.t)
  (input5 : A.array I32.t) (input6 : I64.t)
  (output0 : A.array I64.t) (output1 : A.array I64.t)
  (output2 : A.array F32.t) (output3 : A.array I32.t)
  (#p0 #p1 #p2 #p3 #p4 #p5 : perm)
  (#s0 : erased (S.seq I32.t)) (#s1 : erased (S.seq F32.t))
  (#s2 : erased (S.seq I64.t)) (#s3 : erased (S.seq I64.t))
  (#s4 : erased (S.seq F32.t)) (#s5 : erased (S.seq I32.t))
  (#d0 : erased (S.seq I64.t)) (#d1 : erased (S.seq I64.t))
  (#d2 : erased (S.seq F32.t)) (#d3 : erased (S.seq I32.t))
  preserves
    gpu **
    input0 |-> Frac p0 s0 ** input1 |-> Frac p1 s1 ** input2 |-> Frac p2 s2 **
    input3 |-> Frac p3 s3 ** input4 |-> Frac p4 s4 ** input5 |-> Frac p5 s5
  requires
    output0 |-> d0 ** output1 |-> d1 ** output2 |-> d2 ** output3 |-> d3 **
    pure (A.length input0 == 1 /\ A.length input1 == 1 /\ A.length input2 == 8 /\
          A.length input3 == 256 /\ A.length input4 == 256 /\
          A.length output0 == 8 /\ A.length output1 == 256 /\ A.length output2 == 256 /\
          A.length input5 == 8 * I64.v input6 /\
          A.length output3 == 8 * I64.v input6 + 8)
  ensures
    exists* (o0 : S.seq I64.t) (o1 : S.seq I64.t) (o2 : S.seq F32.t) (o3 : S.seq I32.t).
      output0 |-> o0 ** output1 |-> o1 ** output2 |-> o2 ** output3 |-> o3 **
      pure (base_capacity_routing_relation s0 s1 s2 s3 s4 s5 (I64.v input6) o0 o1 o2 o3)
{
  A.pts_to_len input0; A.pts_to_len input1; A.pts_to_len input2;
  A.pts_to_len input3; A.pts_to_len input4; A.pts_to_len input5;
  A.pts_to_len output0; A.pts_to_len output1; A.pts_to_len output2;
  A.pts_to_len output3;
  if (I64.lte 0L input6 && I64.lt input6 256L) {
    let sel = input0.(0sz);
    if (I32.lte 0l sel && I32.lt sel 8l) {
      small_i64_to_sz input6;
      small_i32_to_sz sel;
      let tsz = SZ.uint32_to_sizet (Cast.int64_to_uint32 input6);
      let ssz = SZ.uint32_to_sizet (Cast.int32_to_uint32 sel);
      let sp = input1.(0sz);
      let c = div sp sp;
      copy_prefix input2 output0 8sz;
      copy_prefix input3 output1 256sz;
      copy_prefix input4 output2 256sz;
      let base = 8sz *^ tsz;
      copy_prefix input5 output3 base;
      let rank = input2.(ssz);
      let nr = (if I64.lt rank max_i64 then I64.add rank 1L else rank);
      output0.(ssz) <- nr;
      write_survival output3 base ssz (I64.lt rank 32L);
      let ok = (I64.lte 0L rank && I64.lt rank 32L);
      let slot = slot_of ssz rank ok;
      write_if output1 ok slot input6;
      write_if output2 ok slot c;
      with o0 o1 o2 o3.
        assert (output0 |-> o0 ** output1 |-> o1 ** output2 |-> o2 ** output3 |-> o3);
      assert pure (step_post (I32.v (S.index s0 0)) (S.index s2 (I32.v (S.index s0 0)))
                     (div (S.index s1 0) (S.index s1 0)) (I64.v input6)
                     s2 s3 s4 s5 o0 o1 o2 o3);
      step_relation s0 s1 s2 s3 s4 s5 (I64.v input6) o0 o1 o2 o3;
      assert pure (base_capacity_routing_relation s0 s1 s2 s3 s4 s5 (I64.v input6) o0 o1 o2 o3);
      ()
    } else {
      out_of_domain s0 s1 s2 s3 s4 s5 (I64.v input6) d0 d1 d2 d3;
    }
  } else {
    out_of_domain s0 s1 s2 s3 s4 s5 (I64.v input6) d0 d1 d2 d3;
  }
}
#pop-options

instance send_global_array_contents
  (#a : Type0)
  (arr : A.array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (input0 : A.larray I32.t 1 { is_global input0 })
  (input1 : A.larray F32.t 1 { is_global input1 })
  (input2 : A.larray I64.t 8 { is_global input2 })
  (input3 : A.larray I64.t 256 { is_global input3 })
  (input4 : A.larray F32.t 256 { is_global input4 })
  (input5 : A.array I32.t { is_global input5 })
  (input6 : I64.t)
  (output0 : A.larray I64.t 8 { is_global output0 })
  (output1 : A.larray I64.t 256 { is_global output1 })
  (output2 : A.larray F32.t 256 { is_global output2 })
  (output3 : A.array I32.t { is_global output3 })
  (#p0 #p1 #p2 #p3 #p4 #p5 : perm)
  (#s0 : erased (S.seq I32.t))
  (#s1 : erased (S.seq F32.t))
  (#s2 : erased (S.seq I64.t))
  (#s3 : erased (S.seq I64.t))
  (#s4 : erased (S.seq F32.t))
  (#s5 : erased (S.seq I32.t))
  preserves
    cpu **
    on gpu_loc (input0 |-> Frac p0 s0) **
    on gpu_loc (input1 |-> Frac p1 s1) **
    on gpu_loc (input2 |-> Frac p2 s2) **
    on gpu_loc (input3 |-> Frac p3 s3) **
    on gpu_loc (input4 |-> Frac p4 s4) **
    on gpu_loc (input5 |-> Frac p5 s5)
  requires
    (exists* (d0 : S.seq I64.t). on gpu_loc (output0 |-> d0)) **
    (exists* (d1 : S.seq I64.t). on gpu_loc (output1 |-> d1)) **
    (exists* (d2 : S.seq F32.t). on gpu_loc (output2 |-> d2)) **
    (exists* (d3 : S.seq I32.t). on gpu_loc (output3 |-> d3)) **
    pure (A.length input5 == 8 * I64.v input6 /\
          A.length output3 == 8 * I64.v input6 + 8)
  ensures
    exists* (o0 : S.seq I64.t) (o1 : S.seq I64.t) (o2 : S.seq F32.t) (o3 : S.seq I32.t).
      on gpu_loc (output0 |-> o0) **
      on gpu_loc (output1 |-> o1) **
      on gpu_loc (output2 |-> o2) **
      on gpu_loc (output3 |-> o3) **
      pure (base_capacity_routing_relation s0 s1 s2 s3 s4 s5 (I64.v input6) o0 o1 o2 o3)
{
  with d0. assert on gpu_loc (output0 |-> d0);
  with d1. assert on gpu_loc (output1 |-> d1);
  with d2. assert on gpu_loc (output2 |-> d2);
  with d3. assert on gpu_loc (output3 |-> d3);
  launch_kernel_1 (fun () ->
    kernel input0 input1 input2 input3 input4 input5 input6
      output0 output1 output2 output3
      #p0 #p1 #p2 #p3 #p4 #p5 #s0 #s1 #s2 #s3 #s4 #s5 #d0 #d1 #d2 #d3);
}
