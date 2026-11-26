module KPlay.Functional

#lang-pulse

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module I32 = FStar.Int32
module SZ = Kuiper.SizeT

let base_scan_succ (row : S.seq f32) (j : nat{1 <= j /\ j < S.length row})
  : Lemma (base_scan row (j + 1) ==
           (let p = base_scan row j in
            if lt (snd p) (S.index row j) then (j, S.index row j) else p))
  = ()

inline_for_extraction noextract
fn router_argmax
  (softmax : larray f32 8)
  (selected_output : larray I32.t 1)
  (selected_probability_output : larray f32 1)
  (#row : erased (S.seq f32))
  (#s0 : erased (S.seq I32.t))
  (#p0 : erased (S.seq f32))
  preserves gpu ** softmax |-> row
  requires selected_output |-> s0 ** selected_probability_output |-> p0
  ensures exists* s p.
    selected_output |-> s ** selected_probability_output |-> p **
    pure (base_argmax_relation row s p)
{
  pts_to_len softmax;
  pts_to_len selected_output;
  pts_to_len selected_probability_output;
  let mut selected : I32.t = 0l;
  let p0v = softmax.(0sz);
  let mut selected_probability : f32 = p0v;
  let mut expert : sz = 1sz;
  let mut expert_i : I32.t = 1l;
  while (!expert <^ 8sz)
    invariant exists* (j : sz) (ji : I32.t) (e : I32.t) (b : f32).
      expert |-> j ** expert_i |-> ji ** selected |-> e ** selected_probability |-> b **
      pure (1 <= SZ.v j /\ SZ.v j <= 8 /\ I32.v ji == SZ.v j /\
            I32.v e == fst (base_scan row (SZ.v j)) /\
            b == snd (base_scan row (SZ.v j)))
    decreases (8 - SZ.v !expert)
  {
    let j = !expert;
    let ji = !expert_i;
    let probability = softmax.(j);
    let b = !selected_probability;
    base_scan_succ row (SZ.v j);
    if (lt b probability) {
      selected := ji;
      selected_probability := probability;
    };
    expert := j +^ 1sz;
    expert_i := I32.add ji 1l;
  };
  let e = !selected;
  let b = !selected_probability;
  selected_output.(0sz) <- e;
  selected_probability_output.(0sz) <- b;
}

instance send_global_array_contents
  (#a : Type0)
  (arr : array a{is_global arr})
  (#f : perm)
  (contents : S.seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (input0 : larray f32 8 { is_global input0 })
  (output0 : larray I32.t 1 { is_global output0 })
  (output1 : larray f32 1 { is_global output1 })
  (#row : erased (S.seq f32))
  preserves cpu ** on gpu_loc (input0 |-> row)
  requires exists* s0. on gpu_loc (output0 |-> s0)
  requires exists* p0. on gpu_loc (output1 |-> p0)
  ensures exists* s p.
    on gpu_loc (output0 |-> s) **
    on gpu_loc (output1 |-> p) **
    pure (base_argmax_relation row s p)
{
  with s0. assert on gpu_loc (output0 |-> s0);
  with p0. assert on gpu_loc (output1 |-> p0);
  launch_kernel_1 (fun () -> router_argmax input0 output0 output1 #row #s0 #p0);
  ()
}
