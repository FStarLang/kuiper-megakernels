module KPlay.Functional

#lang-pulse
open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module SZ = Kuiper.SizeT

inline_for_extraction let () = ()

let rec base_scan_bound (row:S.seq f32) (n:nat{1 <= n /\ n <= S.length row})
  : Lemma (ensures 0 <= fst (base_scan row n) /\ fst (base_scan row n) < n)
          (decreases n)
  = if n = 1 then () else base_scan_bound row (n - 1)

let base_scan_unfold (row:S.seq f32) (n:nat{1 <= n /\ n < S.length row})
  : Lemma (base_scan row (n + 1) ==
           (let p = base_scan row n in
            if lt (snd p) (S.index row n) then (n, S.index row n) else p))
  = ()

let scan_state (srow:S.seq f32) (j:nat) (si:SZ.t) (bv:f32) : prop =
  1 <= j /\ j <= S.length srow /\
  SZ.v si == fst (base_scan srow j) /\ bv == snd (base_scan srow j)

(* Seeded with (0, row[0]) and scanning experts 1..7 (local patch
   flashmoe-argmax-start-row0). *)
inline_for_extraction noextract
fn argmax_kernel
  (row : larray f32 8) (sel : larray i32 1) (sp : larray f32 1)
  (#srow : erased (lseq f32 8))
  (#s0 : erased (lseq i32 1)) (#p0 : erased (lseq f32 1))
  ()
  preserves gpu ** row |-> srow
  requires sel |-> s0 ** sp |-> p0
  ensures exists* (s1 : lseq i32 1) (p1 : lseq f32 1).
    sel |-> s1 ** sp |-> p1 ** pure (optimized_argmax_relation srow s1 p1)
{
  let mut selected : SZ.t = 0sz;
  let x0 = row.(0sz);
  let mut best : f32 = x0;
  let mut e = 1sz;
  while (!e <^ 8sz)
    invariant exists* (j : SZ.t{SZ.v j <= 8}) (si : SZ.t) (bv : f32).
      e |-> j ** selected |-> si ** best |-> bv **
      pure (scan_state srow (SZ.v j) si bv)
    decreases (8 - SZ.v !e)
  {
    let j = !e;
    let x = row.(j);
    let b = !best;
    base_scan_unfold srow (SZ.v j);
    if (lt b x) {
      selected := j;
      best := x;
    };
    e := j +^ 1sz;
  };
  let si = !selected;
  let bv = !best;
  base_scan_bound srow 8;
  let s32 : i32 = FStar.Int.Cast.uint32_to_int32 (SZ.sizet_to_u32 si);
  sel.(0sz) <- s32;
  sp.(0sz) <- bv;
}

instance send_global_array
  (#a : Type0) (arr : array a{is_global arr}) (#f : perm) (contents : seq a)
  : is_send_across gpu_of (Pulse.Lib.Array.pts_to arr #f contents)
  = Kuiper.Array.Core.is_send_pts_to arr #f contents

fn kplay_run
  (row : larray f32 8 { is_global row })
  (sel : larray i32 1 { is_global sel })
  (sp : larray f32 1 { is_global sp })
  (#srow : erased (lseq f32 8))
  preserves cpu ** on gpu_loc (row |-> srow)
  requires (exists* (s0 : lseq i32 1). on gpu_loc (sel |-> s0))
  requires (exists* (p0 : lseq f32 1). on gpu_loc (sp |-> p0))
  ensures exists* (s1 : lseq i32 1) (p1 : lseq f32 1).
    on gpu_loc (sel |-> s1) ** on gpu_loc (sp |-> p1) **
    pure (optimized_argmax_relation srow s1 p1)
{
  with s0. assert on gpu_loc (sel |-> s0);
  with p0. assert on gpu_loc (sp |-> p0);
  launch_kernel_1 (fun () -> argmax_kernel row sel sp #srow #s0 #p0 ());
}
