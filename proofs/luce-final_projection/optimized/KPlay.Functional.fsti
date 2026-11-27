module KPlay.Functional

#lang-pulse

open Kuiper
module S = FStar.Seq
module SZ = Kuiper.SizeT
module F32 = Kuiper.Float32
module BF16 = Kuiper.BFloat16
module I64 = FStar.Int64
module SS = KPlay.SharedSpec

inline_for_extraction let unit_marker : unit = ()

(* ABI order: table, normw, hidden, history (inputs), d (scalar); scores, history' (outputs).
   hn is the element count of history (S*1024), passed by the wrapper. *)
fn kplay_run
  (hn:SZ.t)
  (tab:larray BF16.t 254279680{is_global tab})
  (nw:larray BF16.t 1024{is_global nw})
  (x:larray F32.t 1024{is_global x})
  (hin:larray BF16.t (SZ.v hn){is_global hin})
  (sc:larray F32.t 248320{is_global sc})
  (hout:larray BF16.t (SZ.v hn){is_global hout})
  (d:I64.t)
  (#ft #fw #fx #fh:perm)
  (#ts #ns:erased (S.seq BF16.t)) (#xs:erased (S.seq F32.t)) (#hs:erased (S.seq BF16.t))
  (#sc0:erased (S.seq F32.t)) (#ho0:erased (S.seq BF16.t))
  preserves cpu ** on gpu_loc (tab |-> Frac ft ts) ** on gpu_loc (nw |-> Frac fw ns) **
            on gpu_loc (x |-> Frac fx xs) ** on gpu_loc (hin |-> Frac fh hs)
  requires on gpu_loc (sc |-> sc0) ** on gpu_loc (hout |-> ho0)
  ensures exists* (sv:S.seq F32.t) (hv:S.seq BF16.t).
    on gpu_loc (sc |-> sv) ** on gpu_loc (hout |-> hv) **
    pure (SS.optimized_final_projection_relation ts ns xs hs (I64.v d) sv hv)
