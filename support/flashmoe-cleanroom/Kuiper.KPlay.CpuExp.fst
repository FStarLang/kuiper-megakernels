module Kuiper.KPlay.CpuExp

open Kuiper

private noextract
noeq type program = {
  f : Kuiper.Float32.t -> Kuiper.Float32.t;
  refines :
    x:Kuiper.Float32.t -> r:real ->
    Lemma (requires v_approximates x r) (ensures v_approximates (f x) (exp r));
}

(* The single trusted law of this module; see the interface. *)
private noextract
let trusted_program : program = magic()

[@@custard_extern "kplay_cpu_exp_f32";
   CPrologue "#include \"kplay_cpu_exp.cuh\"\n__device__"]
let cpu_exp_f32 (x : Kuiper.Float32.t) : Kuiper.Float32.t = trusted_program.f x

let cpu_exp_f32_approx (x : Kuiper.Float32.t) (r : real) = trusted_program.refines x r
