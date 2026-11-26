module Kuiper.KPlay.CpuExp

open Kuiper

(* Scalar CUDA form of the vectorized CPU expf used by PyTorch's CPU softmax
   (AVX512), so a CUDA program can reproduce a CPU reference bit for bit; the
   FlashMoE base reduction was accepted bit-exact with it. It is an opaque
   complete program (include/kplay_cpu_exp.cuh). TRUSTED: like Kuiper's CUDA
   expf (fexp), it is assumed to approximate the real exponential. The routine
   is written for inputs <= 0 (e.g. max-shifted softmax scores) and returns
   +0.0 below -104, where exp underflows in binary32. *)
val cpu_exp_f32 : Kuiper.Float32.t -> Kuiper.Float32.t

val cpu_exp_f32_approx (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (cpu_exp_f32 x) (exp r))
      [SMTPat (cpu_exp_f32 x); SMTPat (v_approximates x r)]
