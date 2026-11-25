module Kuiper.KPlay.Transcendentals

open Kuiper

inline_for_extraction noextract
let sigmoid_real (x : real) : real =
  exp_positive (0.0R -. x);
  1.0R /. (1.0R +. exp (0.0R -. x))

inline_for_extraction noextract
let swish_real (x : real) : real =
  x *. sigmoid_real x

inline_for_extraction noextract
let tanh_real (x : real) : real =
  exp_positive (2.0R *. x);
  let e = exp (2.0R *. x) in
  (e -. 1.0R) /. (e +. 1.0R)

inline_for_extraction noextract
let selu_p027_real (x : real) : real =
  1.0507009873554804934193349852946R *.
    (if x >. 0.0R then x else
      1.6732632423543772848170429916717R *. (exp x -. 1.0R))

inline_for_extraction noextract
let softplus_p029_real (x : real) : real =
  if x >. 20.0R then x else log (1.0R +. exp x)

inline_for_extraction noextract
let elu_p031_real (x : real) : real =
  if x >. 0.0R then x else exp x -. 1.0R

(* These are opaque complete scalar programs, not floating expm1/log1p
   primitives. Their only refinement targets are the fixed definitions above.
   KPlay additionally requires the source/semantic hash bindings in its
   capability manifest; the pinned CUDA header preserves each source variant. *)
val selu_p027_base_f32 : Kuiper.Float32.t -> Kuiper.Float32.t
val selu_p027_ftz_f32 : Kuiper.Float32.t -> Kuiper.Float32.t
val softplus_p029_base_f32 : Kuiper.Float32.t -> Kuiper.Float32.t
val softplus_p029_fast_f32 : Kuiper.Float32.t -> Kuiper.Float32.t
val elu_p031_base_f32 : Kuiper.Float32.t -> Kuiper.Float32.t

val selu_p027_base_f32_approx
  (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (selu_p027_base_f32 x) (selu_p027_real r))

val selu_p027_ftz_f32_approx
  (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (selu_p027_ftz_f32 x) (selu_p027_real r))

val softplus_p029_base_f32_approx
  (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (softplus_p029_base_f32 x) (softplus_p029_real r))

val softplus_p029_fast_f32_approx
  (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (softplus_p029_fast_f32 x) (softplus_p029_real r))

val elu_p031_base_f32_approx
  (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (elu_p031_base_f32 x) (elu_p031_real r))

val sigmoid_approx
  (#a : Type)
  {| scalar a, floating a, real_like a, floating_real_like a |}
  (x : a)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures
        v_approximates
          (div #a one (add #a one (fexp #a (sub #a zero x))))
          (sigmoid_real r))

val swish_approx
  (#a : Type)
  {| scalar a, floating a, real_like a, floating_real_like a |}
  (x : a)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures
        v_approximates
          (mul #a x
            (div #a one (add #a one (fexp #a (sub #a zero x)))))
          (swish_real r))

val fast_ftz_swish_f32 : Kuiper.Float32.t -> Kuiper.Float32.t

val fast_ftz_swish_f32_approx
  (x : Kuiper.Float32.t)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures
        v_approximates
          (fast_ftz_swish_f32 x)
          (swish_real r))

[@@FStar.Tactics.Typeclasses.fundeps [1]]
class tanh_real_like
  (a : Type)
  {| scalar a, floating a, real_like a, floating_real_like a |}
= {
  tanh_approx :
    x:a -> r:real ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (tanh #a x) (tanh_real r));
}

val tanh_approx_pat
  (a : Type)
  {|
    scalar a,
    floating a,
    real_like a,
    floating_real_like a,
    tr : tanh_real_like a
  |}
  (x : a)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (tanh #a x) (tanh_real r))
      [SMTPat (tanh #a x);
       SMTPat (v_approximates x r);
       SMTPat (has_type tr (tanh_real_like a))]

(* Kuiper's floating approximation instances are already trusted. These
   matching tanh instances make that existing trust boundary explicit for
   physical tanh, which is absent from floating_real_like. *)
instance val float16_tanh_real_like :
  tanh_real_like Kuiper.Float16.t

instance val bfloat16_tanh_real_like :
  tanh_real_like Kuiper.BFloat16.t

instance val float32_tanh_real_like :
  tanh_real_like Kuiper.Float32.t

instance val float64_tanh_real_like :
  tanh_real_like Kuiper.Float64.t
