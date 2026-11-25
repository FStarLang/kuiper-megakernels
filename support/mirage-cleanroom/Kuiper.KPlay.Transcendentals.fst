module Kuiper.KPlay.Transcendentals

open Kuiper

(* A closed, source-bound physical trust boundary. In particular it supplies
   neither an expm1/log1p arithmetic law nor a physical-to-real comparison law.
   The opaque computations extract only to the separately pinned CUDA header. *)
private noextract
noeq type activation_programs = {
  selu_base : Kuiper.Float32.t -> Kuiper.Float32.t;
  selu_ftz : Kuiper.Float32.t -> Kuiper.Float32.t;
  softplus_base : Kuiper.Float32.t -> Kuiper.Float32.t;
  softplus_fast : Kuiper.Float32.t -> Kuiper.Float32.t;
  elu_base : Kuiper.Float32.t -> Kuiper.Float32.t;
  selu_base_refines :
    x:Kuiper.Float32.t -> r:real ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (selu_base x) (selu_p027_real r));
  selu_ftz_refines :
    x:Kuiper.Float32.t -> r:real ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (selu_ftz x) (selu_p027_real r));
  softplus_base_refines :
    x:Kuiper.Float32.t -> r:real ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (softplus_base x) (softplus_p029_real r));
  softplus_fast_refines :
    x:Kuiper.Float32.t -> r:real ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (softplus_fast x) (softplus_p029_real r));
  elu_base_refines :
    x:Kuiper.Float32.t -> r:real ->
    Lemma
      (requires v_approximates x r)
      (ensures v_approximates (elu_base x) (elu_p031_real r));
}

private noextract
let trusted_activation_programs : activation_programs = magic()

(* KaRaMeL emits extern declarations: bind actual functions, not macro aliases,
   and retain their device linkage after including the pinned header. *)
[@@custard_extern "kplay_selu_p027_base_f32";
   CPrologue "#include \"kplay_transcendentals.cuh\"\n__device__"]
let selu_p027_base_f32 (x : Kuiper.Float32.t) : Kuiper.Float32.t =
  trusted_activation_programs.selu_base x

[@@custard_extern "kplay_selu_p027_ftz_f32";
   CPrologue "#include \"kplay_transcendentals.cuh\"\n__device__"]
let selu_p027_ftz_f32 (x : Kuiper.Float32.t) : Kuiper.Float32.t =
  trusted_activation_programs.selu_ftz x

[@@custard_extern "kplay_softplus_p029_base_f32";
   CPrologue "#include \"kplay_transcendentals.cuh\"\n__device__"]
let softplus_p029_base_f32 (x : Kuiper.Float32.t) : Kuiper.Float32.t =
  trusted_activation_programs.softplus_base x

[@@custard_extern "kplay_softplus_p029_fast_f32";
   CPrologue "#include \"kplay_transcendentals.cuh\"\n__device__"]
let softplus_p029_fast_f32 (x : Kuiper.Float32.t) : Kuiper.Float32.t =
  trusted_activation_programs.softplus_fast x

[@@custard_extern "kplay_elu_p031_base_f32";
   CPrologue "#include \"kplay_transcendentals.cuh\"\n__device__"]
let elu_p031_base_f32 (x : Kuiper.Float32.t) : Kuiper.Float32.t =
  trusted_activation_programs.elu_base x

let selu_p027_base_f32_approx (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (selu_p027_base_f32 x) (selu_p027_real r))
= trusted_activation_programs.selu_base_refines x r

let selu_p027_ftz_f32_approx (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (selu_p027_ftz_f32 x) (selu_p027_real r))
= trusted_activation_programs.selu_ftz_refines x r

let softplus_p029_base_f32_approx (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (softplus_p029_base_f32 x) (softplus_p029_real r))
= trusted_activation_programs.softplus_base_refines x r

let softplus_p029_fast_f32_approx (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (softplus_p029_fast_f32 x) (softplus_p029_real r))
= trusted_activation_programs.softplus_fast_refines x r

let elu_p031_base_f32_approx (x : Kuiper.Float32.t) (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures v_approximates (elu_p031_base_f32 x) (elu_p031_real r))
= trusted_activation_programs.elu_base_refines x r

let sigmoid_approx
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
=
  let negated = sub #a zero x in
  let exponential = fexp #a negated in
  let denominator = add #a one exponential in
  assert (negated %~ (0.0R -. r));
  assert (exponential %~ exp (0.0R -. r));
  a_add one exponential 1.0R (exp (0.0R -. r));
  exp_positive (0.0R -. r);
  assert (1.0R +. exp (0.0R -. r) =!= 0.0R);
  assert (div #a one denominator %~ (1.0R /. (1.0R +. exp (0.0R -. r))))

let swish_approx
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
=
  sigmoid_approx x r;
  a_mul x
    (div #a one (add #a one (fexp #a (sub #a zero x))))
    r
    (sigmoid_real r)

[@@custard_extern "kplay_fast_ftz_swish_f32";
   CPrologue "#include \"kplay_transcendentals.cuh\"\n__device__"]
let fast_ftz_swish_f32 (x : Kuiper.Float32.t) : Kuiper.Float32.t =
  div #Kuiper.Float32.t x
    (add #Kuiper.Float32.t one
      (fexp #Kuiper.Float32.t
        (sub #Kuiper.Float32.t zero x)))

let fast_ftz_swish_f32_approx
  (x : Kuiper.Float32.t)
  (r : real)
  : Lemma
      (requires v_approximates x r)
      (ensures
        v_approximates
          (fast_ftz_swish_f32 x)
          (swish_real r))
=
  assert (has_type Kuiper.Float32.is_floating_real_like
    (floating_real_like Kuiper.Float32.t));
  let negated = sub #Kuiper.Float32.t zero x in
  let exponential = fexp #Kuiper.Float32.t negated in
  let denominator = add #Kuiper.Float32.t one exponential in
  assert (negated %~ (0.0R -. r));
  assert (exponential %~ exp (0.0R -. r));
  a_add one exponential 1.0R (exp (0.0R -. r));
  exp_positive (0.0R -. r);
  assert (1.0R +. exp (0.0R -. r) =!= 0.0R);
  assert (div #Kuiper.Float32.t x denominator %~
    (r /. (1.0R +. exp (0.0R -. r))));
  assert (
    r /. (1.0R +. exp (0.0R -. r))
    ==
    swish_real r)

let tanh_approx_pat
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
= tanh_approx x r

instance float16_tanh_real_like :
  tanh_real_like Kuiper.Float16.t = magic()

instance bfloat16_tanh_real_like :
  tanh_real_like Kuiper.BFloat16.t = magic()

instance float32_tanh_real_like :
  tanh_real_like Kuiper.Float32.t = magic()

instance float64_tanh_real_like :
  tanh_real_like Kuiper.Float64.t = magic()
