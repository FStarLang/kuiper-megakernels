module Kuiper.Float.Casts.Raw

(* Representation conversions only. Semantic casts preserve the ideal field. *)
module F16 = Kuiper.Float16.Raw
module BF16 = Kuiper.BFloat16.Raw
module F32 = Kuiper.Float32.Raw
module F64 = Kuiper.Float64.Raw

val cast_f16_to_f32 : F16.t -> F32.t
val cast_f16_to_f64 : F16.t -> F64.t
val cast_f32_to_f16 : F32.t -> F16.t
val cast_f32_to_f64 : F32.t -> F64.t
val cast_bf16_to_f32 : BF16.t -> F32.t
val cast_f32_to_bf16 : F32.t -> BF16.t
val cast_f16_to_bf16 : F16.t -> BF16.t
val cast_bf16_to_f16 : BF16.t -> F16.t
val cast_bf16_to_f64 : BF16.t -> F64.t
val cast_f64_to_bf16 : F64.t -> BF16.t
val cast_f64_to_f16 : F64.t -> F16.t
val cast_f64_to_f32 : F64.t -> F32.t

(* ====================================================================
   TRUSTED ASSUMPTIONS added by RESOLVE
   (src/KPlay/kuiper_support/numeric/bf16-widening-model.patch;
   listed in the RESOLVE README under "Trusted assumptions").

   BF16 -> F32 widening is exact and maps NaN to NaN. Stated as two laws
   over the abstract representations: the floating class (Finite /
   Infinite / NaN) is preserved, and native <= is preserved.

   Sources:
   - PTX ISA, cvt: a rounding modifier is mandatory only for
     float-to-float conversions whose destination type is smaller than
     the source; the widening cvt.f32.bf16 behind __bfloat162float takes
     none, i.e. it is exact.
   - IEEE 754-2019 5.4.2 (conversion to a wider format is exact) and
     6.2 (NaN operands give NaN results).
   Evidence: src/KPlay/tests/test_bf16_widening_gpu.py runs all 65,536
   BF16 values through cvt.f32.bf16 on the GPU and checks
   result == bits << 16 (NaN in -> NaN out), which implies both laws.
   ==================================================================== *)
val cast_bf16_to_f32_kind (x:BF16.t) :
  Lemma (F32.kind (cast_bf16_to_f32 x) == BF16.kind x)

val cast_bf16_to_f32_lte (x y:BF16.t) :
  Lemma (F32.lte (cast_bf16_to_f32 x) (cast_bf16_to_f32 y) == BF16.lte x y)
