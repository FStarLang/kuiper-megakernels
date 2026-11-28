module KPlay.Numbers
(* Number types for typed specs. A typed spec is written once over
   `number` instances; the kernels are proved against its float instances,
   and tightness.py checks that its exact-real instance defines a function.
   Every instance is fixed here, once, for all specs. *)

open FStar.Real
open Kuiper
open Kuiper.Approximates
module F = Kuiper.Floating.Base
module BF = Kuiper.BFloat16
module F32 = Kuiper.Float32

type row (a:Type0) (n:nat) = i:nat{i < n} -> GTot a

noeq type number (a:Type0) = {
  (* x stands for the real r *)
  approx : a -> real -> prop;
  (* x is a finite value *)
  finite : a -> prop;
  (* the value 0 *)
  zero : a;
  (* j is an index argmax(dim=-1) may return for the row *)
  argmax : n:nat -> row a n -> int -> prop;
  (* v may stand for the model constant c, whose float32 value is c32 *)
  constant : c:real -> c32:real -> v:real -> prop;
}

(* Constants library rule for float types: a kernel may use a constant's
   exact value or its float32 value. *)
let float_constant (c c32 v:real) : prop = v == c \/ v == c32

(* Floats are finite when strictly between -inf and +inf (native order;
   NaN is not). *)
let float_finite (#t:Type0) {| F.floating t |} (x:t) : prop =
  lt (F.neg F.infinity) x /\ lt x F.infinity

(* BF16: x %~ r; argmax as torch.argmax, up to tie order: the first NaN if
   the row has one, otherwise any index whose score is >= every score
   (native order; +0 and -0 tie). *)
let bf16_approx (x:BF.t) (r:real) : prop = x %~ r

let is_nan (x:BF.t) : prop = F.NaN? (F.kind x)

let bf16_argmax (n:nat) (s:row BF.t n) (j:int) : prop =
  0 <= j /\ j < n /\
  ((exists (i:nat{i < n}). is_nan (s i)) ==>
     is_nan (s j) /\ (forall (i:nat{i < n}). i < j ==> ~(is_nan (s i)))) /\
  (~(exists (i:nat{i < n}). is_nan (s i)) ==> (forall (i:nat{i < n}). F.lte (s i) (s j)))

let bf16 : number BF.t =
  { approx = bf16_approx; finite = float_finite; zero = zero; argmax = bf16_argmax;
    constant = float_constant }

(* F32: x %~ r. Argmax library rule: if every score is finite, pick a
   highest one; if any score is NaN or infinite, any pick is allowed. *)
let f32_approx (x:F32.t) (r:real) : prop = x %~ r

let f32_argmax (n:nat) (s:row F32.t n) (j:int) : prop =
  0 <= j /\ j < n /\
  ((forall (i:nat{i < n}). float_finite (s i)) ==> (forall (i:nat{i < n}). F.lte (s i) (s j)))

let f32 : number F32.t =
  { approx = f32_approx; finite = float_finite; zero = zero; argmax = f32_argmax;
    constant = float_constant }

(* Exact reals: a value stands only for itself; argmax is the first maximum;
   a constant is its exact value. *)
let exact_approx (x r:real) : prop = x == r

let exact_argmax (n:nat) (s:row real n) (j:int) : prop =
  0 <= j /\ j < n /\
  (forall (i:nat{i < n}). s i <=. s j) /\
  (forall (i:nat{i < n}). i < j ==> s i <. s j)

let exact : number real =
  { approx = exact_approx; finite = (fun _ -> True); zero = 0.0R; argmax = exact_argmax;
    constant = (fun c _ v -> v == c) }

(* Equality of outputs (same size, equal entries), used by the generated
   uniqueness theorem. *)
let veq (#a:Type0) (#n1 #n2:nat) (x:row a n1) (y:row a n2) : prop =
  n1 == n2 /\ (forall (i:nat{i < n1 /\ i < n2}). x i == y i)

let meq (#a:Type0) (#r1 #c1 #r2 #c2:nat) (x:row (row a c1) r1) (y:row (row a c2) r2) : prop =
  r1 == r2 /\ c1 == c2 /\
  (forall (i:nat{i < r1 /\ i < r2}) (j:nat{j < c1 /\ j < c2}). x i j == y i j)

(* The exact argmax: the first maximum exists and is unique. *)
let rec first_max (#n:pos) (s:row real n) (k:nat{1 <= k /\ k <= n})
  : GTot (j:nat{j < k /\ (forall (i:nat{i < n}). i < k ==> s i <=. s j) /\
                (forall (i:nat{i < n}). i < j ==> s i <. s j)}) (decreases k)
  = if k = 1 then 0
    else let j = first_max s (k - 1) in
         if s j <. s (k - 1) then k - 1 else j

let argmax_unique (#n:nat) (s1 s2:row real n) (j1 j2:int)
  : Lemma (requires veq s1 s2 /\ exact_argmax n s1 j1 /\ exact_argmax n s2 j2)
          (ensures j1 == j2)
  = if j1 < j2 then assert (s2 j1 <. s2 j2 /\ s1 j2 <=. s1 j1)
    else if j2 < j1 then assert (s1 j2 <. s1 j1 /\ s2 j1 <=. s2 j2)
