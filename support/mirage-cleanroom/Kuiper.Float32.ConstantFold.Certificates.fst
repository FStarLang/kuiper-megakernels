module Kuiper.Float32.ConstantFold.Certificates

module Lit = Kuiper.Float32.Literals

noextract
let normal_encoding (mantissa:nat) (exponent:int) : bool =
  8388608 <= mantissa && mantissa < 16777216 &&
  -149 <= exponent && exponent <= 104

noextract
let unit_fraction (exponent:int) : pos & pos =
  if exponent >= 0 then (Lit.binary_power exponent, 1)
  else (1, Lit.binary_power (-exponent))

noextract
let exact_normal (s:string) (mantissa:nat) (exponent:int) : bool =
  normal_encoding mantissa exponent &&
  (match Lit.integral (FStar.String.list_of_string s) 0 false with
   | None -> false
   | Some (n,d) ->
     let (un,ud) = unit_fraction exponent in
     n * ud = mantissa * un * d)

(* Below a normal power of two the spacing halves, except at FLT_MIN. *)
noextract
let normal_cell (mantissa:nat{8388608 <= mantissa}) (exponent:int)
  : nat & nat & pos =
  let (un,ud) = unit_fraction exponent in
  let below = if mantissa = 8388608 && exponent > -149 then 1 else 2 in
  ((4 * mantissa - below) * un, (4 * mantissa + 2) * un, 4 * ud)

(* Reserve 2^-30 ULP at both midpoints for a binary64 intermediate.
   This certifies rounded bits, not equality of ideal real values. *)
noextract
let reciprocal
  (input output:string) (im:nat) (ie:int) (om:nat) (oe:int) : bool =
  if not (normal_encoding om oe) then false
  else
    let (cell_lo,cell_hi,cell_den) = normal_cell om oe in
    let (un,_) = unit_fraction oe in
    let guarded_lo = cell_lo * 268435456 + un in
    let guarded_hi = cell_hi * 268435456 - un in
    let guarded_den = cell_den * 268435456 in
    exact_normal input im ie && exact_normal output om oe &&
    (match Lit.integral (FStar.String.list_of_string input) 0 false with
     | None -> false
     | Some (n,d) ->
       n > 0 &&
       guarded_lo * n < d * guarded_den &&
       d * guarded_den < guarded_hi * n)
