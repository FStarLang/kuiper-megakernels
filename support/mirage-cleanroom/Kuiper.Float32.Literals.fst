module Kuiper.Float32.Literals

open FStar.Real

(* Certificates support unsigned fixed-point decimals, not arbitrary C syntax. *)
noextract
let digit (c : FStar.Char.char) : option nat =
  match c with
  | '0' -> Some 0 | '1' -> Some 1 | '2' -> Some 2 | '3' -> Some 3
  | '4' -> Some 4 | '5' -> Some 5 | '6' -> Some 6 | '7' -> Some 7
  | '8' -> Some 8 | '9' -> Some 9 | _ -> None

noextract
let rec fraction (cs : list FStar.Char.char) (n : nat) (d : pos)
  : Tot (option (nat & pos)) (decreases cs) =
  match cs with
  | [] -> Some (n, d)
  | c :: rest ->
    match digit c with
    | None -> None
    | Some v -> fraction rest (10 * n + v) (10 * d)

noextract
let rec integral (cs : list FStar.Char.char) (n : nat) (seen : bool)
  : Tot (option (nat & pos)) (decreases cs) =
  match cs with
  | '.' :: rest ->
    if seen then (match rest with [] -> None | _ -> fraction rest n 1)
    else None
  | c :: rest ->
    (match digit c with
    | None -> None
    | Some v -> integral rest (10 * n + v) true)
  | [] -> None

noextract
let value (s:string) : FStar.Real.real =
  match integral (FStar.String.list_of_string s) 0 false with
  | Some (n,d) -> FStar.Real.of_int n /. FStar.Real.of_int d
  | None -> 0.0R

noextract
let rec binary_power (n : nat) : Tot pos (decreases n) =
  if n = 0 then 1 else 2 * binary_power (n - 1)
