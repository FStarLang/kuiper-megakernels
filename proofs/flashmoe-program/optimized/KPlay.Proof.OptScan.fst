module KPlay.Proof.OptScan

open Kuiper
open Kuiper.Approximates
open KPlay.SharedSpec
module S = FStar.Seq
module F32 = Kuiper.Float32
module FB = Kuiper.Floating.Base

let lt_irrefl (x:F32.t) : Lemma (~(lt x x)) =
  if lt x x then F32.lt_ordered x x else ()

let lte_of_not_lt (b y:F32.t)
  : Lemma (requires ~(is_nan b) /\ ~(lt b y))
          (ensures is_nan y \/ lte y b)
  = FStar.Classical.move_requires (FB.negate_lt_is_lte b) y

let scan_inv (row:S.seq F32.t) (n:nat{n <= S.length row}) (j:int) (b:F32.t) : prop =
  0 <= j /\ j < (if n = 0 then 1 else n) /\ j < S.length row + 1 /\
  ((j == 0 /\ b == ninf) \/ (j < n /\ b == S.index row j)) /\
  (forall (i:nat). i < n ==> ~(lt b (S.index row i))) /\
  (forall (i:nat). i < j /\ i < S.length row ==> is_nan (S.index row i) \/
     (exists (v:F32.t). lte (S.index row i) v /\ lt v (S.index row j)))

#push-options "--z3rlimit 50 --fuel 1 --ifuel 1"
let rec opt_scan_props (row:S.seq F32.t) (n:nat{n <= S.length row})
  : Lemma (ensures scan_inv row n (fst (opt_scan row n)) (snd (opt_scan row n)))
          (decreases n)
  = if n = 0 then ()
    else begin
      opt_scan_props row (n - 1);
      let p = opt_scan row (n - 1) in
      let b = snd p in
      let x = S.index row (n - 1) in
      if lt b x then begin
        F32.lt_ordered b x;
        lt_irrefl x;
        let aux (i:nat{i < n - 1})
          : Lemma (~(lt x (S.index row i)) /\
                   (is_nan (S.index row i) \/
                    (exists (v:F32.t). lte (S.index row i) v /\ lt v x)))
          = (if lt x (S.index row i) then F32.lt_transitive b x (S.index row i));
            lte_of_not_lt b (S.index row i)
        in
        FStar.Classical.forall_intro aux;
        assert (opt_scan row n == (n - 1, x))
      end else begin
        assert (opt_scan row n == p)
      end
    end
#pop-options

let opt_scan_allowed (row:S.seq F32.t)
  : Lemma (requires S.length row == experts)
          (ensures allowed_choice row (fst (opt_scan row experts)) /\
                   (let j = fst (opt_scan row experts) in
                    lt ninf (S.index row j) ==> snd (opt_scan row experts) == S.index row j))
  = opt_scan_props row experts
