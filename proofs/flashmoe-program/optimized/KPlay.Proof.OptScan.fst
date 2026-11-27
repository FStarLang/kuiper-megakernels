module KPlay.Proof.OptScan

(* The optimized argmax scan (locally patched to seed with row[0], like base:
   first strict maximum) yields an allowed top-1 choice over the row it
   consumed, and its selected probability is that row entry. *)

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module F32 = Kuiper.Float32
module FB = Kuiper.Floating.Base

#set-options "--z3rlimit 20 --fuel 1 --ifuel 1"

let lt_irrefl (x:F32.t) : Lemma (~(lt x x)) =
  if lt x x then F32.lt_ordered x x else ()

let lte_of_not_lt (b y:F32.t)
  : Lemma (requires ~(is_nan b) /\ ~(lt b y))
          (ensures is_nan y \/ lte y b)
  = FStar.Classical.move_requires (FB.negate_lt_is_lte b) y

let scan_inv (row:S.seq F32.t) (n:nat{1 <= n /\ n <= S.length row}) : prop =
  let j = fst (base_scan row n) in
  let b = snd (base_scan row n) in
  0 <= j /\ j < n /\ b == S.index row j /\
  (forall (i:nat). i < n ==> ~(lt b (S.index row i))) /\
  (forall (i:nat). i < j ==>
     is_nan (S.index row i) \/ (exists (v:F32.t). lte (S.index row i) v /\ lt v b))

#push-options "--z3rlimit 50"
let rec scan_props (row:S.seq F32.t) (n:nat{1 <= n /\ n <= S.length row})
  : Lemma (ensures scan_inv row n) (decreases n)
  = if n = 1 then lt_irrefl (S.index row 0)
    else begin
      scan_props row (n - 1);
      let b = snd (base_scan row (n - 1)) in
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
        FStar.Classical.forall_intro aux
      end
    end
#pop-options

let opt_scan_allowed (row:S.seq F32.t)
  : Lemma (requires S.length row == experts)
          (ensures allowed_choice row (fst (base_scan row experts)) /\
                   snd (base_scan row experts) == S.index row (fst (base_scan row experts)))
  = scan_props row experts
