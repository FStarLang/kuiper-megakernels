module KPlay.Proof.Base.Select

(* The base argmax scan (first strict maximum, seeded with row[0]) yields an
   allowed top-1 choice over the row it consumed. Size-generic induction over
   the scan prefix. *)

open Kuiper
open KPlay.SharedSpec
module S = FStar.Seq
module F32 = Kuiper.Float32
module FB = Kuiper.Floating.Base

#set-options "--z3rlimit 20 --fuel 1 --ifuel 1"

let lt_irrefl (x:F32.t) : Lemma (~(lt x x)) =
  Classical.move_requires_2 F32.lt_ordered x x;
  Classical.move_requires_2 FB.negate_lt_is_lte x x;
  Classical.move_requires_2 FB.lte_is_lt_or_eq x x

let lte_of_not_lt (b ri:F32.t)
  : Lemma (requires ~(lt b ri) /\ ~(is_nan b) /\ ~(is_nan ri))
          (ensures lte ri b)
  = FB.negate_lt_is_lte b ri

let scan_inv (row:S.seq F32.t) (n:nat{1 <= n /\ n <= S.length row}) : prop =
  let j = fst (base_scan row n) in
  let b = snd (base_scan row n) in
  0 <= j /\ j < n /\ b == S.index row j /\
  (forall (i:nat). i < n ==> ~(lt b (S.index row i))) /\
  (forall (i:nat). i < j ==>
     is_nan (S.index row i) \/ (exists (v:F32.t). lte (S.index row i) v /\ lt v b))

let earlier_below (row:S.seq F32.t) (b x:F32.t) (i:nat{i < S.length row})
  : Lemma (requires ~(lt b (S.index row i)) /\ lt b x)
          (ensures is_nan (S.index row i) \/
                   (exists (v:F32.t). lte (S.index row i) v /\ lt v x))
  = F32.lt_ordered b x;
    Classical.move_requires_2 lte_of_not_lt b (S.index row i)

let not_above (row:S.seq F32.t) (b x:F32.t) (i:nat{i < S.length row})
  : Lemma (requires ~(lt b (S.index row i)) /\ lt b x)
          (ensures ~(lt x (S.index row i)))
  = Classical.move_requires_3 F32.lt_transitive b x (S.index row i)

let rec scan_lemma (row:S.seq F32.t) (n:nat{1 <= n /\ n <= S.length row})
  : Lemma (ensures scan_inv row n) (decreases n)
  = if n = 1 then lt_irrefl (S.index row 0)
    else begin
      scan_lemma row (n - 1);
      let b = snd (base_scan row (n - 1)) in
      let x = S.index row (n - 1) in
      if lt b x then begin
        lt_irrefl x;
        introduce forall (i:nat). i < n - 1 ==> ~(lt x (S.index row i))
        with introduce _ ==> _ with not_above row b x i;
        introduce forall (i:nat). i < n - 1 ==>
            is_nan (S.index row i) \/ (exists (v:F32.t). lte (S.index row i) v /\ lt v x)
        with introduce _ ==> _ with earlier_below row b x i
      end
    end

let base_scan_allowed (row:S.seq F32.t)
  : Lemma (requires S.length row == experts)
          (ensures allowed_choice row (fst (base_scan row experts)) /\
                   snd (base_scan row experts) == S.index row (fst (base_scan row experts)))
  = scan_lemma row experts
