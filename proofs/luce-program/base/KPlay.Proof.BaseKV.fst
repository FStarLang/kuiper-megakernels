module KPlay.Proof.BaseKV

(* Real model of one base attention KV cache of capacity c, built from the
   reference KV history, and the size-generic layout arithmetic linking
   base_kv_index / base_kv_pos / base_kv_comp. *)

open FStar.Real
open KPlay.SharedSpec
open KPlay.Proof.BaseCongr
module S = FStar.Seq
module ML = FStar.Math.Lemmas

#set-options "--z3rlimit 20 --fuel 0 --ifuel 0"

(*** Layout arithmetic *)

let div_lt (x c:int) (d:pos)
  : Lemma (requires 0 <= x /\ x < c * d) (ensures 0 <= x / d /\ x / d < c)
  = ML.lemma_div_mod x d;
    if x / d >= c then ML.lemma_mult_le_right d c (x / d)

let kv_dec (c:pos) (i:nat)
  : Lemma (requires i < c * 1024)
          (ensures 0 <= base_kv_pos c i /\ base_kv_pos c i < c /\
                   0 <= base_kv_comp c i /\ base_kv_comp c i < 512 /\
                   (i < c * 512 ==> base_kv_comp c i < 512))
  = let m = i % (c * 512) in
    let m2 = m % (c * 256) in
    div_lt m2 c 256;
    div_lt m 2 (c * 256)

let kv_enc (c:pos) (p j:nat)
  : Lemma (requires p < c /\ j < 512)
          (ensures (let i = base_kv_index c p j in
                    0 <= i /\ i < c * 512 /\
                    base_kv_pos c i == p /\ base_kv_comp c i == j /\
                    base_kv_pos c (c * 512 + i) == p /\ base_kv_comp c (c * 512 + i) == j))
  = let i = base_kv_index c p j in
    let b = j % 256 in
    let u = p * 256 + b in
    assert (0 <= u /\ u < c * 256);
    ML.small_mod u (c * 256);
    ML.small_div u (c * 256);
    ML.lemma_div_plus b p 256;
    ML.small_div b 256;
    if j < 256 then begin
      assert (i == u);
      ML.lemma_mod_plus b p 256;
      ML.small_mod b 256
    end else begin
      assert (i == u + 1 * (c * 256));
      ML.lemma_mod_plus u 1 (c * 256);
      ML.lemma_div_plus u 1 (c * 256);
      ML.lemma_mod_plus b (c + p) 256;
      ML.small_mod b 256
    end;
    assert (0 <= i /\ i < c * 512);
    ML.small_mod i (c * 512);
    ML.lemma_mod_plus i 1 (c * 512);
    ML.lemma_mod_plus i (2 * c) 256

(*** Real KV cache from the reference history *)

let kv_ent (c:int) (kvs:S.seq (S.seq real)) (g:int) (i:int) : real =
  let p = base_kv_pos c i in
  let e = base_kv_comp c i in
  if p < 0 then 0.0R
  else if i < c * 512 then kv_get kvs p (g * 1024 + e) else kv_get kvs p (g * 1024 + 512 + e)

let rkv_of (c:int) (kvs:S.seq (S.seq real)) (g:int) : S.seq real =
  if c > 0 then S.init (c * 1024) (fun (i:nat{i < c * 1024}) -> kv_ent c kvs g i) else S.empty

let rkv_at (c:pos) (kvs:S.seq (S.seq real)) (g:int) (i:int)
  : Lemma (requires 0 <= i /\ i < c * 1024)
          (ensures S.length (rkv_of c kvs g) == c * 1024 /\ at (rkv_of c kvs g) i == kv_ent c kvs g i)
  = ()

let rkv_empty (c:pos) (g:int)
  : Lemma (rkv_of c S.empty g == S.create (c * 1024) 0.0R)
  = let aux (i:nat{i < c * 1024}) : Lemma (S.index (rkv_of c S.empty g) i == S.index (S.create (c * 1024) 0.0R) i) =
      rkv_at c S.empty g i
    in
    Classical.forall_intro aux;
    S.lemma_eq_intro (rkv_of c S.empty g) (S.create (c * 1024) 0.0R)

(* The base history views of rkv_of agree with the reference history of
   attention layer l (the lambdas are spelled as in run_layers). *)
let hk_agree (c:pos) (kvs:S.seq (S.seq real)) (l:nat) (t:nat)
  : Lemma (requires t <= c)
          (ensures hagree (base_hk c (rkv_of c kvs (l / 4))) (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + j)) t /\
                   hagree (base_hv c (rkv_of c kvs (l / 4))) (fun (p j:nat) -> kv_get kvs p ((l / 4) * 1024 + 512 + j)) t)
  = let g = l / 4 in
    let aux (p j:nat)
      : Lemma (p < t /\ j < 512 ==>
                 base_hk c (rkv_of c kvs g) p j == kv_get kvs p (g * 1024 + j) /\
                 base_hv c (rkv_of c kvs g) p j == kv_get kvs p (g * 1024 + 512 + j)) =
      if p < t && j < 512 then begin
        kv_enc c p j;
        let i = base_kv_index c p j in
        rkv_at c kvs g i;
        rkv_at c kvs g (c * 512 + i)
      end
    in
    Classical.forall_intro_2 aux

let snoc_get (kvs:S.seq (S.seq real)) (cur:S.seq real) (p:nat) (o:int)
  : Lemma (kv_get (S.snoc kvs cur) p o == (if p = S.length kvs then at cur o else kv_get kvs p o))
  = let n = S.length kvs in
    S.lemma_len_append kvs (S.create 1 cur);
    if p < n then S.lemma_index_app1 kvs (S.create 1 cur) p
    else if p = n then S.lemma_index_app2 kvs (S.create 1 cur) p
    else ()

(* Writing position t of rkv_of (history of length t) gives rkv_of of the
   extended history. *)
let kv_write_ok (c:pos) (t:nat) (kvs:S.seq (S.seq real)) (cur kn vn:S.seq real) (g:int)
  : Lemma (requires t < c /\ S.length kvs == t /\
                    (forall (j:nat). j < 512 ==> at cur (g * 1024 + j) == at kn j /\ at cur (g * 1024 + 512 + j) == at vn j))
          (ensures base_kv_write c t (rkv_of c kvs g) kn vn == rkv_of c (S.snoc kvs cur) g)
  = let rk = rkv_of c kvs g in
    let rk' = rkv_of c (S.snoc kvs cur) g in
    let aux (i:nat{i < c * 1024}) : Lemma (S.index (base_kv_write c t rk kn vn) i == S.index rk' i) =
      kv_dec c i;
      rkv_at c kvs g i;
      rkv_at c (S.snoc kvs cur) g i;
      let p : nat = base_kv_pos c i in
      let e = base_kv_comp c i in
      snoc_get kvs cur p (g * 1024 + e);
      snoc_get kvs cur p (g * 1024 + 512 + e);
      if i < c * 512 then () else ()
    in
    Classical.forall_intro aux;
    S.lemma_eq_intro (base_kv_write c t rk kn vn) rk'
