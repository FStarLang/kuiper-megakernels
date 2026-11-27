module KPlay.Proof.OptState

(* Real encodings of the reference decoder state in the optimized buffer
   layouts, and the slot / put / write laws the composition needs. *)

open FStar.Real
open KPlay.SharedSpec
module S = FStar.Seq
module M = FStar.Math.Lemmas

#set-options "--fuel 0 --ifuel 0 --z3rlimit 50"

let divmod (q r n:nat)
  : Lemma (requires r < n) (ensures (q * n + r) / n == q /\ (q * n + r) % n == r)
  = M.lemma_div_plus r q n; M.small_div r n;
    M.lemma_mod_plus r q n; M.small_mod r n

let index_snoc (s:S.seq (S.seq real)) (x:S.seq real) (i:nat)
  : Lemma (requires i <= S.length s)
          (ensures S.index (S.snoc s x) i == (if i < S.length s then S.index s i else x))
  = if i < S.length s then S.lemma_index_app1 s (S.create 1 x) i
    else S.lemma_index_app2 s (S.create 1 x) i

let sget_snoc (s:S.seq (S.seq real)) (x:S.seq real) (i:int)
  : Lemma (sget (S.snoc s x) i == (if 0 <= i && i < S.length s then S.index s i
                                   else if i = S.length s then x else S.empty))
  = if 0 <= i && i <= S.length s then index_snoc s x i

(*** Conv and recurrent slots *)

let shaped (convs recs:S.seq (S.seq real)) : prop =
  S.length convs == 18 /\ S.length recs == 18 /\
  (forall (d:nat). d < 18 ==> S.length (S.index convs d) == 24576 /\ S.length (S.index recs d) == 262144)

[@@"opaque_to_smt"]
let enc_conv (convs:S.seq (S.seq real)) : S.seq real =
  S.init 442368 (fun (i:nat{i < 442368}) -> at (sget convs (i / 24576)) (i % 24576))

[@@"opaque_to_smt"]
let enc_rec (recs:S.seq (S.seq real)) : S.seq real =
  S.init 4718592 (fun (i:nat{i < 4718592}) -> at (sget recs (i / 262144)) (rec_perm (i % 262144)))

let conv_slot (convs recs:S.seq (S.seq real)) (d:nat)
  : Lemma (requires shaped convs recs /\ d < 18)
          (ensures opt_conv_slot (enc_conv convs) d == sget convs d)
  = reveal_opaque (`%enc_conv) (enc_conv convs);
    introduce forall (j:nat). j < 24576 ==> at (enc_conv convs) (d * 24576 + j) == S.index (sget convs d) j
    with introduce _ ==> _ with divmod d j 24576;
    S.lemma_eq_intro (opt_conv_slot (enc_conv convs) d) (sget convs d)

let conv_put (convs:S.seq (S.seq real)) (d:nat) (c':S.seq real)
  : Lemma (requires S.length convs == 18 /\ d < 18)
          (ensures opt_conv_put (enc_conv convs) d c' == enc_conv (supd convs d c'))
  = reveal_opaque (`%enc_conv) (enc_conv convs);
    reveal_opaque (`%enc_conv) (enc_conv (supd convs d c'));
    S.lemma_eq_intro (opt_conv_put (enc_conv convs) d c') (enc_conv (supd convs d c'))

let rec_perm_inv (idx:nat)
  : Lemma (requires idx < 262144)
          (ensures rec_perm idx < 262144 /\ rec_perm (rec_perm idx) == idx)
  = let h = idx / 16384 in
    let a = (idx % 16384) / 128 in
    let b = idx % 128 in
    M.euclidean_division_definition idx 16384;
    M.euclidean_division_definition (idx % 16384) 128;
    M.modulo_modulo_lemma idx 128 128;
    assert (idx % 128 == (idx % 16384) % 128) by (FStar.Tactics.V2.smt ());
    assert (idx == h * 16384 + a * 128 + b);
    divmod a b 128;
    divmod h (b * 128 + a) 16384;
    divmod b a 128

let rec_slot (convs recs:S.seq (S.seq real)) (d:nat)
  : Lemma (requires shaped convs recs /\ d < 18)
          (ensures opt_rec_slot (enc_rec recs) d == sget recs d)
  = reveal_opaque (`%enc_rec) (enc_rec recs);
    introduce forall (idx:nat). idx < 262144 ==> at (enc_rec recs) (d * 262144 + rec_perm idx) == S.index (sget recs d) idx
    with introduce _ ==> _ with (rec_perm_inv idx; divmod d (rec_perm idx) 262144);
    S.lemma_eq_intro (opt_rec_slot (enc_rec recs) d) (sget recs d)

let rec_put (recs:S.seq (S.seq real)) (d:nat) (s':S.seq real)
  : Lemma (requires S.length recs == 18 /\ d < 18)
          (ensures opt_rec_put (enc_rec recs) d s' == enc_rec (supd recs d s'))
  = reveal_opaque (`%enc_rec) (enc_rec recs);
    reveal_opaque (`%enc_rec) (enc_rec (supd recs d s'));
    S.lemma_eq_intro (opt_rec_put (enc_rec recs) d s') (enc_rec (supd recs d s'))

let shaped_put (convs recs:S.seq (S.seq real)) (d:nat) (c' s':S.seq real)
  : Lemma (requires shaped convs recs /\ d < 18 /\ S.length c' == 24576 /\ S.length s' == 262144)
          (ensures shaped (supd convs d c') (supd recs d s'))
  = ()

let init_enc ()
  : Lemma (shaped init_state.convs init_state.recs /\
           enc_conv init_state.convs == S.create 442368 0.0R /\
           enc_rec init_state.recs == S.create 4718592 0.0R)
  = reveal_opaque (`%enc_conv) (enc_conv init_state.convs);
    reveal_opaque (`%enc_rec) (enc_rec init_state.recs);
    S.lemma_eq_intro (enc_conv init_state.convs) (S.create 442368 0.0R);
    S.lemma_eq_intro (enc_rec init_state.recs) (S.create 4718592 0.0R)

(*** KV caches *)

(* koff = 0 encodes keys, koff = 512 values; entries of past positions come
   from kvs, the current position from cur. *)
[@@"opaque_to_smt"]
let enc_kv (koff:nat) (kvs:S.seq (S.seq real)) (cur:S.seq real) : S.seq real =
  S.init 6291456 (fun (i:nat{i < 6291456}) ->
    kv_get (S.snoc kvs cur) ((i % 524288) / 256) ((i / 1048576) * 1024 + koff + opt_kv_comp i))

let kv_idx (g p j:nat)
  : Lemma (requires g < 6 /\ p < 2048 /\ j < 512)
          (ensures opt_kv_index g p j < 6291456 /\
                   opt_kv_index g p j / 1048576 == g /\
                   (opt_kv_index g p j % 524288) / 256 == p /\
                   opt_kv_comp (opt_kv_index g p j) == j)
  = let a = j / 256 in
    let b = j % 256 in
    let i = opt_kv_index g p j in
    let r1 = a * 524288 + p * 256 + b in
    assert (i == g * 1048576 + r1);
    divmod g r1 1048576;
    divmod a (p * 256 + b) 524288;
    assert (i == (g * 2 + a) * 524288 + (p * 256 + b));
    divmod (g * 2 + a) (p * 256 + b) 524288;
    divmod p b 256;
    assert (i == (g * 4096 + a * 2048 + p) * 256 + b);
    divmod (g * 4096 + a * 2048 + p) b 256

let kv_read (koff:nat) (kvs:S.seq (S.seq real)) (cur:S.seq real) (g p j:nat)
  : Lemma (requires g < 6 /\ p < S.length kvs /\ p < 2048 /\ j < 512)
          (ensures opt_hk g (enc_kv koff kvs cur) p j == kv_get kvs p (g * 1024 + koff + j))
  = reveal_opaque (`%enc_kv) (enc_kv koff kvs cur);
    kv_idx g p j;
    sget_snoc kvs cur p

let kv_comp_bound (i:nat)
  : Lemma (requires i < 6291456)
          (ensures i / 1048576 < 6 /\ opt_kv_comp i < 512 /\ (i % 524288) / 256 < 2048)
  = M.lemma_mod_lt i 1048576; M.lemma_div_le (i % 1048576) 1048575 524288; M.lemma_div_le (i % 524288) 524287 256; M.lemma_div_le i 6291455 1048576;
    M.lemma_mod_lt i 256; M.lemma_mod_lt i 524288

let kv_cur_idx (koff g:nat) (c:nat)
  : Lemma (requires g < 6 /\ koff + c < 1024)
          (ensures (g * 1024 + koff + c) / 1024 == g /\ (g * 1024 + koff + c) % 1024 == koff + c)
  = divmod g (koff + c) 1024

let kv_write (koff:nat) (kvs:S.seq (S.seq real)) (cur:S.seq real) (g t:nat) (kn vn nw:S.seq real)
  : Lemma (requires g < 6 /\ t == S.length kvs /\
                    ((koff == 0 /\ nw == kn) \/ (koff == 512 /\ nw == vn)))
          (ensures opt_kv_write g t (enc_kv koff kvs cur) nw == enc_kv koff kvs (write_kv cur g kn vn))
  = reveal_opaque (`%enc_kv) (enc_kv koff kvs cur);
    reveal_opaque (`%enc_kv) (enc_kv koff kvs (write_kv cur g kn vn));
    let cur' = write_kv cur g kn vn in
    introduce forall (i:nat). i < 6291456 ==>
      S.index (opt_kv_write g t (enc_kv koff kvs cur) nw) i == S.index (enc_kv koff kvs cur') i
    with introduce _ ==> _ with begin
      let gg = i / 1048576 in
      let p = (i % 524288) / 256 in
      let c = opt_kv_comp i in
      kv_comp_bound i;
      kv_cur_idx koff gg c;
      sget_snoc kvs cur p;
      sget_snoc kvs cur' p
    end;
    S.lemma_eq_intro (opt_kv_write g t (enc_kv koff kvs cur) nw) (enc_kv koff kvs cur')

let kv_new (koff:nat) (kvs:S.seq (S.seq real)) (cur:S.seq real)
  : Lemma (requires koff <= 512)
          (ensures enc_kv koff (S.snoc kvs cur) (S.create 6144 0.0R) == enc_kv koff kvs cur)
  = reveal_opaque (`%enc_kv) (enc_kv koff (S.snoc kvs cur) (S.create 6144 0.0R));
    reveal_opaque (`%enc_kv) (enc_kv koff kvs cur);
    introduce forall (i:nat). i < 6291456 ==>
      S.index (enc_kv koff (S.snoc kvs cur) (S.create 6144 0.0R)) i == S.index (enc_kv koff kvs cur) i
    with introduce _ ==> _ with begin
      let p = (i % 524288) / 256 in
      kv_comp_bound i;
      sget_snoc (S.snoc kvs cur) (S.create 6144 0.0R) p;
      sget_snoc kvs cur p
    end;
    S.lemma_eq_intro (enc_kv koff (S.snoc kvs cur) (S.create 6144 0.0R)) (enc_kv koff kvs cur)

let kv_init (koff:nat)
  : Lemma (requires koff <= 512)
          (ensures enc_kv koff S.empty (S.create 6144 0.0R) == S.create 6291456 0.0R)
  = reveal_opaque (`%enc_kv) (enc_kv koff S.empty (S.create 6144 0.0R));
    introduce forall (i:nat). i < 6291456 ==>
      S.index (enc_kv koff S.empty (S.create 6144 0.0R)) i == 0.0R
    with introduce _ ==> _ with begin
      kv_comp_bound i;
      sget_snoc S.empty (S.create 6144 0.0R) ((i % 524288) / 256)
    end;
    S.lemma_eq_intro (enc_kv koff S.empty (S.create 6144 0.0R)) (S.create 6291456 0.0R)
