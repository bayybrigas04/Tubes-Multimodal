"""
M2 | Audit leakage train -> val (tahap 1, tidak bergantung pada varian eksperimen).

Pemeriksaan
  A. caption persis sama (definisi sama dengan M1)
  B. caption mirip: TF-IDF karakter n-gram + kemiripan makna (embedding)
  C. gambar near-duplicate: dHash/pHash + cosine embedding gambar CLIP
  D. ID gambar bersebelahan (bukti tambahan, bukan kriteria penyaring)

Keluaran (default: <Tubes>/results/m2_audit/)
  leakage_filter.csv          satu baris per gambar: status, alasan, keluarkan_dari_train, val_bocor
  caption_group.csv           caption_group_id (caption sama = grup sama) untuk mask batch
  dev_split.csv               fold dev dari train (grouped) agar pemilihan konfigurasi tidak memakai val
  pasangan_mirip.csv          pasangan val-train dengan semua skor, untuk ditinjau manual
  audit_leakage_ringkasan.md  ringkasan temuan untuk laporan
  cache/                      embedding dan hash (agar run berikutnya cepat)

Hanya CPU. Tidak ada pelatihan. Val tidak diubah; hanya ditandai.

Contoh
  python M2/audit_leakage.py
  python M2/audit_leakage.py --skip-embedding          # tanpa model (hanya TF-IDF dan hash)
  python M2/audit_leakage.py --tfidf-tinjau 0.80 --img-cos-tinjau 0.93
"""
import argparse
import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TIER = {"bersih": 0, "tinjau": 1, "pasti": 2}
TIER_NAME = {v: k for k, v in TIER.items()}


# ---------------------------------------------------------------- util teks
def norm(s):
    """Sama persis dengan M1."""
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def bersihkan(s):
    """Ambil bagian sebelum karakter tab (ada 2 baris train dengan caption ganda)."""
    return str(s).split("\t")[0].strip()


def judul(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- hash gambar
def hash_gambar(img):
    g = img.convert("L")
    d = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.float32)
    dh = (d[:, 1:] > d[:, :-1]).flatten()
    p = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.float32)
    c = dct(dct(p, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()
    ph = c > np.median(c[1:])
    return dh, ph


def hamming(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    return (A.shape[1] - (A @ B.T + (1 - A) @ (1 - B).T)).round().astype(np.int16)


def path_gambar(f, root):
    for p in (f, os.path.join(root, f), os.path.join(root, "images", os.path.basename(f))):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f)


def proses_gambar(df, root, cache, pakai_model, batch=32):
    """Hash selalu dihitung; embedding gambar hanya jika pakai_model."""
    os.makedirs(cache, exist_ok=True)
    f_hash = os.path.join(cache, "hash.npz")
    f_img = os.path.join(cache, "img.npy")
    ada_hash = os.path.exists(f_hash)
    ada_img = os.path.exists(f_img)
    if ada_hash and (ada_img or not pakai_model):
        print("[INFO] Memakai cache hash" + (" dan embedding gambar." if pakai_model else "."))
        z = np.load(f_hash)
        return z["dh"], z["ph"], (np.load(f_img) if pakai_model else None)

    model = None
    if pakai_model:
        from sentence_transformers import SentenceTransformer
        print("[INFO] Memuat clip-ViT-B-32 ...")
        model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")

    files = [path_gambar(f, root) for f in df["file"]]
    DH, PH, EM = [], [], []
    for i in range(0, len(files), batch):
        imgs = []
        for p in files[i:i + batch]:
            with Image.open(p) as im:
                im = im.convert("RGB")
                dh, ph = hash_gambar(im)
                DH.append(dh)
                PH.append(ph)
                imgs.append(im.copy())
        if model is not None:
            EM.append(model.encode(imgs, batch_size=batch, normalize_embeddings=True,
                                   show_progress_bar=False))
        print(f"  gambar {min(i + batch, len(files))}/{len(files)}", end="\r")
    print()
    DH, PH = np.array(DH), np.array(PH)
    np.savez(f_hash, dh=DH, ph=PH)
    emb = None
    if model is not None:
        emb = np.vstack(EM)
        np.save(f_img, emb)
    return DH, PH, emb


def proses_teks(df, cache):
    os.makedirs(cache, exist_ok=True)
    paths = {k: os.path.join(cache, f"{k}.npy") for k in ("ti", "te")}
    if all(os.path.exists(p) for p in paths.values()):
        print("[INFO] Memakai cache embedding teks.")
        return np.load(paths["ti"]), np.load(paths["te"])
    from sentence_transformers import SentenceTransformer
    print("[INFO] Memuat model teks ...")
    en_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    id_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")
    Ti = id_model.encode(list(df["cap_id"]), batch_size=64, normalize_embeddings=True)
    Te = en_model.encode(list(df["cap_en"]), batch_size=64, normalize_embeddings=True)
    np.save(paths["ti"], Ti)
    np.save(paths["te"], Te)
    return Ti, Te


# ---------------------------------------------------------------- union-find
class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        ra, rb = self.f(a), self.f(b)
        if ra != rb:
            self.p[rb] = ra


def kelompok_dari_kunci(keys_list, n):
    """Gabungkan indeks yang berbagi salah satu kunci (mis. norm EN atau norm ID)."""
    uf = UF(n)
    for keys in keys_list:
        first = {}
        for i, k in enumerate(keys):
            if k in first:
                uf.u(first[k], i)
            else:
                first[k] = i
    return np.array([uf.f(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser(description="Audit leakage M2 (tahap 1).")
    ap.add_argument("--data", default=str(ROOT / "bundle-mahasiswa" / "bundle-mahasiswa"))
    ap.add_argument("--out", default=str(ROOT / "results" / "m2_audit"))
    ap.add_argument("--skip-embedding", action="store_true",
                    help="Lewati model neural: hanya TF-IDF dan hash gambar.")
    # ambang "tinjau" (ditandai, tidak otomatis dikeluarkan)
    ap.add_argument("--tfidf-tinjau", type=float, default=0.85)
    ap.add_argument("--sem-tinjau", type=float, default=0.98,
                    help="Skor makna caption jenuh (caption makanan sudah >0.95), jadi ambang dibuat tinggi.")
    ap.add_argument("--hash-tinjau", type=int, default=8)
    ap.add_argument("--img-cos-tinjau", type=float, default=0.95)
    # ambang "pasti" untuk gambar (caption: pasti = persis sama)
    ap.add_argument("--hash-pasti", type=int, default=2)
    ap.add_argument("--img-cos-pasti", type=float, default=0.98)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--id-gap", type=int, default=2, help="Selisih ID gambar dianggap bersebelahan.")
    args = ap.parse_args()

    pakai_model = not args.skip_embedding
    data = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    cache = out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    random.seed(SEED)

    judul("1. MEMUAT DATA")
    tr = pd.read_csv(data / "train.csv")
    va = pd.read_csv(data / "val.csv")
    tr["split"], va["split"] = "train", "val"
    df = pd.concat([tr, va], ignore_index=True)
    df["cap_en"] = df["caption_en"].map(bersihkan)
    df["cap_id"] = df["caption_id"].map(bersihkan)
    n_tab = int(((df["caption_en"] != df["cap_en"]) | (df["caption_id"] != df["cap_id"])).sum())
    df["n_en"] = df["cap_en"].map(norm)
    df["n_id"] = df["cap_id"].map(norm)
    df["num"] = df["image_id"].str.replace("svl_", "").astype(int)
    itr = np.where(df["split"] == "train")[0]
    iva = np.where(df["split"] == "val")[0]
    Ntr, Nva = len(itr), len(iva)
    print(f"train={Ntr}  val={Nva}  baris dengan karakter tab dibersihkan={n_tab}")

    # ---------------------------------------------------------- A. persis sama
    judul("2. PEMERIKSAAN A | CAPTION PERSIS SAMA")
    ve, vi = df.loc[iva, "n_en"].values[:, None], df.loc[iva, "n_id"].values[:, None]
    te, ti = df.loc[itr, "n_en"].values[None, :], df.loc[itr, "n_id"].values[None, :]
    exact = (ve == te) | (vi == ti) | (ve == ti) | (vi == te)
    # versi M1 (pada caption mentah, tanpa pembersihan tab) sebagai pemeriksaan konsistensi
    tr_raw = set(norm(c) for c in tr["caption_id"]) | set(norm(c) for c in tr["caption_en"])
    m1_bocor = np.array([(norm(i) in tr_raw) or (norm(e) in tr_raw)
                         for i, e in zip(va["caption_id"], va["caption_en"])])
    print(f"Baris val dengan caption persis sama di train: {int(exact.any(1).sum())}")
    print(f"  (definisi M1 pada caption mentah: {int(m1_bocor.sum())}; harus 12)")

    # ---------------------------------------------------------- B. caption mirip
    judul("3. PEMERIKSAAN B | CAPTION MIRIP")
    tf = {}
    for lang in ("en", "id"):
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
        X = v.fit_transform(df[f"n_{lang}"])
        tf[lang] = (X[iva] @ X[itr].T).toarray().astype(np.float32)
        print(f"TF-IDF {lang.upper()}: skor maks per baris val, median={np.median(tf[lang].max(1)):.3f}")

    if pakai_model:
        Ti_all, Te_all = proses_teks(df, str(cache))
        sem_id = Ti_all[iva] @ Ti_all[itr].T
        sem_en = Te_all[iva] @ Te_all[itr].T
        print(f"Semantik: median skor maks ID={np.median(sem_id.max(1)):.3f} EN={np.median(sem_en.max(1)):.3f}")
    else:
        Ti_all = Te_all = None
        sem_id = sem_en = np.zeros((Nva, Ntr), dtype=np.float32)

    # ---------------------------------------------------------- C. gambar mirip
    judul("4. PEMERIKSAAN C | GAMBAR NEAR-DUPLICATE")
    DH, PH, EM = proses_gambar(df, str(data), str(cache), pakai_model)
    ham_d = hamming(DH[iva], DH[itr])
    ham_p = hamming(PH[iva], PH[itr])
    ham = np.minimum(ham_d, ham_p)
    cos_img = (EM[iva] @ EM[itr].T) if pakai_model else np.zeros((Nva, Ntr), dtype=np.float32)
    print(f"Hamming minimum (dHash/pHash) terkecil: {int(ham.min())}")
    print(f"Pasangan val-train dengan hamming<={args.hash_pasti}: {int((ham <= args.hash_pasti).sum())}"
          f"; <= {args.hash_tinjau}: {int((ham <= args.hash_tinjau).sum())}")
    if pakai_model:
        print(f"Pasangan dengan cosine gambar>={args.img_cos_pasti}: {int((cos_img >= args.img_cos_pasti).sum())}"
              f"; >= {args.img_cos_tinjau}: {int((cos_img >= args.img_cos_tinjau).sum())}")

    # ---------------------------------------------------------- D. ID bersebelahan
    judul("5. PEMERIKSAAN D | ID GAMBAR BERSEBELAHAN (bukti tambahan)")
    selisih = np.abs(df.loc[iva, "num"].values[:, None] - df.loc[itr, "num"].values[None, :])
    adj = selisih <= args.id_gap
    print(f"Pasangan val-train dengan selisih ID <= {args.id_gap}: {int(adj.sum())} "
          f"({int(adj.any(1).sum())} dari {Nva} baris val punya tetangga ID di train)")
    print("ID rapat di seluruh dataset, jadi ini hanya dipakai sebagai bukti tambahan.")

    # ---------------------------------------------------------- tingkat per pasangan
    cap_skor_tf = np.maximum(tf["en"], tf["id"])
    cap_skor_sem = np.maximum(sem_en, sem_id)
    cap_tinjau = (cap_skor_tf >= args.tfidf_tinjau) | (pakai_model & (cap_skor_sem >= args.sem_tinjau))
    img_pasti = (ham <= args.hash_pasti) | (pakai_model & (cos_img >= args.img_cos_pasti))
    img_tinjau = (ham <= args.hash_tinjau) | (pakai_model & (cos_img >= args.img_cos_tinjau))

    tier = np.zeros((Nva, Ntr), dtype=np.int8)
    tier[cap_tinjau | img_tinjau] = TIER["tinjau"]
    tier[exact | img_pasti | (cap_tinjau & img_tinjau)] = TIER["pasti"]

    # pasangan untuk ditinjau: tier>=tinjau atau berada pada pita longgar di sekitar ambang
    longgar = ((cap_skor_tf >= 0.65) | (pakai_model & (cap_skor_sem >= 0.90))
               | (ham <= 12) | (pakai_model & (cos_img >= 0.90)))
    kandidat = np.argwhere((tier > 0) | longgar)
    rows = []
    for a, b in kandidat:
        ia, ib = iva[a], itr[b]
        alasan = []
        if exact[a, b]:
            alasan.append("caption persis sama")
        if cap_skor_tf[a, b] >= args.tfidf_tinjau:
            alasan.append(f"caption mirip (tfidf {cap_skor_tf[a, b]:.2f})")
        if pakai_model and cap_skor_sem[a, b] >= args.sem_tinjau:
            alasan.append(f"caption mirip (makna {cap_skor_sem[a, b]:.2f})")
        if ham[a, b] <= args.hash_tinjau:
            alasan.append(f"gambar mirip (hash {int(ham[a, b])})")
        if pakai_model and cos_img[a, b] >= args.img_cos_tinjau:
            alasan.append(f"gambar mirip (cos {cos_img[a, b]:.2f})")
        rows.append({
            "val_image": df.at[ia, "image_id"], "train_image": df.at[ib, "image_id"],
            "tingkat": TIER_NAME[int(tier[a, b])], "alasan": "; ".join(alasan),
            "persis": bool(exact[a, b]),
            "tfidf_en": round(float(tf["en"][a, b]), 3), "tfidf_id": round(float(tf["id"][a, b]), 3),
            "sem_en": round(float(sem_en[a, b]), 3) if pakai_model else np.nan,
            "sem_id": round(float(sem_id[a, b]), 3) if pakai_model else np.nan,
            "hamming_min": int(ham[a, b]),
            "cos_gambar": round(float(cos_img[a, b]), 3) if pakai_model else np.nan,
            "id_bersebelahan": bool(adj[a, b]),
            "kategori_val": df.at[ia, "category"], "kategori_train": df.at[ib, "category"],
            "caption_en_val": df.at[ia, "cap_en"], "caption_en_train": df.at[ib, "cap_en"],
            "caption_id_val": df.at[ia, "cap_id"], "caption_id_train": df.at[ib, "cap_id"],
        })
    pasangan = pd.DataFrame(rows)
    if len(pasangan):
        pasangan["_t"] = pasangan["tingkat"].map(TIER)
        pasangan = pasangan.sort_values(["_t", "tfidf_en", "cos_gambar"], ascending=False).drop(columns="_t")
    pasangan.to_csv(out / "pasangan_mirip.csv", index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------- status per gambar
    judul("6. STATUS PER GAMBAR DAN DAFTAR KELUAR")
    status = np.zeros(len(df), dtype=np.int8)
    alasan_baris = [""] * len(df)
    mitra = [""] * len(df)
    skor_maks = np.zeros(len(df), dtype=np.float32)
    if len(pasangan):
        idx = {k: i for i, k in enumerate(df["image_id"])}
        for _, r in pasangan[pasangan["tingkat"] != "bersih"].iterrows():
            for me, other in ((idx[r.val_image], r.train_image), (idx[r.train_image], r.val_image)):
                t = TIER[r.tingkat]
                if t > status[me]:
                    status[me] = t
                    alasan_baris[me] = r.alasan
                    mitra[me] = other
    val_pasti = status[iva] == TIER["pasti"]
    val_persis = exact.any(1)
    tr_pasti_mask = status[itr] == TIER["pasti"]

    filt = pd.DataFrame({
        "image_id": df["image_id"], "split": df["split"],
        "status": [TIER_NAME[int(s)] for s in status],
        "alasan": alasan_baris, "pasangan_di_split_lain": mitra,
        "keluarkan_dari_train": False, "val_bocor": False, "val_bocor_persis_m1": False,
    })
    filt.loc[itr, "keluarkan_dari_train"] = tr_pasti_mask
    filt.loc[iva, "val_bocor"] = val_pasti
    filt.loc[iva, "val_bocor_persis_m1"] = val_persis
    filt.to_csv(out / "leakage_filter.csv", index=False, encoding="utf-8-sig")
    print(f"Val: bersih={int((status[iva] == 0).sum())}  tinjau={int((status[iva] == 1).sum())}  pasti={int(val_pasti.sum())}")
    print(f"Train: bersih={int((status[itr] == 0).sum())}  tinjau={int((status[itr] == 1).sum())}  pasti={int(tr_pasti_mask.sum())}  -> dikeluarkan dari train")

    # ---------------------------------------------------------- caption_group
    cg_root = kelompok_dari_kunci([df["n_en"].tolist(), df["n_id"].tolist()], len(df))
    ukuran = pd.Series(cg_root).map(pd.Series(cg_root).value_counts())
    nomor = {r: f"CG{k + 1:04d}" for k, r in enumerate(pd.unique(cg_root[ukuran.values > 1]))}
    cg = pd.DataFrame({
        "image_id": df["image_id"], "split": df["split"],
        "caption_group_id": [nomor.get(r, f"S_{i}") for r, i in zip(cg_root, df["image_id"])],
        "ukuran_grup": ukuran.values,
    })
    cg.to_csv(out / "caption_group.csv", index=False, encoding="utf-8-sig")
    print(f"Grup caption berukuran >1: {len(nomor)} grup, {int((ukuran > 1).sum())} baris")

    # ---------------------------------------------------------- dev split (train saja)
    judul("7. DEV SPLIT DARI TRAIN (grouped, stratified per kategori)")
    # grup = caption sama + gambar/caption near-duplicate di dalam train
    ntr = len(itr)
    uf = UF(ntr)
    root_tr = kelompok_dari_kunci([df.loc[itr, "n_en"].tolist(), df.loc[itr, "n_id"].tolist()], ntr)
    for i, r in enumerate(root_tr):  # caption sama (EN atau ID) -> satu grup
        uf.u(int(r), i)
    htr = np.minimum(hamming(DH[itr], DH[itr]), hamming(PH[itr], PH[itr]))
    mg = htr <= args.hash_pasti
    if pakai_model:
        mg |= (EM[itr] @ EM[itr].T) >= args.img_cos_pasti
    a_idx, b_idx = np.where(np.triu(mg, 1))
    for a, b in zip(a_idx, b_idx):
        uf.u(int(a), int(b))
    grp = np.array([uf.f(i) for i in range(ntr)])
    gs = pd.Series(grp).value_counts()
    print(f"Grup di train: {len(gs)}; terbesar={int(gs.max())}; grup >1: {int((gs > 1).sum())}")
    dev = pd.DataFrame({"image_id": df.loc[itr, "image_id"].values,
                        "category": df.loc[itr, "category"].values,
                        "group_id": [f"G{g}" for g in grp],
                        "keluarkan_dari_train": tr_pasti_mask})
    sgk = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    dev["fold"] = -1
    for f, (_, te_idx) in enumerate(sgk.split(dev, dev["category"], dev["group_id"])):
        dev.loc[te_idx, "fold"] = f
    dev.to_csv(out / "dev_split.csv", index=False, encoding="utf-8-sig")
    print("Ukuran per fold:", dev["fold"].value_counts().sort_index().to_dict())

    # ---------------------------------------------------------- ringkasan
    judul("8. RINGKASAN")
    pas = pasangan[pasangan["tingkat"] != "bersih"] if len(pasangan) else pasangan
    md = []
    md.append("# Ringkasan audit leakage M2 (tahap 1)\n")
    md.append(f"Data: train {Ntr} baris, val {Nva} baris. Seed {SEED}. Model neural: "
              f"{'ya (CLIP ViT-B/32 dan multilingual-v1)' if pakai_model else 'tidak (hanya TF-IDF dan hash)'}.\n")
    md.append("## Ambang\n")
    md.append(f"| Pemeriksaan | Tinjau | Pasti |\n|---|---|---|\n"
              f"| Caption persis | - | normalisasi sama (EN atau ID) |\n"
              f"| Caption mirip (TF-IDF karakter) | >= {args.tfidf_tinjau} | bersama gambar mirip |\n"
              f"| Caption mirip (makna) | >= {args.sem_tinjau} | bersama gambar mirip |\n"
              f"| Gambar (hash dHash/pHash, hamming) | <= {args.hash_tinjau} | <= {args.hash_pasti} |\n"
              f"| Gambar (cosine CLIP) | >= {args.img_cos_tinjau} | >= {args.img_cos_pasti} |\n")
    md.append("Aturan: pasangan **pasti** bila caption persis sama, atau gambar sangat mirip, atau caption mirip dan gambar mirip sekaligus. Pasangan **tinjau** ditandai tetapi tidak otomatis dikeluarkan.\n")
    md.append("## Temuan\n")
    md.append(f"- Caption persis sama: {int(val_persis.sum())} baris val (M1: {int(m1_bocor.sum())}).")
    md.append(f"- Baris val berstatus pasti: {int(val_pasti.sum())}; tinjau: {int((status[iva] == 1).sum())}; bersih: {int((status[iva] == 0).sum())}.")
    md.append(f"- Baris train dikeluarkan (pasti): {int(tr_pasti_mask.sum())}; tinjau: {int((status[itr] == 1).sum())}.")
    md.append(f"- Pasangan val-train dengan hamming gambar <= {args.hash_tinjau}: {int((ham <= args.hash_tinjau).sum())}.")
    if pakai_model:
        md.append(f"- Pasangan dengan cosine gambar >= {args.img_cos_tinjau}: {int((cos_img >= args.img_cos_tinjau).sum())}.")
    md.append(f"- Baris val dengan tetangga ID (selisih <= {args.id_gap}) di train: {int(adj.any(1).sum())} dari {Nva}; ID rapat sehingga hanya bukti tambahan.")
    md.append(f"- Baris dengan karakter tab pada caption (dibersihkan): {n_tab}.")
    md.append(f"- Dev split: {args.folds} fold, {len(gs)} grup di train (terbesar {int(gs.max())}).\n")
    md.append("## File\n")
    md.append("`leakage_filter.csv`, `caption_group.csv`, `dev_split.csv`, `pasangan_mirip.csv`.\n")
    md.append("## Cara dipakai peran 3\n")
    md.append("- Train: buang baris dengan `keluarkan_dari_train = True`.")
    md.append("- Val bersih: pakai baris dengan `val_bocor = False` (angka val penuh tetap dilaporkan).")
    md.append("- Memilih konfigurasi: gunakan `fold` di `dev_split.csv`, bukan val.")
    md.append("- Batch kontrastif: baris dengan `caption_group_id` sama jangan dianggap negatif satu sama lain.\n")
    (out / "audit_leakage_ringkasan.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSemua hasil tersimpan di: {out}")


if __name__ == "__main__":
    main()
