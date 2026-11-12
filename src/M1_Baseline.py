import argparse, os, re, json, random
import numpy as np, pandas as pd
from PIL import Image

SEED = 42
KS = (1, 5, 10)
MAP_K = 10
ARAH = ["gambar->teks-ID", "gambar->teks-EN", "teks-ID->gambar", "teks-EN->gambar"]


def norm(s):
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def ranks_of_truth(A, B):
    """Peringkat (1 = terbaik) pasangan benar A[i]<->B[i] di antara semua B."""
    sim = A @ B.T
    diag = np.diag(sim)[:, None]
    return (sim > diag).sum(1) + 1


def metrik(A, B):
    """Recall@K dan mAP@K; satu item relevan per query, jadi AP@K = 1/rank jika rank<=K."""
    r = ranks_of_truth(A, B)
    out = {f"Recall@{k}": float((r <= k).mean()) for k in KS}
    out[f"mAP@{MAP_K}"] = float(np.where(r <= MAP_K, 1.0 / r, 0.0).mean())
    return out, r


def muat_gambar(df, root):
    imgs = []
    for f in df["file"]:
        for p in (f, os.path.join(root, f), os.path.join(root, "images", os.path.basename(f))):
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB")); break
        else:
            raise FileNotFoundError(f)
    return imgs


def embed(df, root, cache):
    paths = {k: os.path.join(cache, f"{k}.npy") for k in ("img", "ti", "te")}
    if all(os.path.exists(p) for p in paths.values()):
        return {k: np.load(p) for k, p in paths.items()}
    from sentence_transformers import SentenceTransformer
    img_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    txt_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")
    E = img_model.encode(muat_gambar(df, root), batch_size=32, normalize_embeddings=True)
    Ti = txt_model.encode(list(df["caption_id"]), batch_size=64, normalize_embeddings=True)
    Te = img_model.encode(list(df["caption_en"]), batch_size=64, normalize_embeddings=True)
    os.makedirs(cache, exist_ok=True)
    for k, v in zip(("img", "ti", "te"), (E, Ti, Te)):
        np.save(paths[k], v)
    return {"img": E, "ti": Ti, "te": Te}


def evaluasi(emb, mask=None):
    E, Ti, Te = emb["img"], emb["ti"], emb["te"]
    if mask is not None:
        E, Ti, Te = E[mask], Ti[mask], Te[mask]
    pasangan = {ARAH[0]: (E, Ti), ARAH[1]: (E, Te), ARAH[2]: (Ti, E), ARAH[3]: (Te, E)}
    hasil, ranks = {}, {}
    for nama, (A, B) in pasangan.items():
        hasil[nama], ranks[nama] = metrik(A, B)
    return pd.DataFrame(hasil).T, ranks


def contoh_query(df, emb, ranks_teks_id_ke_gambar, n=10, min_gagal=2, topk=3):
    sim = emb["ti"] @ emb["img"].T
    top = np.argsort(-sim, axis=1)[:, :topk]
    gagal = [i for i in range(len(df)) if top[i, 0] != i]
    rng = random.Random(SEED)
    pilih = rng.sample(gagal, min(min_gagal, len(gagal)))
    sisa = [i for i in range(len(df)) if i not in pilih]
    pilih += rng.sample(sisa, n - len(pilih))
    rng.shuffle(pilih)
    baris = []
    for i in pilih:
        baris.append({
            "query_id": df["caption_id"].iloc[i],
            "benar": os.path.basename(df["file"].iloc[i]),
            "top1": os.path.basename(df["file"].iloc[top[i, 0]]),
            "top2": os.path.basename(df["file"].iloc[top[i, 1]]),
            "top3": os.path.basename(df["file"].iloc[top[i, 2]]),
            "top1_benar": bool(top[i, 0] == i),
            "peringkat_benar": int(ranks_teks_id_ke_gambar[i]),
            "kategori": df["category"].iloc[i] if "category" in df else "",
        })
    return pd.DataFrame(baris)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="hasil_m1")
    a = ap.parse_args()
    np.random.seed(SEED); random.seed(SEED)
    os.makedirs(a.out, exist_ok=True)

    val = pd.read_csv(os.path.join(a.data, "val.csv"))
    train = pd.read_csv(os.path.join(a.data, "train.csv"))
    N = len(val)
    emb = embed(val, a.data, os.path.join(a.out, "cache_val"))

    tabel, ranks = evaluasi(emb)
    tabel.to_csv(os.path.join(a.out, "tabel_recall_val.csv"), float_format="%.4f")
    print(f"N val = {N}; pembanding acak Recall@1 = {1/N:.4f}")
    print(tabel.round(4).to_string())

    tr = set(norm(c) for c in train["caption_id"]) | set(norm(c) for c in train["caption_en"])
    bocor = np.array([(norm(i) in tr) or (norm(e) in tr)
                      for i, e in zip(val["caption_id"], val["caption_en"])])
    print(f"\nBaris val dengan caption yang juga ada di train: {int(bocor.sum())}")
    if bocor.sum():
        t2, _ = evaluasi(emb, mask=~bocor)
        t2.to_csv(os.path.join(a.out, "tabel_recall_val_tanpa_overlap.csv"), float_format="%.4f")
        print(t2.round(4).to_string())

    gap = {"gambar->teks": tabel.loc[ARAH[1], "Recall@1"] - tabel.loc[ARAH[0], "Recall@1"],
           "teks->gambar": tabel.loc[ARAH[3], "Recall@1"] - tabel.loc[ARAH[2], "Recall@1"]}
    print("\nLanguage gap Recall@1 (EN - ID):", {k: round(v, 4) for k, v in gap.items()})
    json.dump(gap, open(os.path.join(a.out, "language_gap.json"), "w"), indent=2)

    ex = contoh_query(val, emb, ranks[ARAH[2]])
    ex.to_csv(os.path.join(a.out, "contoh_query_id.csv"), index=False)
    print("\nSepuluh contoh query Indonesia:")
    print(ex.to_string(index=False))


if __name__ == "__main__":
    main()