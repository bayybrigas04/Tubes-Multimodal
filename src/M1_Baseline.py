import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SEED = 42
KS = (1, 5, 10)
MAP_K = 10
ARAH = [
    "gambar->teks-ID",
    "gambar->teks-EN",
    "teks-ID->gambar",
    "teks-EN->gambar",
]


def norm(s):
    return re.sub(r"[^\w\s]", "", str(s).lower()).strip()


def ranks_of_truth(A, B):
    sim = A @ B.T
    diag = np.diag(sim)[:, None]
    return (sim > diag).sum(1) + 1


def metrik(A, B):
    r = ranks_of_truth(A, B)
    out = {f"Recall@{k}": float((r <= k).mean()) for k in KS}
    out[f"mAP@{MAP_K}"] = float(np.where(r <= MAP_K, 1.0 / r, 0.0).mean())
    return out, r


def muat_gambar(df, root):
    imgs = []
    for f in df["file"]:
        for p in (
            f,
            os.path.join(root, f),
            os.path.join(root, "images", os.path.basename(f)),
        ):
            if os.path.exists(p):
                with Image.open(p) as image:
                    imgs.append(image.convert("RGB"))
                break
        else:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {f}")
    return imgs


def embed(df, root, cache):
    paths = {k: os.path.join(cache, f"{k}.npy") for k in ("img", "ti", "te")}
    if all(os.path.exists(p) for p in paths.values()):
        print("[INFO] Menggunakan cache embedding yang tersedia.")
        return {k: np.load(p) for k, p in paths.items()}

    from sentence_transformers import SentenceTransformer

    print("[INFO] Memuat encoder CLIP...")
    img_model = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    txt_model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )

    print("[INFO] Menghitung embedding gambar...")
    E = img_model.encode(
        muat_gambar(df, root),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print("[INFO] Menghitung embedding caption Indonesia...")
    Ti = txt_model.encode(
        list(df["caption_id"]),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print("[INFO] Menghitung embedding caption Inggris...")
    Te = img_model.encode(
        list(df["caption_en"]),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    os.makedirs(cache, exist_ok=True)
    for k, v in zip(("img", "ti", "te"), (E, Ti, Te)):
        np.save(paths[k], v)

    return {"img": E, "ti": Ti, "te": Te}


def evaluasi(emb, mask=None):
    E, Ti, Te = emb["img"], emb["ti"], emb["te"]
    if mask is not None:
        E, Ti, Te = E[mask], Ti[mask], Te[mask]

    pasangan = {
        ARAH[0]: (E, Ti),
        ARAH[1]: (E, Te),
        ARAH[2]: (Ti, E),
        ARAH[3]: (Te, E),
    }
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
    pilih += rng.sample(sisa, min(n - len(pilih), len(sisa)))
    rng.shuffle(pilih)

    baris = []
    for i in pilih:
        baris.append(
            {
                "query_id": df["caption_id"].iloc[i],
                "benar": os.path.basename(df["file"].iloc[i]),
                "top1": os.path.basename(df["file"].iloc[top[i, 0]]),
                "top2": os.path.basename(df["file"].iloc[top[i, 1]]),
                "top3": os.path.basename(df["file"].iloc[top[i, 2]]),
                "top1_benar": bool(top[i, 0] == i),
                "peringkat_benar": int(ranks_teks_id_ke_gambar[i]),
                "kategori": df["category"].iloc[i] if "category" in df else "",
            }
        )
    return pd.DataFrame(baris)


def judul(teks):
    print("\n" + "=" * 78)
    print(teks)
    print("=" * 78)


def tampilkan_tabel(df, max_colwidth=60):
    with pd.option_context(
        "display.max_colwidth", max_colwidth,
        "display.width", 180,
        "display.max_columns", None,
    ):
        print(df.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Baseline M1 pencarian lintas modal SEA-VL Indonesia."
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Folder yang berisi train.csv, val.csv, dan folder images (default: data).",
    )
    parser.add_argument(
        "--out",
        default="results/m1",
        help="Folder penyimpanan tabel, contoh query, dan cache (default: results/m1).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    np.random.seed(SEED)
    random.seed(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    judul("M1 | BASELINE PENCARIAN LINTAS MODAL")
    print(f"Dataset : {data_dir}")
    print(f"Output  : {out_dir}")
    print(f"Seed    : {SEED}")
    print(f"Metrik  : Recall@{', @'.join(map(str, KS))} dan mAP@{MAP_K}")

    required_files = ["train.csv", "val.csv"]
    missing = [name for name in required_files if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"File dataset tidak ditemukan di {data_dir}: {', '.join(missing)}"
        )

    judul("1. MEMUAT DATASET")
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    required_columns = {"file", "caption_id", "caption_en"}
    missing_columns = required_columns - set(val.columns)
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan di val.csv: {sorted(missing_columns)}")
    missing_train_columns = {"caption_id", "caption_en"} - set(train.columns)
    if missing_train_columns:
        raise ValueError(
            f"Kolom wajib tidak ditemukan di train.csv: {sorted(missing_train_columns)}"
        )

    N = len(val)
    if N == 0:
        raise ValueError("val.csv tidak memiliki data.")

    print(f"Jumlah data train : {len(train):,}")
    print(f"Jumlah data val   : {N:,}")
    print(f"Pembanding acak Recall@1: {1 / N:.4f}")

    judul("2. EMBEDDING")
    emb = embed(val, str(data_dir), str(out_dir / "cache_val"))
    print("Embedding selesai.")
    print(f"Gambar       : {emb['img'].shape}")
    print(f"Caption ID   : {emb['ti'].shape}")
    print(f"Caption EN   : {emb['te'].shape}")

    judul("3. EVALUASI BASELINE | RECALL@K DAN MAP@K")
    tabel, ranks = evaluasi(emb)
    tabel.to_csv(out_dir / "tabel_recall_val.csv", float_format="%.4f")
    tampilkan_tabel(tabel.round(4))
    print(f"\nDisimpan: {out_dir / 'tabel_recall_val.csv'}")

    judul("4. SENSITIVITAS LEAKAGE CAPTION TRAIN-VAL")
    tr = set(norm(c) for c in train["caption_id"]) | set(
        norm(c) for c in train["caption_en"]
    )
    bocor = np.array(
        [
            (norm(i) in tr) or (norm(e) in tr)
            for i, e in zip(val["caption_id"], val["caption_en"])
        ]
    )
    jumlah_bocor = int(bocor.sum())
    print(f"Baris val dengan caption yang juga ada di train: {jumlah_bocor:,} ({jumlah_bocor / N:.2%})")

    if jumlah_bocor:
        tabel_tanpa_overlap, _ = evaluasi(emb, mask=~bocor)
        path_tanpa_overlap = out_dir / "tabel_recall_val_tanpa_overlap.csv"
        tabel_tanpa_overlap.to_csv(path_tanpa_overlap, float_format="%.4f")
        print("\nHasil evaluasi setelah baris overlap dikeluarkan:")
        tampilkan_tabel(tabel_tanpa_overlap.round(4))
        print(f"\nDisimpan: {path_tanpa_overlap}")
    else:
        print("Tidak ditemukan caption yang sama antara train dan val.")

    judul("5. LANGUAGE GAP | RECALL@1 (EN - ID)")
    gap = {
        "gambar->teks": tabel.loc[ARAH[1], "Recall@1"]
        - tabel.loc[ARAH[0], "Recall@1"],
        "teks->gambar": tabel.loc[ARAH[3], "Recall@1"]
        - tabel.loc[ARAH[2], "Recall@1"],
    }
    gap_table = pd.DataFrame(
        [
            {
                "Arah": "Gambar -> Teks",
                "Recall@1 Indonesia": tabel.loc[ARAH[0], "Recall@1"],
                "Recall@1 Inggris": tabel.loc[ARAH[1], "Recall@1"],
                "Gap (EN - ID)": gap["gambar->teks"],
            },
            {
                "Arah": "Teks -> Gambar",
                "Recall@1 Indonesia": tabel.loc[ARAH[2], "Recall@1"],
                "Recall@1 Inggris": tabel.loc[ARAH[3], "Recall@1"],
                "Gap (EN - ID)": gap["teks->gambar"],
            },
        ]
    ).set_index("Arah")
    tampilkan_tabel(gap_table.round(4))

    with open(out_dir / "language_gap.json", "w", encoding="utf-8") as file:
        json.dump(gap, file, indent=2)
    print(f"\nDisimpan: {out_dir / 'language_gap.json'}")

    judul("6. SEPULUH CONTOH QUERY INDONESIA")
    ex = contoh_query(val, emb, ranks[ARAH[2]])
    ex.to_csv(out_dir / "contoh_query_id.csv", index=False)
    tampilkan_tabel(ex, max_colwidth=55)
    jumlah_gagal = int((~ex["top1_benar"]).sum()) if not ex.empty else 0
    print(f"\nJumlah contoh gagal pada tabel: {jumlah_gagal}")
    print(f"Disimpan: {out_dir / 'contoh_query_id.csv'}")

    judul("SELESAI")
    print(f"Seluruh hasil tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()