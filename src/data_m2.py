"""
Pemuat data M2 yang sudah memakai hasil audit leakage (peran 1).

File dataset asli TIDAK diubah. Baris "pasti bocor" hanya disaring saat dimuat.

Pemakaian (dari folder Tubes):
    from M2.data_m2 import muat_data
    d = muat_data()
    d["train"]      # train tanpa 21 baris pasti bocor  -> untuk melatih
    d["val"]        # val penuh 275 baris               -> angka utama, sebanding M1
    d["val_bersih"] # boolean mask (275,)               -> evaluasi(emb, mask=d["val_bersih"])
    d["dev"]        # fold dev dari train               -> memilih konfigurasi, BUKAN val

Tanpa perlu import, cukup pakai leakage_filter.csv:
    f = pd.read_csv("results/m2_audit/leakage_filter.csv")
    buang = set(f.loc[f.keluarkan_dari_train, "image_id"])
    train = train[~train.image_id.isin(buang)]
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "bundle-mahasiswa" / "bundle-mahasiswa"
AUDIT = ROOT / "results" / "m2_audit"


def bersihkan(s):
    """Dua baris train punya caption ganda dipisah tab; ambil bagian pertama."""
    return str(s).split("\t")[0].strip()


def muat_data(data_dir=DATA, audit_dir=AUDIT):
    data_dir, audit_dir = Path(data_dir), Path(audit_dir)
    train = pd.read_csv(data_dir / "train.csv")
    val = pd.read_csv(data_dir / "val.csv")
    filt = pd.read_csv(audit_dir / "leakage_filter.csv")
    dev = pd.read_csv(audit_dir / "dev_split.csv")
    grup = pd.read_csv(audit_dir / "caption_group.csv")

    for df in (train, val):
        df["caption_en"] = df["caption_en"].map(bersihkan)
        df["caption_id"] = df["caption_id"].map(bersihkan)

    buang = set(filt.loc[filt["keluarkan_dari_train"], "image_id"])
    bocor_val = set(filt.loc[filt["val_bocor"], "image_id"])

    train_bersih = train[~train["image_id"].isin(buang)].reset_index(drop=True)
    train_bersih = train_bersih.merge(dev[["image_id", "fold"]], on="image_id", how="left")
    train_bersih = train_bersih.merge(grup[["image_id", "caption_group_id"]], on="image_id", how="left")
    val = val.merge(grup[["image_id", "caption_group_id"]], on="image_id", how="left")
    val_bersih = ~val["image_id"].isin(bocor_val).to_numpy()

    return {
        "train": train_bersih,
        "val": val,
        "val_bersih": val_bersih,
        "dev": train_bersih[["image_id", "fold"]],
        "dibuang_dari_train": sorted(buang),
    }


if __name__ == "__main__":
    d = muat_data()
    print(f"train asli 2072 -> train bersih {len(d['train'])} (dibuang {len(d['dibuang_dari_train'])})")
    print(f"val penuh {len(d['val'])} | val bersih {int(d['val_bersih'].sum())}")
    print("fold dev:", d["train"]["fold"].value_counts().sort_index().to_dict())
