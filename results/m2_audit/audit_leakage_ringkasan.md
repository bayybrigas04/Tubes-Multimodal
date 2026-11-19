# Ringkasan audit leakage M2 (tahap 1)

Data: train 2072 baris, val 275 baris. Seed 42. Model neural: ya (CLIP ViT-B/32 dan multilingual-v1).

## Ambang

| Pemeriksaan | Tinjau | Pasti |
|---|---|---|
| Caption persis | - | normalisasi sama (EN atau ID) |
| Caption mirip (TF-IDF karakter) | >= 0.85 | bersama gambar mirip |
| Caption mirip (makna) | >= 0.98 | bersama gambar mirip |
| Gambar (hash dHash/pHash, hamming) | <= 8 | <= 2 |
| Gambar (cosine CLIP) | >= 0.95 | >= 0.98 |

Aturan: pasangan **pasti** bila caption persis sama, atau gambar sangat mirip, atau caption mirip dan gambar mirip sekaligus. Pasangan **tinjau** ditandai tetapi tidak otomatis dikeluarkan.

## Temuan

- Caption persis sama: 12 baris val (M1: 12).
- Baris val berstatus pasti: 13; tinjau: 24; bersih: 238.
- Baris train dikeluarkan (pasti): 21; tinjau: 38.
- Pasangan val-train dengan hamming gambar <= 8: 2.
- Pasangan dengan cosine gambar >= 0.95: 6.
- Baris val dengan tetangga ID (selisih <= 2) di train: 260 dari 275; ID rapat sehingga hanya bukti tambahan.
- Baris dengan karakter tab pada caption (dibersihkan): 2.
- Dev split: 5 fold, 2018 grup di train (terbesar 6).

## File

`leakage_filter.csv`, `caption_group.csv`, `dev_split.csv`, `pasangan_mirip.csv`.

## Cara dipakai peran 3

- Train: buang baris dengan `keluarkan_dari_train = True`.
- Val bersih: pakai baris dengan `val_bocor = False` (angka val penuh tetap dilaporkan).
- Memilih konfigurasi: gunakan `fold` di `dev_split.csv`, bukan val.
- Batch kontrastif: baris dengan `caption_group_id` sama jangan dianggap negatif satu sama lain.
