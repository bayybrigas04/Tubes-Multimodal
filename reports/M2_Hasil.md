# Hasil M2 — pencarian gambar–teks budaya Indonesia

**Kelompok 03 · IF25-40304 Pembelajaran Mesin Multimodal**
**Status per 19 November 2026:** evaluasi E1 dan E2 tanpa proyeksi serta ablasi E1+proyeksi linear/MLP pada *validation set* sudah dijalankan. Belum ada evaluasi *test* privat. Berkas ini mencatat hasil terukur dan batas auditnya; bukan revisi hasil M1.

## 1. Tujuan dan protokol

M2 membandingkan encoder pra-latih untuk *retrieval* gambar–teks Indonesia dan Inggris terhadap baseline M1 (B0), kemudian menguji apakah proyeksi yang dilatih pada embedding E1 meningkatkan hasil. Matriks yang direncanakan adalah **B0, E1, E2, E1+P-L (linear), dan E1+P-M (MLP)**; E2+proyeksi tidak termasuk matriks. Rancangan ada di [M2_Rencana_Eksperimen_T03.md](M2_Rencana_Eksperimen_T03.md).

- **Data:** `train.csv` berisi 2.072 pasangan, 2.021 pasangan unik digunakan untuk optimasi proyeksi; `val.csv` berisi 275 pasangan. Keempat arah menggunakan galeri validasi dan urutan pasangan yang sama. Train hanya untuk proyeksi, val untuk evaluasi; *test* privat tidak diakses.
- **B0:** tabel M1 di [`../results/m1/tabel_recall_val.csv`](../results/m1/tabel_recall_val.csv) dibaca tanpa mengubah atau melatih ulang M1. Angka B0 pada CSV telah dibulatkan empat desimal.
- **E1:** `google/siglip2-base-patch16-224`, revisi `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` (Apache-2.0). **E2:** `visheratin/nllb-clip-base-siglip`, revisi `a7c4a9bf24872c73a2c059ab2f9e9123a4d4144e` (CC BY-NC 4.0; penggunaan nonkomersial). Encoder dijalankan pada CPU, dalam mode evaluasi, dengan parameter dibekukan.
- **Ablasi:** proyektor yang sama untuk embedding gambar dan teks, dilatih pada train unik dengan *loss* kontrastif simetris. Konfigurasi: 5 epoch, *learning rate* 0,001, batch 4, seed 42. Linear diinisialisasi identitas; MLP residual memulai dari fungsi identitas. Encoder E1 tetap beku. Hasil proyeksi dibandingkan dengan **E1 pada run proyeksi yang sama**.
- **Metrik:** Recall@1/5/10 dan mAP@10; satu pasangan baris dianggap relevan untuk setiap kueri. Nilai tabel di bawah berbentuk persen dan dibulatkan dua desimal. Selisih yang disebut sebagai *poin persentase* dihitung dari Recall@1 sebelum pembulatan tampilan.

## 2. Matriks hasil validasi

| Varian | Arah | R@1 (%) | R@5 (%) | R@10 (%) | mAP@10 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 (M1) | Gambar → teks ID | 21,45 | 37,45 | 50,55 | 29,30 |
| B0 (M1) | Gambar → teks EN | 44,73 | 71,27 | 83,27 | 55,94 |
| B0 (M1) | Teks ID → gambar | 19,64 | 41,82 | 50,55 | 28,30 |
| B0 (M1) | Teks EN → gambar | 38,18 | 67,27 | 77,09 | 50,73 |
| E1 | Gambar → teks ID | 35,27 | 62,55 | 72,73 | 46,88 |
| E1 | Gambar → teks EN | 41,09 | 71,64 | 81,45 | 54,55 |
| E1 | Teks ID → gambar | 32,00 | 57,82 | 68,00 | 43,02 |
| E1 | Teks EN → gambar | 41,82 | 70,91 | 78,55 | 53,18 |
| E2 | Gambar → teks ID | 22,18 | 43,27 | 50,55 | 31,01 |
| E2 | Gambar → teks EN | 29,45 | 52,00 | 65,09 | 39,91 |
| E2 | Teks ID → gambar | 19,64 | 36,36 | 45,82 | 26,42 |
| E2 | Teks EN → gambar | 28,00 | 54,91 | 66,18 | 39,19 |
| E1+P-L | Gambar → teks ID | 30,55 | 61,82 | 72,36 | 43,31 |
| E1+P-L | Gambar → teks EN | 34,18 | 64,00 | 79,64 | 47,42 |
| E1+P-L | Teks ID → gambar | 26,55 | 53,09 | 66,18 | 38,31 |
| E1+P-L | Teks EN → gambar | 33,09 | 64,73 | 76,36 | 46,54 |
| E1+P-M | Gambar → teks ID | 25,09 | 55,27 | 69,45 | 37,66 |
| E1+P-M | Gambar → teks EN | 28,73 | 61,82 | 74,55 | 42,32 |
| E1+P-M | Teks ID → gambar | 25,09 | 53,82 | 65,09 | 37,95 |
| E1+P-M | Teks EN → gambar | 30,18 | 62,18 | 71,64 | 43,26 |

### Perbandingan Recall@1 dan *language gap*

*Language gap* adalah Recall@1 EN dikurangi Recall@1 ID pada arah pencarian yang sama. Gap lebih kecil **tidak otomatis** berarti hasil bahasa Indonesia membaik.

| Varian | Gambar → teks ID (%) | Teks ID → gambar (%) | Gap gambar → teks (poin) | Gap teks → gambar (poin) |
| --- | ---: | ---: | ---: | ---: |
| B0 | 21,45 | 19,64 | 23,27¹ | 18,55¹ |
| E1 | 35,27 | 32,00 | 5,82 | 9,82 |
| E2 | 22,18 | 19,64 | 7,27 | 8,36 |
| E1+P-L | 30,55 | 26,55 | 3,64 | 6,55 |
| E1+P-M | 25,09 | 25,09 | 3,64 | 5,09 |

¹ Gap B0 presisi berasal dari [`../results/m1/language_gap.json`](../results/m1/language_gap.json), yaitu 0,232727... dan 0,185455...; gap dalam CSV M2 yang dihitung dari tabel B0 empat desimal dapat sedikit berbeda (0,2328 dan 0,1854).

- **E1 vs B0:** Recall@1 Indonesia naik **13,82 poin** (gambar → teks) dan **12,36 poin** (teks → gambar). Inggris pada gambar → teks turun **3,64 poin**, sedangkan teks → gambar naik **3,64 poin**. Jadi sebagian penyempitan gap E1 berasal dari turunnya skor Inggris.
- **E2 vs B0:** Indonesia naik **0,73 poin** pada gambar → teks dan praktis tidak berubah pada teks → gambar; Inggris turun **15,28** dan **10,18 poin**. E2 memperkecil gap terutama karena Inggris turun, bukan karena peningkatan besar bahasa Indonesia.
- **Proyeksi vs E1 pada run yang sama:** linear menurunkan Recall@1 Indonesia **4,73** dan **5,45 poin** pada dua arah; MLP menurunkannya **10,18** dan **6,91 poin**. Skor Inggris juga turun pada kedua proyeksi. Untuk konfigurasi dan validasi ini, E1 tanpa proyeksi tetap paling baik pada kedua arah bahasa Indonesia.

## 3. Biaya komputasi yang tercatat

| Run/komponen | Waktu yang dicatat | Parameter encoder beku | Parameter proyektor dilatih |
| --- | ---: | ---: | ---: |
| E1 tanpa proyeksi, tahap model + embedding val | 564,31 detik | 375.187.970 | 0 |
| E2 tanpa proyeksi, tahap model + embedding val | 1.091,16 detik | 507.474.178 | 0 |
| Run proyeksi: tahap model + embedding val E1 | 433,81 detik | 375.187.970 | 0 |
| Run proyeksi: tahap model + embedding train E1 | 3.163,27 detik | 375.187.970 | 0 |
| Latihan E1+P-L | 36,48 detik | 375.187.970 | 589.824 |
| Latihan E1+P-M | 22,81 detik | 375.187.970 | 197.504 |

Embedding val dan train dalam run proyeksi **dipakai bersama**, bukan dikerjakan terpisah sekali untuk linear dan sekali untuk MLP; jangan menjumlahkannya dua kali. Waktu `embedding_val_s` mencakup tahap pemuatan model/unduhan bila diperlukan, bukan waktu inferensi murni. Run terpisah mengalami kondisi cache/lingkungan yang mungkin berbeda, sehingga selisih waktu E1 antar-run bukan pengukuran *speedup* terkontrol. Waktu retrieval, memori puncak, dan total waktu dinding end-to-end tidak tersedia sebagai metrik tersendiri.

## 4. Jejak run dan perbaikan

| Run | Lokasi hasil di Drive | Status dan sumber verifikasi |
| --- | --- | --- |
| E1 | `m2-runs/e1-val-20261118T152523Z/` | Selesai; `m2_val.csv`, `costs.json`, `manifest.json` tersedia pada salinan lokal `../m2-runs/m2-runs/`. |
| E2 | `m2-runs/e2-val-20261118T155446Z/` | Selesai; tiga berkas hasil tersedia pada salinan lokal yang sama. |
| Proyeksi awal | `m2-runs/e1-projection-20261118T162823994583Z/` | Gagal (`exit code 1`): MLP tanpa residual menghasilkan embedding bernorm nol. **Jangan gunakan sebagai hasil evaluasi atau menghapus bukti kegagalannya.** |
| Proyeksi setelah perbaikan | `m2-runs/e1-projection-20261118T174237737431Z/` | `exit code 0`; `m2_val.csv`, `costs.json`, dan `manifest.json` kini tersedia pada salinan lokal dan cocok dengan output Colab Mak. Keempat varian B0, E1, E1+P-L, E1+P-M tercatat; manifest `completed_models` berisi ketiga varian E1. Log run juga tersimpan. |

Perbaikan `src/M2_Encoders.py` menambahkan jalur residual MLP sehingga output awal mempertahankan embedding E1 dan tidak nol. Commit `bd6e62f` (`update`) sudah di-push ke branch `eksperimen-m2`, hanya berisi berkas tersebut; branch `main` tidak diubah. Uji lokal proyektor dan tes yang tersedia lulus **5/5** sebelum run ulang; itu bukan pengganti audit hasil Colab. Perubahan lokal lain pada notebook, panduan, dan berkas hasil tidak termasuk commit tersebut.

## 5. Batas interpretasi dan pekerjaan yang belum dilakukan

- Kesimpulan berlaku untuk 275 pasangan val dan konfigurasi ini; belum ada ulangan lintas seed, interval kepercayaan/paired bootstrap, atau uji signifikansi. Tidak ada dasar untuk mengklaim semua proyeksi selalu menurunkan performa.
- Kueri dianggap hanya memiliki satu pasangan benar berdasarkan baris dataset; caption duplikat/semantik serupa dapat membuat relevansi aktual berbeda. Analisis sensitivitas irisan caption train–val belum dijalankan untuk M2.
- Belum ada peringkat/galat per kueri yang disimpan untuk audit kegagalan terperinci. *Test set* privat tidak disentuh dan tidak ada klaim generalisasi ke sana.
- Ketiga folder hasil sukses E1, E2, dan proyeksi sudah tersedia pada `../m2-runs/m2-runs/` secara terpisah. Hasil awal yang gagal tidak digunakan sebagai metrik dan tidak perlu dijalankan ulang.
- Pemakaian model E2 tunduk pada lisensi CC BY-NC 4.0. Atribusi dataset mengikuti [`../ATTRIBUTION.md`](../ATTRIBUTION.md).
