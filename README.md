# Tubes Multimodal — Kelompok 03 (Titut Encoder)

Proyek akhir **IF25-40304 Pembelajaran Mesin Multimodal**: pencarian lintas modal gambar ↔ teks pada data budaya Indonesia. Dataset yang digunakan adalah bundel mahasiswa SEA-VL Indonesian subset dengan caption bahasa Indonesia dan Inggris. Proyek mengikuti [TOR](Proyek_Akhir_TOR.pdf); encoder pra-latih harus tetap beku, data tambahan di luar bundel tidak digunakan, dan test set privat tidak diakses sebelum dibuka dosen.

> **Status dokumentasi:** README ini mencatat pekerjaan yang sudah terlihat pada berkas lokal dan repo, bukan klaim bahwa semua milestone sudah selesai. Kode serta angka M1 yang ada tidak diubah.

## Progres milestone

| Milestone | Ketentuan ringkas dari TOR | Status yang dapat diverifikasi dari berkas saat ini |
| --- | --- | --- |
| **M0 — Proposal dan reproduksi baseline** | Proposal, angka InfoNCE/Recall@1 dari lembar kerja, pembagian peran, dan pernyataan integritas. | **Artefak tersedia:** [`reports/M0_Proposal_T03.pdf`](reports/M0_Proposal_T03.pdf) dan [`M0_Pernyataan Integritas_T03.pdf`](M0_Pernyataan%20Integritas_T03.pdf). Proposal mencatat demonstrasi sintetis enam pasangan: InfoNCE 10,6138 → 0,1618 dan Recall@1 1/6 → 6/6. Angka ini **bukan** metrik retrieval SEA-VL. Proposal merencanakan penggantian encoder dan proyeksi kecil untuk M2. |
| **M1 — Baseline pencarian lintas modal** | Pipeline gambar–teks, Recall@1/5/10 dan mAP@K untuk empat arah pada validasi, serta 10 contoh query Indonesia termasuk kegagalan. | **Kode tersedia:** [`src/M1_Baseline.py`](src/M1_Baseline.py) dan [`notebooks/m1_baseline.ipynb`](notebooks/m1_baseline.ipynb). Output eksekusi M1 tersimpan di `Q:\Tubes_Multimodal\K03_Notebook.ipynb` **di luar repo**; notebook M1 dalam repo belum menyimpan output sel. Ringkasan angka dan contoh ada di bawah. Tidak ada alasan untuk menyunting hasil M1 hanya karena contoh gagal. |
| **M2 — Strategi penyejajaran/fusi** | Bandingkan sedikitnya dua varian dengan baseline M1 dan laporkan ablation serta biaya komputasi. | **Belum ditemukan implementasi atau hasil M2 di repo.** SigLIP 2 Base, NLLB-CLIP-SigLIP, dan proyeksi linear/MLP masih berupa rencana dalam proposal M0, bukan hasil terukur. |
| **M3 — Kasus sulit dan etika** | Uji kasus sulit, analisis keterbatasan, etika, dan atribusi. | **Belum ditemukan artefak M3 khusus.** Atribusi dataset sudah tersedia di [`ATTRIBUTION.md`](ATTRIBUTION.md). |
| **M4 — Protokol dan test privat** | Protokol evaluasi serta satu pengajuan prediksi test privat sesuai petunjuk dosen. | **Belum ditemukan artefak M4 atau test privat.** Jangan menebak/merekonstruksi test set yang belum dibuka. |

Status di atas menunjukkan **keberadaan artefak**, bukan bukti pengumpulan atau penilaian dosen.

## Data dan pembagian kerja

Bundel yang tersedia secara lokal di `Q:\Tubes_Multimodal\bundle-mahasiswa\bundle-mahasiswa\bundle-mahasiswa\` berisi `train.csv` (2.072 baris), `val.csv` (275 baris), dan 2.347 berkas dalam `images/`. CSV memuat `file`, `caption_id`, `caption_en`, `category`, `loc`, serta metadata lain. Dataset tidak disertakan dalam repo; `data/README.txt` hanya menjelaskan formatnya. Pastikan path `--data` menunjuk ke direktori yang **langsung** berisi kedua CSV dan `images/`.

Peran kelompok menurut proposal M0:

| Peran | Penanggung jawab |
| --- | --- |
| Data dan Etika Data | Ahmad Ali Mukti |
| Baseline dan Pipeline | Bayu Brigas Novaldi |
| Eksperimen Penyejajaran/Fusi | Anisah Octa Rohila |
| Evaluasi dan Uji Kasus Sulit | Aryasatya Widyatna Akbar |
| Integrasi dan Koordinasi | Jesika Filosovi Br P-A |

## M1: yang sudah dibuat

[`src/M1_Baseline.py`](src/M1_Baseline.py) membaca kedua CSV; `train.csv` digunakan untuk mendeteksi caption yang beririsan dengan validasi, **bukan untuk melatih ulang encoder M1**. Skrip memuat gambar validasi sebagai RGB dan membuat tiga kelompok embedding ternormalisasi:

- **Gambar:** `sentence-transformers/clip-ViT-B-32`.
- **Caption Indonesia:** `sentence-transformers/clip-ViT-B-32-multilingual-v1`.
- **Caption Inggris:** `sentence-transformers/clip-ViT-B-32`.

Perkalian matriks embedding yang telah dinormalisasi menghasilkan skor kesamaan kosinus. Kandidat kemudian diperingkat untuk gambar → teks Indonesia/Inggris dan teks Indonesia/Inggris → gambar. Evaluasi memakai pasangan pada baris CSV yang sama sebagai satu jawaban benar per query; mAP@10 pada implementasi ini adalah rata-rata `1/peringkat` jika jawaban benar masuk sepuluh besar, atau nol jika tidak. Seed yang ditetapkan skrip adalah **42**. Encoder pada baseline tidak dilatih.

### Angka M1 yang tersimpan di notebook lokal

Angka berikut **disalin dari output** `Q:\Tubes_Multimodal\K03_Notebook.ipynb`, bukan hasil eksekusi ulang saat README dibuat. Pembanding acak Recall@1 untuk 275 kandidat adalah `1/275 ≈ 0,0036`.

| Arah pencarian | Recall@1 | Recall@5 | Recall@10 | mAP@10 |
| --- | ---: | ---: | ---: | ---: |
| Gambar → teks Indonesia | 0,2145 | 0,3745 | 0,5055 | 0,2930 |
| Gambar → teks Inggris | 0,4473 | 0,7127 | 0,8327 | 0,5594 |
| Teks Indonesia → gambar | 0,1964 | 0,4182 | 0,5055 | 0,2830 |
| Teks Inggris → gambar | 0,3818 | 0,6727 | 0,7709 | 0,5073 |

Selisih Recall@1 (Inggris dikurangi Indonesia), dihitung dari output M1 sebelum pembulatan tabel, adalah **0,2327** untuk gambar → teks dan **0,1855** untuk teks → gambar. Ini adalah *language gap* pada split validasi yang tersedia, bukan klaim performa test privat.

### Sepuluh contoh query Indonesia → gambar

Sampel berikut mengikuti output notebook lokal. Kolom Top-1 adalah prediksi sistem, bukan gambar pasangan benar. Teks query panjang ditampilkan ringkas; teks lengkap serta Top-2/Top-3 tersedia pada output notebook. **Kegagalan tidak dihapus atau dikoreksi.**

| No. | Query (ringkas) | Gambar benar | Top-1 | Benar? | Peringkat gambar benar |
| ---: | --- | --- | --- | :---: | ---: |
| 1 | Kota Tua Jakarta / Oud Batavia | `svl_143496710.jpg` | `svl_144425430.jpg` | Tidak | 18 |
| 2 | Bakso: bola daging, kuah, mi atau sayuran | `svl_144425048.jpg` | `svl_144424955.jpg` | Tidak | 13 |
| 3 | Halaman di tengah Fort Marlborough | `svl_140084276.jpg` | `svl_140084276.jpg` | Ya | 1 |
| 4 | Bedug sebagai bukti akulturasi dan penanda waktu salat | `svl_140084842.jpg` | `svl_140084783.jpg` | Tidak | 40 |
| 5 | Kemeja batik dengan pola, warna, dan simbol | `svl_143496870.jpg` | `svl_140084373.jpg` | Tidak | 20 |
| 6 | Martabak keju dengan susu kental manis | `svl_142113138.jpg` | `svl_147562476.jpg` | Tidak | 148 |
| 7 | Dewi Shinta pada tari Kecak di Pura Uluwatu | `svl_141004111.jpg` | `svl_141004074.jpg` | Tidak | 97 |
| 8 | Buto Cakil bersama dua penari Manipuren | `svl_140084705.jpg` | `svl_140084701.jpg` | Tidak | 19 |
| 9 | Aneka gorengan dengan sambal dan serundeng | `svl_147525876.jpg` | `svl_144425061.jpg` | Tidak | 49 |
| 10 | Rempeyek atau peyek kacang | `svl_147525769.jpg` | `svl_144923822.jpg` | Tidak | 80 |

Output skrip melaporkan **12 baris validasi** yang caption Indonesia **atau** Inggrisnya juga ada di train menurut normalisasi skrip. Ini tidak otomatis sama dengan angka **10 caption lintas split** dalam proposal M0 karena cara menghitungnya berbeda. Skrip juga menghitung tabel sensitivitas setelah membuang baris yang terdeteksi; pada tabel itu ukuran **query dan galeri** sama-sama berubah, sehingga jangan menafsirkannya sebagai perubahan yang hanya menghapus query bocor. Kategori yang jarang dan caption yang sama/semantik mirip juga membatasi interpretasi satu-pasangan-benar-per-query.

## Cara menjalankan ulang M1 (opsional; tidak diperlukan untuk membaca hasil di atas)

Gunakan **Python 3.10+**, lingkungan virtual, dan dependensi pada [`requirements.txt`](requirements.txt). Model pra-latih diunduh oleh `sentence-transformers` saat belum ada di cache: diperlukan koneksi internet, ruang penyimpanan, dan waktu komputasi; eksekusi tetap harus memungkinkan pada CPU. Jangan menganggap model sudah tersedia secara lokal.

Contoh dari **PowerShell** di direktori repo (pilih lokasi output baru agar tidak menimpa hasil sebelumnya):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\M1_Baseline.py --data 'Q:\Tubes_Multimodal\bundle-mahasiswa\bundle-mahasiswa\bundle-mahasiswa' --out 'results\m1_reproduksi_baru'
```

Skrip akan menghasilkan `tabel_recall_val.csv`, `tabel_recall_val_tanpa_overlap.csv` bila ada caption beririsan, `language_gap.json`, `contoh_query_id.csv`, dan `cache_val/` di lokasi `--out`. **Perintah di atas adalah panduan, bukan klaim bahwa reproduksi baru telah dijalankan.** Notebook Colab lokal `Q:\Tubes_Multimodal\K03_Notebook.ipynb` juga memuat output M1, tetapi berkas itu belum termasuk dalam repo. Notebook `notebooks/m1_baseline.ipynb` yang ada di repo belum menyimpan output eksekusi.

## Atribusi dan batasan

Data berasal dari **SEA-VL Indonesian subset** (`KORIKA-AI/sea-vl_crowdsourcing_id`, diturunkan dari `SEACrowd/sea-vl_crowdsourcing`), lisensi **CC BY-SA 4.0**. Rujukan: Cahyawijaya dkk. (2025), *Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia*, ACL. Kewajiban lisensi dan modifikasi bundel dijelaskan di [`ATTRIBUTION.md`](ATTRIBUTION.md).

**Catatan dokumentasi:** `ATTRIBUTION.md` menyebut 2.763 sampel yang dipertahankan, sedangkan bundel mahasiswa lokal yang diperiksa berisi **2.347** gambar/baris train+val. Perbedaan ini belum dijelaskan oleh berkas yang ada; gunakan angka split yang benar-benar tersedia saat melaporkan eksperimen, tanpa mengubah atribusi sumber secara diam-diam.
