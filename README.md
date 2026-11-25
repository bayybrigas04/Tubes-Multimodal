# Tubes Multimodal — Kelompok 03 (Titut Encoder)

Pencarian lintas modal gambar ↔ teks pada data budaya Indonesia. Dataset yang digunakan adalah bundel mahasiswa SEA-VL Indonesian subset dengan caption bahasa Indonesia dan Inggris. Encoder pra-latih harus tetap beku, data tambahan di luar bundel tidak digunakan, dan test set privat tidak diakses sebelum dibuka dosen.

## Dataset

Data berasal dari **SEA-VL Indonesian subset** (`KORIKA-AI/sea-vl_crowdsourcing_id`, diturunkan dari `SEACrowd/sea-vl_crowdsourcing`), lisensi **CC BY-SA 4.0**. Rujukan: Cahyawijaya dkk. (2025), *Crowdsource, Crawl, or Generate? Creating SEA-VL, a Multicultural Vision-Language Dataset for Southeast Asia*, ACL. Kewajiban lisensi dan modifikasi bundel dijelaskan di [`ATTRIBUTION.md`](ATTRIBUTION.md).

## Progres milestone

| Milestone | Ketentuan ringkas dari TOR | Status yang dapat diverifikasi dari berkas saat ini |
| --- | --- | --- |
| **M0 — Proposal dan reproduksi baseline** | Proposal, angka InfoNCE/Recall@1 dari lembar kerja, pembagian peran, dan pernyataan integritas. | **Artefak tersedia:** [`reports/M0_Proposal_T03.pdf`](reports/M0_Proposal_T03.pdf) dan [`M0_Pernyataan Integritas_T03.pdf`](M0_Pernyataan%20Integritas_T03.pdf). Proposal mencatat demonstrasi sintetis enam pasangan: InfoNCE 10,6138 → 0,1618 dan Recall@1 1/6 → 6/6. Angka ini **bukan** metrik retrieval SEA-VL. Proposal merencanakan penggantian encoder dan proyeksi kecil untuk M2. |
| **M1 — Baseline pencarian lintas modal** | Pipeline gambar–teks, Recall@1/5/10 dan mAP@K untuk empat arah pada validasi, serta 10 contoh query Indonesia termasuk kegagalan. | **Kode tersedia:** [`src/M1_Baseline.py`](src/M1_Baseline.py) dan [`notebooks/m1_baseline.ipynb`](notebooks/m1_baseline.ipynb). Output eksekusi M1 tersimpan di `Q:\Tubes_Multimodal\K03_Notebook.ipynb` **di luar repo**; notebook M1 dalam repo belum menyimpan output sel. Ringkasan angka dan contoh ada di bawah. Tidak ada alasan untuk menyunting hasil M1 hanya karena contoh gagal. |
| **M2 — Strategi penyejajaran/fusi** | Bandingkan sedikitnya dua varian dengan baseline M1 dan laporkan ablation serta biaya komputasi. | **Hasil utama tersedia secara lokal, di luar repo**, pada `Q:\Tubes_Multimodal\m2_perbandingan.ipynb`. Repo ini menyimpan [notebook audit kebocoran M2](notebooks/m2_audit_leakage.ipynb) dan [hasil auditnya](results/m2_audit/), bukan notebook maupun tabel hasil utama M2. Ringkasan keduanya ada di bawah. |
| **M3 — Kasus sulit dan etika** | Uji kasus sulit, analisis keterbatasan, etika, dan atribusi. | **Eksperimen validasi tersedia secara lokal:** [notebook uji](src/uji_kasus_sulit.ipynb), [hasil tiga seed dan draf analisis/etika](results/m3/). Lima kasus utama (A), lima tambahan (B), dan kontrol diuji pada seed 42, 123, dan 456. **Belum ada laporan PDF final.** Notebook saat ini memiliki ketidaksesuaian pemeriksaan seed yang perlu diperbaiki sebelum dijalankan ulang; draf di repo belum memuat bagian atribusi tersendiri. |
| **M4 — Protokol dan test privat** | Protokol evaluasi serta satu pengajuan prediksi test privat sesuai petunjuk dosen. | **Belum ditemukan artefak M4 atau test privat.** Jangan menebak/merekonstruksi test set yang belum dibuka. |


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

Sampel berikut mengikuti output notebook lokal. Kolom Top-1 adalah prediksi sistem, bukan gambar pasangan benar. Teks query panjang ditampilkan ringkas; teks lengkap serta Top-2/Top-3 tersedia pada output notebook. 

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

## M2: perbandingan model dan audit kebocoran

### Hasil utama

Evaluasi menggunakan 275 pasangan validasi; satu pasangan pada baris yang sama dianggap benar untuk setiap kueri. Encoder pra-latih tetap beku, sedangkan varian `+ proyeksi` melatih proyeksi residual pada embedding teks Indonesia. Nilai `±` adalah **rerata ± simpangan baku tiga seed** untuk varian yang dilatih; angka tanpa proyeksi berasal dari satu evaluasi. ID = teks Indonesia, G = gambar.

| Varian | R@1 ID→G | R@10 ID→G | mAP@10 ID→G | R@1 G→ID |
| --- | ---: | ---: | ---: | ---: |
| Baseline M1 | 0,1964 | 0,5055 | 0,2830 | 0,2145 |
| SigLIP 2 Base | 0,4982 | 0,8545 | 0,6122 | 0,5527 |
| NLLB-CLIP-SigLIP | 0,1964 | 0,4582 | 0,2642 | 0,2218 |
| Baseline + proyeksi | 0,2848 ± 0,0181 | 0,7018 ± 0,0059 | 0,4126 ± 0,0145 | 0,3176 ± 0,0249 |
| SigLIP 2 + proyeksi | **0,5685 ± 0,0045** | **0,9236 ± 0,0079** | **0,6861 ± 0,0014** | **0,5733 ± 0,0062** |

Pada arah ID→G, SigLIP 2 tanpa proyeksi meningkatkan R@1 sebesar **0,3018** dari baseline M1; penambahan proyeksi di atas SigLIP 2 meningkatkan rerata R@1 sebesar **0,0703**. NLLB-CLIP-SigLIP tidak meningkatkan R@1 ID→G dibanding baseline. Waktu encode validasi yang tercatat pada CPU adalah **235,3 detik** (baseline), **415,8 detik** (SigLIP 2), dan **933,1 detik** (NLLB-CLIP); waktu melatih proyeksi sekitar **2,8–3,2 detik per seed**, di luar biaya menyiapkan embedding train. Angka waktu ini bukan benchmark inferensi dengan lingkungan yang dikontrol seragam. Belum ada hasil test privat; encoder terbaik dipilih berdasarkan validasi yang juga dipakai melaporkan metrik.

### Audit kebocoran

[`notebooks/m2_audit_leakage.ipynb`](notebooks/m2_audit_leakage.ipynb) dan [`results/m2_audit/`](results/m2_audit/) memeriksa irisan caption, kemiripan gambar lintas train–val, sensitivitas R@1, serta ablation filter proyeksi. Dari **2.072 train** dan **275 val**, audit menandai **21 train untuk dikeluarkan** (20 berdasarkan caption seperti pada notebook utama, ditambah satu gambar sangat mirip); pada val ada **13 pasti bocor**, **24 perlu ditinjau**, dan **238 bersih**. Sebanyak **12 val** memiliki irisan caption persis dengan train awal. Filter caption pada eksperimen utama menyisakan **2.052 train**, sedangkan filter audit menyisakan **2.051**; jangan menganggapnya satu run yang sama.

R@1 pada val penuh untuk **tiga encoder tanpa proyeksi** cocok dengan notebook utama. Untuk varian proyeksi, audit menguji **10 seed** pada tiga kondisi filter dan hanya menyimpan R@1, sedangkan notebook utama melaporkan **3 seed** serta Recall@5/10 dan mAP@10. Sebagai contoh, dengan filter caption, R@1 ID→G di audit adalah **0,2749 ± 0,0149** (baseline + proyeksi) dan **0,5629 ± 0,0063** (SigLIP 2 + proyeksi); perbedaannya dari tabel utama tidak boleh ditafsirkan sebagai hasil mAP@10 yang bertentangan.

**Batas interpretasi:** evaluasi “val bersih” membuang 13 baris dari **kueri sekaligus galeri**, sehingga perbandingannya dengan 275 kandidat pada val penuh tidak mengisolasi pengaruh kebocoran. Uji antar-kondisi dalam [`audit_uji_seed.csv`](results/m2_audit/audit_uji_seed.csv) memakai Welch meskipun kondisi menggunakan seed yang sama; nilai *p* tersebut perlu ditinjau sebelum dijadikan bukti signifikansi. Audit tidak memeriksa kemungkinan kontaminasi data saat pra-latih encoder.

## M3: uji kasus sulit dan etika

Uji M3 memakai sepuluh gambar dari `val.csv`: [lima kasus utama A dan lima tambahan B](results/m3/draf_hasil_analisis_etika_m3.md). Setiap gambar dibandingkan dengan dua teks yang mirip tetapi berbeda negasi, atribut, relasi spasial, benda, atau ciri arsitektur; ada pula pasangan kontrol yang lebih kontras. Skor kandidat dinormalisasi dengan *softmax* bertemperatur 0,07 **di dalam masing-masing pasangan dua teks**. Angkanya adalah probabilitas relatif terhadap pengecoh yang dipilih, **bukan** keyakinan terkalibrasi atau metrik retrieval pada seluruh galeri.

| Seed proyeksi | Hasil lengkap | Kasus utama A | Kasus tambahan B | Kontrol A dan B |
| ---: | --- | ---: | ---: | ---: |
| 42 | [`hasil_kandidat.csv` dan `metadata.json`](results/m3/m3_seed42_20261125T122156Z_464e7b8e/) | 5/5 | 5/5 | 10/10 |
| 123 | [`hasil_kandidat.csv` dan `metadata.json`](results/m3/m3_seed123_20261125T123801Z_72c5b540/) | 5/5 | 5/5 | 10/10 |
| 456 | [`hasil_kandidat.csv` dan `metadata.json`](results/m3/m3_seed456_20261125T123801Z_e9899fa4/) | 5/5 | 5/5 | 10/10 |

Tiap folder berisi 40 baris kandidat (10 gambar × 2 jenis pasangan × 2 teks), identitas checkpoint dan hash input di `metadata.json`. Tabel berikut menampilkan **kedua kandidat** pada tiap pasangan: `S/P` = probabilitas relatif teks **sesuai/pengecoh**, dalam persen, dibulatkan dua desimal. Semua pasangan berstatus lolos pada ketiga seed; skor kosinus tanpa pembulatan ada di CSV. Kelompok **A** adalah lima gambar utama TOR, sedangkan **B** adalah lima gambar tambahan.

### Pasangan sulit: relasi, negasi, atribut, dan penanda budaya

| Kasus · fokus | `image_id` | Teks sesuai (S) | Teks pengecoh (P) | Seed 42 S/P (%) | Seed 123 S/P (%) | Seed 456 S/P (%) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| A1 · negasi | `svl_142112996` | Tiang bendera tanpa bendera. | Tiang bendera dengan bendera. | 60,06/39,94 | 57,96/42,04 | 59,81/40,19 |
| A2 · pose | `svl_140084661` | Tiga penari berbaju kuning duduk. | Tiga penari berbaju kuning berdiri. | 59,93/40,07 | 61,24/38,76 | 61,35/38,65 |
| A3 · warna | `svl_140084668` | Penari berbaju putih berdiri. | Penari berbaju kuning berdiri. | 62,20/37,80 | 62,70/37,30 | 65,30/34,70 |
| A4 · spasial | `svl_141004074` | Pedang di kiri foto, perisai di kanan foto. | Pedang di kanan foto, perisai di kiri foto. | 53,08/46,92 | 52,76/47,24 | 54,11/45,89 |
| A5 · benda budaya | `svl_140084373` | Kemeja batik di atas meja. | Taplak batik menutupi meja. | 76,94/23,06 | 74,12/25,88 | 75,24/24,76 |
| B1 · alat makan | `svl_140084155` | Bakmi di piring dengan sumpit hitam. | Bakmi di piring dengan sendok hijau. | 58,90/41,10 | 62,03/37,97 | 60,21/39,79 |
| B2 · alat makan | `svl_140084156` | Bakmi di piring dengan sendok hijau. | Bakmi di piring dengan sumpit hitam. | 57,94/42,06 | 58,01/41,99 | 59,88/40,12 |
| B3 · benda tarian | `svl_140084408` | Tiga penari mengangkat selendang jingga. | Tiga penari mengangkat payung jingga. | 58,93/41,07 | 60,50/39,50 | 56,65/43,35 |
| B4 · arsitektur budaya | `svl_143496795` | Stupa Borobudur berbentuk lonceng. | Menara Prambanan berujung runcing. | 87,04/12,96 | 83,69/16,31 | 82,62/17,38 |
| B5 · arsitektur budaya | `svl_144425377` | Menara Prambanan berujung runcing. | Stupa Borobudur berbentuk lonceng. | 62,76/37,24 | 65,46/34,54 | 79,01/20,99 |

### Pasangan kontrol: pengenalan objek yang lebih kontras

Kontrol memakai **gambar yang sama** dengan kasus sulit pada baris ber-ID sama. Angka tetap dinormalisasi hanya terhadap dua teks kontrol di baris tersebut; jangan bandingkan persentasenya lintas pasangan sebagai ukuran keyakinan terkalibrasi.

| Kasus | Teks sesuai (S) | Teks pengecoh (P) | Seed 42 S/P (%) | Seed 123 S/P (%) | Seed 456 S/P (%) |
| --- | --- | --- | ---: | ---: | ---: |
| A1 | Foto tiang bendera. | Foto menara candi. | 80,65/19,35 | 80,85/19,15 | 84,57/15,43 |
| A2 | Foto para penari. | Foto stupa candi. | 87,10/12,90 | 86,17/13,83 | 91,18/8,82 |
| A3 | Foto para penari. | Foto stupa candi. | 96,98/3,02 | 97,31/2,69 | 97,86/2,14 |
| A4 | Foto para penari. | Foto stupa candi. | 93,32/6,68 | 93,01/6,99 | 95,68/4,32 |
| A5 | Foto kemeja batik. | Foto menara candi. | 98,87/1,13 | 98,98/1,02 | 99,16/0,84 |
| B1 | Foto sepiring bakmi. | Foto stupa candi. | 98,53/1,47 | 98,79/1,21 | 99,06/0,94 |
| B2 | Foto sepiring bakmi. | Foto stupa candi. | 95,70/4,30 | 96,91/3,09 | 97,74/2,26 |
| B3 | Foto para penari. | Foto stupa candi. | 92,55/7,45 | 90,37/9,63 | 93,79/6,21 |
| B4 | Foto stupa candi. | Foto para penari. | 98,27/1,73 | 98,27/1,73 | 97,45/2,55 |
| B5 | Foto menara candi. | Foto sepiring bakmi. | 99,74/0,26 | 99,69/0,31 | 99,60/0,40 |

Pada kasus spasial **A4**, probabilitas relatif teks sesuai hanya **53,08% / 52,76% / 54,11%** untuk seed 42/123/456: pilihan benar, tetapi selisihnya kecil. Karena kasus sulit dan kontrol sama-sama lolos serta menggunakan pasangan pengecoh dengan tingkat kontras berbeda, hasil ini **tidak membuktikan** bahwa sistem secara umum lebih mudah menangani objek tunggal daripada relasi, apalagi memahami budaya. Sepuluh gambar yang sama pada tiga seed juga bukan tiga sampel independen. Analisis *dual-encoder* dan bahan pernyataan etika ada di [draf M3](results/m3/draf_hasil_analisis_etika_m3.md).

**Reproduksi dan status pengumpulan:** [notebook M3](src/uji_kasus_sulit.ipynb) memerlukan bundel data dan checkpoint proyeksi M2 seed 42/123/456 di Google Drive; keduanya **tidak disertakan dalam repo**. Metadata hasil mencatat hash checkpoint untuk pemeriksaan asal, bukan menyediakan bobotnya. Versi notebook di repo belum dapat dijalankan ulang apa adanya: `SEEDS` berisi ketiga seed, tetapi fungsi `periksa_berkas_awal` hanya menerima seed 123/456. Hasil CSV yang sudah ada tetap merupakan hasil eksperimen yang tersimpan, bukan output yang dibuat ulang dari notebook dalam repo saat ini. Draf yang tersedia **belum** merupakan laporan pengumpulan: bagian atribusi perlu dipastikan masuk laporan akhir (lihat [`ATTRIBUTION.md`](ATTRIBUTION.md)), pernyataan etika harus diperiksa agar maksimal satu halaman, dan PDF serta kode uji disiapkan sesuai petunjuk M3.


