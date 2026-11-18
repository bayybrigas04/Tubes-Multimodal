# Rancangan M2 — Penyejajaran dan Fusi

**Kelompok 03 (Titut Encoder)** · IF25-40304 Pembelajaran Mesin Multimodal
**Status:** dokumen ini mencatat rancangan awal M2. Implementasi ada di [`../src/M2_Encoders.py`](../src/M2_Encoders.py); eksperimen val E1, E2, dan ablasi proyeksi sudah dijalankan. Lihat hasil aktual di [M2_Hasil.md](M2_Hasil.md) dan catatan kode di [M2_Panduan_Kode_T03.md](M2_Panduan_Kode_T03.md). Butir rencana di bawah dipertahankan sebagai jejak rancangan, bukan klaim seluruh analisis lanjutan telah selesai.

## 1. Tujuan dan batasan

M2 menguji hipotesis dalam [`M0_Proposal_T03.pdf`](M0_Proposal_T03.pdf): apakah penggantian encoder multibahasa meningkatkan retrieval bahasa Indonesia dan mempersempit *language gap* dibanding baseline M1; apakah proyeksi kecil yang dilatih di atas embedding beku memberi tambahan perbaikan. Pendekatan utama adalah **(a) penggantian encoder**, dengan **(c) proyeksi/adapter kecil** sebagai kombinasi. Ini bukan perubahan metode M1: angka M1 lama tetap pembanding apa adanya.

Patuhi [TOR](../Proyek_Akhir_TOR.pdf): encoder pra-latih selalu dibekukan, tidak ada pelatihan dari nol, hanya data dari bundel mahasiswa, tidak mengakses test privat sebelum dibuka, seluruh eksperimen dapat dijalankan di CPU, dan angka hanya dilaporkan setelah benar-benar diperoleh. Pengunduhan bobot encoder pra-latih yang diizinkan berbeda dari menambahkan sampel dataset; tidak boleh menambah gambar/caption di luar bundel.

## 2. Data, baseline, dan protokol tetap

- **Data:** `train.csv` (2.072 pasangan) untuk melatih proyeksi kecil; `val.csv` (275 pasangan) sebagai evaluasi final M2. Gambar dan caption Indonesia/Inggris berasal dari bundel dosen. Kecocokan nama file, missing values, dan irisan train–val diperiksa sebelum menjalankan varian. Jangan mengedit CSV atau gambar asli.
- **Baseline M1:** [`../src/M1_Baseline.py`](../src/M1_Baseline.py) menggunakan `clip-ViT-B-32` untuk gambar dan caption Inggris serta `clip-ViT-B-32-multilingual-v1` untuk caption Indonesia. Empat arah, 275 kandidat, Recall@1/5/10 dan mAP@10 dipertahankan. Output notebook M1 ada di `Q:\Tubes_Multimodal\K03_Notebook.ipynb` (di luar repo); reproduksi lokal juga ada di `results/m1/` bila tersedia pada mesin yang sama. Jangan menimpa keduanya.
- **Pembandingan adil:** urutan 275 pasangan dan definisi satu pasangan relevan per query harus sama pada semua varian; embedding dinormalisasi dan skor menggunakan kesamaan kosinus. Peringkat dihitung pada galeri validasi yang sama, tanpa mencampur train ke galeri.
- **Label metrik:** Recall@K adalah proporsi query yang pasangan barisnya ditemukan sampai peringkat K. Dengan satu pasangan relevan per query, AP@10 = `1/rank` bila `rank <= 10`, selain itu nol; mAP@10 adalah rata-rata AP@10. Caption yang sama/semantik mirip dapat menghasilkan lebih dari satu jawaban yang tampak benar bagi manusia; keterbatasan label ini harus dicatat.
- **Leakage/sensitivitas:** proposal M0 mencatat 10 caption lintas split; skrip M1 menandai 12 *baris validasi* apabila caption ID **atau** EN beririsan dengan train. Angka ini memakai unit/definisi berbeda. Laporan utama tetap memakai val resmi; analisis sensitivitas boleh membatasi **query** yang beririsan sambil mempertahankan galeri 275 kandidat yang sama. Jangan mengganti angka baseline M1 atau menyamakan sensitivitas baru dengan tabel M1 lama yang juga mengecilkan galeri.

## 3. Matriks varian dan ablation

Semua baris adalah **rencana**. Tentukan ID model, revisi bobot, lisensi, arsitektur pasangan image–text, dimensi output, tokenisasi, dan dukungan CPU **sebelum** implementasi. Nama model dalam proposal belum cukup untuk memastikan checkpoint tertentu kompatibel; jangan diam-diam mengganti salah satu kandidat dengan model lain.

| Kode | Encoder gambar + teks | Proyeksi yang dilatih | Tujuan pembandingan |
| --- | --- | --- | --- |
| **B0** | Pasangan CLIP baseline M1 | Tidak ada | Acuan empat arah; pakai angka M1 yang tersimpan, jangan ditimpa. |
| **E1** | SigLIP 2 Base yang cocok untuk kedua modal dan caption Indonesia/Inggris | Tidak ada | Efek penggantian encoder terhadap B0. |
| **E2** | NLLB-CLIP-SigLIP dengan pasangan encoder gambar–teks yang memang kompatibel | Tidak ada | Varian kedua pendekatan utama; bandingkan dengan B0 dan E1. |
| **E1+P-L** | Encoder **sama persis** dengan E1 | Proyeksi linear kecil | Efek proyeksi: E1+P-L dibanding E1, tanpa mengganti encoder. |
| **E1+P-M** | Encoder **sama persis** dengan E1 | Proyeksi MLP kecil | Varian proyeksi kedua: E1+P-M dibanding E1 dan E1+P-L. |

**Keputusan implementasi yang harus dikunci sebelum eksekusi:** checkpoint SigLIP 2 Base dan NLLB-CLIP-SigLIP yang tepat; apakah ruang embedding image–text tiap kandidat sudah kompatibel tanpa adapter; dimensi proyeksi, fungsi loss kontrastif, jumlah epoch, batch size, learning rate, dan seed. Tidak ada angka/konfigurasi yang boleh diklaim sebagai hasil hanya dari rancangan ini. Jika salah satu kandidat tidak memenuhi batas CPU/kompatibilitas/akses bobot, diskusikan perubahan dengan kelompok dan revisi proposal/protokol secara eksplisit sebelum mengganti varian.

## 4. Urutan implementasi yang direncanakan (belum dieksekusi)

1. **Pra-eksekusi:** cocokkan versi/revisi checkpoint dan periksa kebutuhan RAM/disk/waktu CPU. Kunci seed, perangkat, pustaka, cara memakai processor/tokenizer bawaan tiap model, serta format cache yang unik per model/split/revisi. Jangan memakai cache embedding M1 untuk encoder baru.
2. **Embedding beku:** muat model sesuai pasangan yang direkomendasikan checkpoint, set mode evaluasi, matikan gradien encoder, lalu hitung embedding gambar dan kedua bahasa pada train/val yang dibutuhkan. Cache menyertakan ID model, revisi, split, urutan `image_id`, dan konfigurasi preprocessing untuk mencegah hasil tertukar. B0 hanya dibaca sebagai pembanding; tidak dilatih ulang untuk M2.
3. **Retrieval tanpa proyeksi:** jalankan E1 dan E2 pada val untuk menghasilkan keempat tabel metrik. Pastikan dimensi gambar/teks kompatibel sebelum perkalian kosinus. Jangan menyejajarkan embedding dua model berbeda tanpa justifikasi/komponen eksplisit.
4. **Pelatihan proyeksi kecil:** bekukan encoder E1 dan latih hanya lapisan proyeksi menggunakan pasangan `train.csv`. Atur pembagian train-internal untuk memilih konfigurasi/epoch, bukan memilih varian terbaik berulang kali dari val. Periksa caption duplikat dalam batch karena negatif palsu dapat mengganggu loss kontrastif. Catat parameter yang dilatih, seed, dan waktu. Terapkan proyeksi yang sesuai pada embedding gambar/teks dan normalisasi ulang sebelum kosinus; jangan mengubah bobot encoder.
5. **Evaluasi akhir:** dengan kandidat dan protokol tetap, hitung metrik empat arah untuk seluruh varian. Simpan peringkat per query agar analisis kegagalan, selisih dengan B0, dan paired bootstrap yang direncanakan dalam proposal dapat diaudit. Bila menghitung interval kepercayaan 95%, lakukan bootstrap berpasangan pada unit query yang sama; laporkan hasil apa adanya, bukan memaksa perbaikan.
6. **Biaya:** catat total waktu embedding, waktu pelatihan proyeksi, waktu retrieval, perangkat/CPU, konsumsi memori bila terukur, jumlah parameter encoder yang dibekukan dan parameter proyeksi yang dilatih, serta kesulitan instalasi/reproduksi. Angka biaya tidak diisi sebelum pengukuran.

## 5. Bentuk laporan M2 yang akan diisi setelah ada hasil nyata

Satu baris per **varian × arah**, minimal lima varian di atas bila seluruhnya lulus eksekusi. Kolom: `varian`, `arah`, `Recall@1`, `Recall@5`, `Recall@10`, `mAP@10`, `selisih Recall@1 terhadap B0`, `waktu`, `parameter dilatih`, `catatan`. Tambahkan tabel *language gap* Recall@1 (EN − ID) per arah dan varian, efek ablation E1+P terhadap E1, serta interval kepercayaan bila berhasil dihitung. Laporkan kegagalan/varian tidak dapat dijalankan sebagai keterbatasan, bukan mengisi angka perkiraan.

Keterangan baseline M1 yang sudah ada (hanya referensi, **bukan hasil M2**):

| Arah | Recall@1 | Recall@5 | Recall@10 | mAP@10 |
| --- | ---: | ---: | ---: | ---: |
| Gambar → teks Indonesia | 0,2145 | 0,3745 | 0,5055 | 0,2930 |
| Gambar → teks Inggris | 0,4473 | 0,7127 | 0,8327 | 0,5594 |
| Teks Indonesia → gambar | 0,1964 | 0,4182 | 0,5055 | 0,2830 |
| Teks Inggris → gambar | 0,3818 | 0,6727 | 0,7709 | 0,5073 |

## 6. Kaitan dengan peran kelompok

- **Bayu Brigas Novaldi — Baseline/Pipeline:** integrasi encoder baru, embedding beku, cache dan ranking yang dapat dijalankan ulang.
- **Anisah Octa Rohila — Eksperimen:** dua varian encoder, dua varian proyeksi, matriks ablation, waktu/parameter.
- **Aryasatya Widyatna Akbar — Evaluasi:** empat arah Recall@K/mAP@10, *language gap*, paired comparison/ketidakpastian, dan daftar contoh keberhasilan/kegagalan.
- **Ahmad Ali Mukti — Data/Etika:** audit irisan dan duplikat untuk varian, lisensi dan keterwakilan data.
- **Jesika Filosovi Br P-A — Integrasi:** menyatukan laporan M2, petunjuk reproduksi, serta bukti kontribusi.

Semua hasil nantinya wajib menyertakan atribusi SEA-VL/SEACrowd/KORIKA-AI, Cahyawijaya dkk. (2025), dan lisensi **CC BY-SA 4.0** sebagaimana [`../ATTRIBUTION.md`](../ATTRIBUTION.md).
