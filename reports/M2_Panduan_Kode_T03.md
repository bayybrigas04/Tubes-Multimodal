# M2 — kode siap, keputusan eksekusi menunggu Mak

**Status:** `src/M2_Encoders.py` baru diperiksa sintaks; **belum** diunduh bobotnya, dijalankan pada data, dilatih, atau dievaluasi. Jangan menafsirkan perkiraan berikut sebagai hasil ukur.

**Pilihan lingkungan terbaru:** Mak menyetujui Colab CPU. Peluncur bertahap ada di [`../notebooks/m2_colab_cpu_launcher.ipynb`](../notebooks/m2_colab_cpu_launcher.ipynb) pada branch `eksperimen-m2`: clone branch tersebut langsung, jalankan E1 val tanpa proyeksi terlebih dahulu, lalu kirim hasil/error sebelum E2 atau latihan proyeksi. Skrip M2 diambil dari branch yang sama; tidak perlu upload skrip lokal. Preflight lokal menemukan 2.072 train, 275 val, semua gambar tersedia, tetapi laptop hanya punya sekitar 0,75 GB RAM bebas dari total 8 GB; karena itu **tidak ada pemuatan model di laptop**.

## Pilihan checkpoint untuk disetujui bersama

| Kode | Kandidat dan revisi Hugging Face (SHA, bukan tag bergerak) | Lisensi | Bobot utama |
| --- | --- | --- | ---: |
| E1 | [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224), `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | Apache-2.0 | 1.500.800.904 byte |
| E2 | [`visheratin/nllb-clip-base-siglip`](https://huggingface.co/visheratin/nllb-clip-base-siglip), `a7c4a9bf24872c73a2c059ab2f9e9123a4d4144e` | CC BY-NC 4.0 | 2.030.040.226 byte |

Metadata diakses 18 November 2026 dari endpoint API Hugging Face. Total berkas bobot utama sekitar 3,53 GB desimal; tokenizer, cache, paket, dan kemungkinan dependensi `facebook/nllb-200-distilled-600M` OpenCLIP dapat menambah unduhan/penyimpanan. **E2 melarang pemakaian komersial sesuai lisensinya**; pastikan penggunaan kelompok nonkomersial. E2 memakai pengaturan bahasa `ind_Latn` dan `eng_Latn` via tokenizer OpenCLIP. Kode mengambil snapshot E2 pada SHA dan memuatnya lewat `local-dir:`; beberapa dependensi internal model NLLB masih bisa diambil terpisah oleh library, sehingga pin penuh transitive dependency belum dibuktikan tanpa smoke test.

## Perkiraan biaya CPU (bukan pengukuran)

Laptop lokal terdeteksi Intel Core i5-12450H, 8 core/12 logical processor. Tanpa benchmark pada mesin, rentang kerja awal yang **sengaja longgar** untuk embedding E1 train+val dan E2 val adalah **beberapa jam hingga >1 hari**; memori sebaiknya tersedia **≥8 GB**, lebih nyaman **16 GB**, dan ruang kosong **≥10 GB** untuk bobot, cache, paket, dan overhead. E2 memakai gambar 384×384 dan teks NLLB; lazimnya lebih berat daripada E1 224×224. Latihan proyektor di embedding kecil biasanya lebih murah daripada embedding encoder, tetapi **belum ada angka menit yang dapat dipertanggungjawabkan**. Unduhan 3,53 GB saja membutuhkan ~47 menit pada koneksi stabil 10 Mbps (tanpa overhead/berkas tambahan); pada 50 Mbps ~9,4 menit. Ini hitungan ukuran/kecepatan teoritis, bukan estimasi durasi run keseluruhan. Pemakaian laptop lokal tidak menimbulkan biaya cloud tambahan, tetapi memerlukan waktu, listrik, bandwidth; jangan menyamakan dengan biaya CPU Colab/VM yang belum dipilih.

Sebelum eksekusi: pilih tempat run (laptop/Colab), cek RAM dan disk, sepakati maksimum waktu/bandwidth, serta tentukan `epochs`, `lr`, `batch-size`, seed. Contoh **usulan, belum dikunci**: 5 epoch, learning rate `0.001`, batch 8, seed 42; pengaturan sama untuk proyektor linear dan MLP. Untuk klaim efek proyeksi yang kuat, pemilihan hyperparameter/epoch seharusnya memakai train-internal holdout, bukan berulang kali memilih nilai terbaik di val. Kode saat ini hanya satu konfigurasi eksplisit; tidak mengotomatiskan pencarian hyperparameter.

## Yang disiapkan

- Baca CSV dan gambar tanpa memodifikasinya; baseline M1 dibaca dari `results/m1/tabel_recall_val.csv` tanpa menulis ulang file tersebut.
- E1 via Transformers, E2 via OpenCLIP; encoder `eval()`/`requires_grad_(False)` pada CPU. Embedding val digunakan untuk E1/E2; embedding train hanya untuk proyeksi E1. Normalisasi dan metrik memakai definisi M1 (satu pasangan relevan per baris, galeri 275).
- Ablasi `E1+P-L` dan `E1+P-M` memakai encoder E1 yang sama; proyektor bersama diterapkan ke gambar/teks dan dilatih hanya dengan train. Pasangan train dengan gambar/caption duplikat persis dikeluarkan dari optimasi untuk mengurangi negatif palsu; jumlahnya dicatat di manifest. Ini keputusan desain yang perlu disetujui kelompok.
- Output baru (`m2_val.csv`, `costs.json`, `manifest.json`) hanya di folder `--out` **kosong**; menyimpan tabel empat arah, gap Recall@1 (pada baris Indonesia), delta dari B0, waktu embedding/training, dan hitungan parameter. Belum menyimpan ranking query individual, mengukur memori puncak, mengaudit output dinamis, atau menghitung interval kepercayaan; jangan klaim hal tersebut selesai.
- Tidak membuat `results/m2` otomatis sebelum CLI benar-benar dijalankan.

## Kebutuhan instalasi ketika nanti disetujui

Sediakan lingkungan Python yang kompatibel dengan PyTorch, Transformers (mendukung SigLIP 2), torchvision, dan OpenCLIP. Gunakan **pip pada lingkungan virtual terpisah** yang disiapkan untuk proyek, jangan instalasi global. Paket dasar M1 sudah dicantumkan di `requirements.txt`; tambahan untuk E2 adalah `open_clip_torch` (beserta `timm`, `huggingface_hub` dan dependensi transitifnya). Unduhan terbesar umumnya bobot model (>3,5 GB), bukan skrip; Torch/torchvision sendiri juga dapat memerlukan ratusan MB atau lebih. Tidak ada paket diinstal pada sesi penulisan ini. Verifikasi versi bersama nanti, karena antarmuka OpenCLIP/Transformers dapat berubah.

**Perintah di bawah hanya contoh pemakaian setelah ACC, JANGAN dijalankan sekarang:**

```powershell
# Jalankan dari root repo dengan venv yang sudah dipersiapkan, setelah ACC Mak.
python -m pip install -r requirements.txt open_clip_torch torchvision
python src/M2_Encoders.py --data "Q:\Tubes_Multimodal\bundle-mahasiswa\bundle-mahasiswa\bundle-mahasiswa" --out results/m2-run-pertama --baseline results/m1/tabel_recall_val.csv --e1-revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 --e2-revision a7c4a9bf24872c73a2c059ab2f9e9123a4d4144e --epochs 5 --lr 0.001 --batch-size 8 --seed 42
```

**Gate sebelum klaim hasil:** smoke test API kedua checkpoint, cek dimensi serta RAM/disk, audit pelatihan proyektor, jalankan evaluasi val sekali sesuai protokol, periksa baseline/peringkat. Semua ini **belum dilaksanakan**.
