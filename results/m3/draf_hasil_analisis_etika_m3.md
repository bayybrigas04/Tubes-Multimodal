# Draf M3 — uji kasus sulit, analisis, dan etika

**Status:** bahan laporan berbasis hasil yang sudah dijalankan; belum menjadi PDF pengumpulan. Lima kasus A memenuhi jumlah minimum TOR; lima kasus B adalah uji tambahan, bukan pengganti A. Semua gambar berasal dari `val.csv`, bukan *test set* privat.

## Rancangan dan pembacaan probabilitas

Sistem memakai SigLIP 2 Base (`ViT-B-16-SigLIP2-256`, `webli`) dengan proyeksi residual teks Indonesia dari checkpoint run ulang M2. Encoder gambar dan teks dipakai untuk membandingkan satu gambar dengan **dua** deskripsi. Seed proyeksi 42, 123, dan 456 diuji pada sepuluh gambar dan pasangan teks yang sama; hasil seed 42 adalah uji awal, sedangkan dua seed lain memeriksa konsistensi pilihan. Skor kosinus diubah menjadi *softmax* dengan temperatur 0,07 pada **setiap pasangan**, bukan di seluruh kumpulan gambar. Angka S/P di bawah berarti persentase relatif teks **sesuai/pengecoh**; keduanya berjumlah 100% sebelum pembulatan. Ini **bukan** probabilitas terkalibrasi bahwa deskripsi benar. Pada semua baris, S dipilih (skor S > P); aturan seri ialah gagal.

### Kelompok A — lima gambar utama

| Kasus | `image_id` | Teks sesuai | Teks pengecoh | Seed 42 S/P (%) | Seed 123 S/P (%) | Seed 456 S/P (%) |
|---|---|---|---|---:|---:|---:|
| A1 · negasi | `svl_142112996` | Tiang bendera tanpa bendera. | Tiang bendera dengan bendera. | 60,06/39,94 | 57,96/42,04 | 59,81/40,19 |
| A2 · pose | `svl_140084661` | Tiga penari berbaju kuning duduk. | Tiga penari berbaju kuning berdiri. | 59,93/40,07 | 61,24/38,76 | 61,35/38,65 |
| A3 · atribut warna | `svl_140084668` | Penari berbaju putih berdiri. | Penari berbaju kuning berdiri. | 62,20/37,80 | 62,70/37,30 | 65,30/34,70 |
| A4 · relasi spasial | `svl_141004074` | Pedang di kiri foto, perisai di kanan foto. | Pedang di kanan foto, perisai di kiri foto. | 53,08/46,92 | 52,76/47,24 | 54,11/45,89 |
| A5 · objek bermotif budaya | `svl_140084373` | Kemeja batik di atas meja. | Taplak batik menutupi meja. | 76,94/23,06 | 74,12/25,88 | 75,24/24,76 |

**Kontrol untuk gambar A yang sama:** pasangan kontrol sengaja membedakan kategori objek yang jauh lebih kontras daripada pasangan sulit. Persentase tetap S/P untuk kedua kandidat pada baris tersebut.

| Kasus | Kontrol sesuai | Kontrol pengecoh | Seed 42 S/P (%) | Seed 123 S/P (%) | Seed 456 S/P (%) |
|---|---|---|---:|---:|---:|
| A1 | Foto tiang bendera. | Foto menara candi. | 80,65/19,35 | 80,85/19,15 | 84,57/15,43 |
| A2 | Foto para penari. | Foto stupa candi. | 87,10/12,90 | 86,17/13,83 | 91,18/8,82 |
| A3 | Foto para penari. | Foto stupa candi. | 96,98/3,02 | 97,31/2,69 | 97,86/2,14 |
| A4 | Foto para penari. | Foto stupa candi. | 93,32/6,68 | 93,01/6,99 | 95,68/4,32 |
| A5 | Foto kemeja batik. | Foto menara candi. | 98,87/1,13 | 98,98/1,02 | 99,16/0,84 |

### Kelompok B — lima gambar tambahan

| Kasus | `image_id` | Teks sesuai | Teks pengecoh | Seed 42 S/P (%) | Seed 123 S/P (%) | Seed 456 S/P (%) |
|---|---|---|---|---:|---:|---:|
| B1 · alat makan | `svl_140084155` | Bakmi di piring dengan sumpit hitam. | Bakmi di piring dengan sendok hijau. | 58,90/41,10 | 62,03/37,97 | 60,21/39,79 |
| B2 · alat makan | `svl_140084156` | Bakmi di piring dengan sendok hijau. | Bakmi di piring dengan sumpit hitam. | 57,94/42,06 | 58,01/41,99 | 59,88/40,12 |
| B3 · benda dalam tarian | `svl_140084408` | Tiga penari mengangkat selendang jingga. | Tiga penari mengangkat payung jingga. | 58,93/41,07 | 60,50/39,50 | 56,65/43,35 |
| B4 · arsitektur | `svl_143496795` | Stupa Borobudur berbentuk lonceng. | Menara Prambanan berujung runcing. | 87,04/12,96 | 83,69/16,31 | 82,62/17,38 |
| B5 · arsitektur | `svl_144425377` | Menara Prambanan berujung runcing. | Stupa Borobudur berbentuk lonceng. | 62,76/37,24 | 65,46/34,54 | 79,01/20,99 |

| Kasus | Kontrol sesuai | Kontrol pengecoh | Seed 42 S/P (%) | Seed 123 S/P (%) | Seed 456 S/P (%) |
|---|---|---|---:|---:|---:|
| B1 | Foto sepiring bakmi. | Foto stupa candi. | 98,53/1,47 | 98,79/1,21 | 99,06/0,94 |
| B2 | Foto sepiring bakmi. | Foto stupa candi. | 95,70/4,30 | 96,91/3,09 | 97,74/2,26 |
| B3 | Foto para penari. | Foto stupa candi. | 92,55/7,45 | 90,37/9,63 | 93,79/6,21 |
| B4 | Foto stupa candi. | Foto para penari. | 98,27/1,73 | 98,27/1,73 | 97,45/2,55 |
| B5 | Foto menara candi. | Foto sepiring bakmi. | 99,74/0,26 | 99,69/0,31 | 99,60/0,40 |

## Hasil dan analisis kritis

Pada tiap seed, kelima kasus sulit A, kelima kasus tambahan B, dan seluruh kontrol dipilih dengan benar: masing-masing **5/5** pada setiap kelompok dan jenis uji, tanpa kegagalan pilihan yang teramati. Karena semua lolos, tingkat keberhasilan sampel ini **tidak membuktikan** objek tunggal lebih mudah daripada relasi. Namun pada setiap gambar, pasangan kontrol yang membedakan kategori jauh berbeda umumnya memberi keunggulan relatif S yang lebih besar daripada pasangan sulit. Perbandingan ini hanya deskriptif: kontrol memiliki pengecoh yang lebih mudah dibedakan, panjang dan isi kalimat tidak selalu sepadan, dan nilai *softmax* dari **pasangan kandidat berbeda** tidak bisa diperlakukan sebagai ukuran kesulitan yang terkalibrasi.

A4 paling dekat ke batas pilihan pada tiga seed: S/P = 53,08/46,92; 52,76/47,24; dan 54,11/45,89. Artinya susunan kiri–kanan yang benar unggul tipis atas susunan terbalik **pada foto ini**. A1 (negasi) dan B1–B3 (alat/benda yang berubah) juga memiliki pasangan yang relatif dekat; keberhasilan memilih tidak menunjukkan sistem memahami negasi, posisi, atau peran benda di gambar lain. B4 lebih mudah dibedakan daripada B5 pada ketiga seed, tetapi nama lokasi dan bentuk bangunan berubah bersama-sama. Karena itu hasil B4/B5 tidak mengisolasi pemahaman istilah budaya dari pengenalan bentuk visual atau asosiasi nama situs.

Pada arsitektur *dual-encoder*, gambar dan teks dipetakan secara terpisah menjadi vektor lalu dibandingkan menggunakan kemiripan global. Pembedaan antara dua teks yang berbagi hampir semua kata bergantung pada apakah perubahan negasi, atribut, atau urutan kiri–kanan cukup menggeser vektor teks dan selaras dengan vektor gambar. Arsitektur ini memungkinkan pencocokan semantik umum tetapi **tidak menjamin** pemeriksaan eksplisit setiap objek dan relasinya. Margin kecil A4 konsisten dengan keterbatasan tersebut, bukan bukti bahwa arsitektur menyebabkan satu kesalahan tertentu: A4 tetap benar dan tidak ada uji kausal yang mengisolasi mekanismenya. Pemilihan gambar/kasus pada split validasi juga membatasi generalisasi; tiga seed mengulang pasangan yang sama, bukan tiga sampel gambar independen.

## Pernyataan etika dan limitasi — teks ringkas untuk maksimum satu halaman

**Kesenjangan bahasa.** Pada output M2 untuk SigLIP 2 Base + proyeksi, Recall@1 validasi teks Indonesia→gambar adalah **0,5685 ± 0,0045** (rerata ± simpangan baku tiga seed), sedangkan teks Inggris→gambar **0,5891**; selisih Inggris−Indonesia **0,0206** (2,06 poin persentase). Untuk gambar→teks, Indonesia **0,5733 ± 0,0062** dan Inggris **0,5673**; selisih rerata sekitar **−0,0061** (−0,61 poin; perbedaan pembulatan). Jadi bahasa Inggris tidak unggul pada kedua arah. Pada baseline M1, gap masing-masing 18,55 dan 23,27 poin, tetapi modelnya berbeda; angka M1 bukan ukuran gap sistem M3. Pengguna yang mencari konten budaya dengan bahasa Indonesia berpotensi mendapat peringkat yang lebih buruk pada arah teks→gambar, terutama bila bentuk bahasa/istilahnya tidak terwakili; hasil ini belum mengukur dampak per kelompok pengguna atau bahasa daerah.

**Keterwakilan.** Audit langsung `train.csv`/`val.csv` menunjukkan **2.072/275** baris; kategori `kuliner` mencakup **792/106** baris, sedangkan `alat musik` hanya **4/0**. Kolom `native_lang` mencatat `Indonesian (ind)` pada **2.065/272** baris dan `Javanese (jav)` pada **7/3** baris. Kategori adalah label kasar berbasis kata kunci, bukan daftar kelompok budaya yang tervalidasi; `native_lang` juga bukan identitas etnis. Kolom `loc` berisi negara (misalnya “Indonesia”), **bukan provinsi**, sehingga sebaran antarwilayah Indonesia dan kelompok budaya yang tidak muncul tidak bisa dihitung dari kolom ini. Jangan menganggap seluruh budaya Indonesia terwakili hanya karena baris diberi label Indonesia.

**Kejujuran evaluasi.** Data memisahkan `train` untuk proyeksi dan `val` untuk pelaporan M2/M3; *test set* privat tidak dipakai untuk uji ini. Tanpa split yang jelas, contoh atau caption yang sama dapat masuk pelatihan dan evaluasi sehingga skor tampak lebih baik dari kemampuan pada data baru. Audit train–val mencatat **12** baris validasi dengan caption persis beririsan dengan train awal, **13** baris berstatus pasti bocor, **24** perlu ditinjau, dan **238** bersih; audit ini tidak memastikan pra-latih encoder bebas kontaminasi. Model M2 dipilih berdasarkan validasi yang juga dilaporkan; kasus M3 memakai gambar validasi dan tidak memberi estimasi generalisasi independen. Laporkan kegagalan dan hasil seri apa adanya; jangan mempresentasikan 10/10 pada kasus terpilih sebagai bukti sistem aman untuk semua relasi, penanda budaya, atau bahasa daerah.
