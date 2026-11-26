#import "../_shared/lib.typ": *

#show: instruksi.with(
  kunci: "m3",
  judul: "M3: Uji kasus sulit, etika, dan atribusi",
  subjudul: "Instruksi deliverable milestone keempat",
)

= Ringkasan

#milestone.m3.fokus

#info("Dikumpulkan pada Pertemuan 13")[
  Batas waktu persisnya diumumkan lewat LMS. *Test set dibuka pada Pertemuan 13.* Siapkan draf laporan dan prediksi
  _test set_ sejak pertemuan ini, karena M4 dikumpulkan pada Pertemuan 14.
]

= Yang dikumpulkan

#checklist(
  [Hasil uji kasus sulit: lima gambar dan _query_ yang memakai kata serupa tetapi berbeda relasi, negasi, atau penanda budaya, lengkap dengan probabilitas tiap kandidat.],
  [Analisis apakah sistem lebih mudah menangani objek tunggal daripada relasi, dihubungkan dengan keterbatasan arsitektur _dual-encoder_.],
  [Pernyataan etika dan limitasi, *maksimal satu halaman*.],
  [Atribusi CC BY-SA 4.0 yang benar pada laporan dan kode.],
)

Templat yang dipakai: `laporan-m3.typ` (uji kasus sulit, analisis, atribusi) dan `pernyataan-etika.typ` (satu halaman).

= Langkah pengerjaan

+ Rancang uji kasus sulit. Pilih lima gambar. Untuk tiap gambar, tulis _query_ yang hampir sama kata-katanya tetapi
  berbeda maknanya. Contoh dari data ini:
  - negasi: "nasi padang dengan perkedel" versus "nasi padang tanpa perkedel";
  - atribut: "tari topeng dari Cirebon berkostum merah" versus "berkostum biru".
+ Jalankan sistem terbaik Anda dari M2 pada pasangan itu, dan laporkan probabilitas (atau skor yang dinormalisasi) tiap
  kandidat. Tampilkan kegagalan apa adanya.
+ Analisis. Apakah sistem lebih mudah menangani objek tunggal daripada relasi? Kaitkan temuan dengan cara kerja
  _dual-encoder_: gambar dan teks disematkan secara terpisah sebelum dibandingkan.
+ Tulis pernyataan etika dan limitasi (maksimal satu halaman) yang membahas tiga hal:
  - kesenjangan kinerja antara _query_ Indonesia dan Inggris, serta siapa yang dirugikan olehnya;
  - keterwakilan wilayah dan kelompok budaya di dalam data;
  - kejujuran _benchmark_: mengapa laporan tanpa _split_ yang jelas sulit dipercaya.
+ Cantumkan atribusi CC BY-SA 4.0 dengan benar pada laporan dan kode.

#peringatan("Test set belum untuk disentuh")[
  Pembukaan _test set_ tidak mengubah aturannya: jangan memakainya untuk memilih model atau menyetel apa pun. Pengajuan
  prediksi baru dilakukan di M4 dan hanya boleh satu kali.
]

= Patokan kontribusi per peran

#tabel-peran-milestone("m3")

= Rubrik

#tabel-rubrik-milestone()

#catatan("Yang dinilai pada analisis")[
  M3 menilai seberapa kritis Anda menilai sistem sendiri. Analisis kegagalan yang jujur lebih bernilai daripada contoh yang dipilih agar sistem tampak berhasil.
]

= Cara mengumpulkan

Kumpulkan PDF laporan (`m3_laporan_kelompok_NN.pdf`) yang memuat pernyataan etika, serta kode uji kasus sulit. Bila
petunjuk LMS berbeda, ikuti LMS.

```bash
typst compile --root . m3/laporan-m3.typ m3/m3_laporan_kelompok_NN.pdf
typst compile --root . m3/pernyataan-etika.typ m3/m3_etika_kelompok_NN.pdf
```
