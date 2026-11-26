// M3: Uji kasus sulit, etika, dan atribusi  ·  templat laporan kelompok
// Ganti semua teks ungu ⟦…⟧. Data kelompok diisi di ../grup.typ.
#import "../_shared/lib.typ": *
#import "../grup.typ": grup

#show: deliverable.with(
  kunci: "m3",
  judul: "Laporan M3",
  subjudul: "Uji kasus sulit dan atribusi",
  label: "Laporan M3",
  grup: grup,
)

= Ringkasan

#isi[Tiga sampai lima kalimat: apa yang diuji, kegagalan terbesar, dan pelajaran utamanya.]

= Rancangan uji kasus sulit

Lima gambar; tiap gambar punya dua atau lebih _query_ yang kata-katanya mirip tetapi relasi, negasi, atau penanda budayanya berbeda.

#tabel-isian(
  ([\#], [Gambar (`image_id`)], [Jenis kasus], [Query A], [Query B]),
  baris: 5,
  lebar: (0.7cm, 3.4cm, 2.4cm, 4.2cm, 1fr),
)
#v(-0.2em)
#text(size: 9pt, fill: c-sub)[Jenis kasus: negasi, atribut, relasi, atau penanda budaya.]

= Hasil

Laporkan probabilitas tiap kandidat yang dinormalisasi (_softmax_ atas kandidat), dengan model terbaik dari M2.

#tabel-isian(
  ([\#], [Query], [P(kandidat benar)], [P(pengecoh)], [Lolos?]),
  baris: 10,
  lebar: (0.7cm, 6cm, 3.2cm, 3cm, 1fr),
)

#isi[Ringkas pola: jenis kasus mana yang paling sering gagal, dan seberapa percaya diri sistem saat salah.]

= Analisis: objek tunggal dan relasi

#isi[Apakah sistem lebih mudah menangani objek tunggal daripada relasi? Beri bukti dari tabel di atas dan hubungkan dengan keterbatasan dual-encoder.]

= Atribusi dan lisensi

#atribusi-data

#checklist(
  [Atribusi tercantum di laporan ini.],
  [Atribusi tercantum di `README` dan kode.],
  [Bila ada artefak turunan yang dibagikan, lisensinya CC BY-SA 4.0.],
)

#info("Pernyataan etika dan limitasi")[
  Pernyataan etika ditulis terpisah di `pernyataan-etika.typ` (maksimal satu halaman) dan dilampirkan bersama laporan ini.
]
