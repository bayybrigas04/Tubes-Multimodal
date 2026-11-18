"""Evaluasi dua encoder M2 dan ablasi proyeksi; tidak dijalankan saat import."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from M1_Baseline import ARAH, metrik

MODEL_IDS = {
    "E1": "google/siglip2-base-patch16-224",
    "E2": "visheratin/nllb-clip-base-siglip",
}
LANGUAGES = {"caption_id": "ind_Latn", "caption_en": "eng_Latn"}
REQUIRED = ("file", "caption_id", "caption_en")


def load_split(root: Path, split: str) -> pd.DataFrame:
    """Baca CSV dan validasi pasangan serta gambar sebelum memuat model.

    Args:
        root: Direktori bundel mahasiswa.
        split: Nama split, hanya train atau val.
    Returns:
        Tabel dengan path gambar absolut tambahan.
    Raises:
        ValueError: Split/isi CSV tidak valid.
        FileNotFoundError: Berkas CSV atau gambar hilang.
    """
    if split not in ("train", "val"):
        raise ValueError("split hanya boleh 'train' atau 'val'")
    csv_path = root / f"{split}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    table = pd.read_csv(csv_path)
    missing = set(REQUIRED) - set(table.columns)
    if missing or table.empty or table[list(REQUIRED)].isna().any().any():
        raise ValueError(f"{csv_path}: kolom hilang {sorted(missing)}, data kosong/null")
    for column in REQUIRED:
        if not table[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"{csv_path}: nilai kosong/bukan teks pada {column}")
    paths = []
    for name in table["file"]:
        relative = Path(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Path gambar tidak aman: {name}")
        image_path = root / relative
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        paths.append(str(image_path.resolve()))
    table = table.copy()
    table["image_path"] = paths
    if split == "val" and table["file"].duplicated().any():
        raise ValueError(f"{csv_path}: file gambar berulang; diagonal tidak lagi unik")
    return table


def unit_rows(values: np.ndarray) -> np.ndarray:
    """Pastikan embedding finite lalu normalisasi L2 untuk kosinus.

    Args:
        values: Matriks embedding [N,D].
    Returns:
        Matriks float32 ternormalisasi.
    Raises:
        ValueError: Bentuk, nilai, atau norm tidak valid.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embedding harus matriks 2D finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise ValueError("Embedding bernorm nol tidak dapat dibandingkan")
    return values / norms


def encode_split(model_code: str, revision: str, table: pd.DataFrame, batch_size: int) -> tuple[dict[str, np.ndarray], int]:
    """Hitung embedding beku untuk satu split menggunakan processor bawaan.

    Args:
        model_code: E1 (Transformers) atau E2 (OpenCLIP).
        revision: SHA commit Hugging Face yang disepakati.
        table: Tabel hasil load_split.
        batch_size: Ukuran batch CPU positif.
    Returns:
        Embedding image/ID/EN dan jumlah parameter encoder (beku).
    Raises:
        ValueError: Konfigurasi tidak valid.
    """
    if model_code not in MODEL_IDS or len(revision) != 40 or batch_size < 1:
        raise ValueError("Perlu kode E1/E2, SHA revisi 40 karakter dan batch_size > 0")
    import torch
    from PIL import Image

    if model_code == "E1":
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(MODEL_IDS[model_code], revision=revision).eval().to("cpu")
        processor = AutoProcessor.from_pretrained(MODEL_IDS[model_code], revision=revision)

        def image_batch(images: list[Image.Image]) -> torch.Tensor:
            """Encode gambar SigLIP 2 tanpa gradien."""
            return model.get_image_features(**processor(images=images, return_tensors="pt"))

        def text_batch(texts: list[str], language: str) -> torch.Tensor:
            """Encode teks SigLIP 2; tokenizer multilingual yang sama untuk ID/EN."""
            return model.get_text_features(**processor(text=texts, padding="max_length", truncation=True, return_tensors="pt"))

    else:
        import open_clip
        from huggingface_hub import snapshot_download

        # OpenCLIP tidak menerima @revision di URI hf-hub; ambil snapshot SHA dahulu.
        snapshot = snapshot_download(
            MODEL_IDS[model_code], revision=revision,
            allow_patterns=["open_clip_config.json", "open_clip_pytorch_model.bin",
                            "tokenizer.json", "tokenizer_config.json",
                            "sentencepiece.bpe.model", "special_tokens_map.json"],
        )
        uri = f"local-dir:{snapshot}"
        model, _, preprocess = open_clip.create_model_and_transforms(uri, device="cpu")
        tokenizer = open_clip.get_tokenizer(uri)
        model.eval()

        def image_batch(images: list[Image.Image]) -> torch.Tensor:
            """Encode gambar E2 dengan preprocessing OpenCLIP resmi."""
            return model.encode_image(torch.stack([preprocess(image) for image in images]))

        def text_batch(texts: list[str], language: str) -> torch.Tensor:
            """Atur token bahasa NLLB sebelum setiap batch teks."""
            tokenizer.set_language(language)
            return model.encode_text(tokenizer(texts))

    model.requires_grad_(False)
    frozen_parameters = sum(parameter.numel() for parameter in model.parameters())

    def collect(column: str) -> np.ndarray:
        """Encode satu kolom dalam batch CPU tanpa menyimpan semua gambar."""
        parts = []
        for start in range(0, len(table), batch_size):
            rows = table.iloc[start:start + batch_size]
            with torch.inference_mode():
                if column == "image_path":
                    with_images = []
                    for path in rows[column]:
                        with Image.open(path) as image:
                            with_images.append(image.convert("RGB"))
                    output = image_batch(with_images)
                else:
                    output = text_batch(rows[column].tolist(), LANGUAGES[column])
            parts.append(output.detach().cpu().float().numpy())
        return unit_rows(np.concatenate(parts, axis=0))

    embeddings = {"img": collect("image_path"), "ti": collect("caption_id"), "te": collect("caption_en")}
    if len({array.shape[1] for array in embeddings.values()}) != 1:
        raise ValueError("Dimensi encoder gambar/teks tidak kompatibel")
    return embeddings, frozen_parameters


def build_projector(kind: str, dim: int):
    """Bangun proyektor residual kecil; init identitas untuk ablasi adil.

    Args:
        kind: linear atau mlp.
        dim: Dimensi embedding.
    Returns:
        Modul PyTorch dengan satu jalur bersama untuk gambar/teks.
    Raises:
        ValueError: Konfigurasi tidak didukung.
    """
    import torch

    if dim < 1 or kind not in ("linear", "mlp"):
        raise ValueError("kind harus linear/mlp dan dim positif")
    if kind == "linear":
        layer = torch.nn.Linear(dim, dim, bias=False)
        torch.nn.init.eye_(layer.weight)
        return layer
    layer = torch.nn.Sequential(torch.nn.Linear(dim, 128), torch.nn.ReLU(), torch.nn.Linear(128, dim))
    torch.nn.init.zeros_(layer[-1].weight)
    torch.nn.init.zeros_(layer[-1].bias)
    return layer


def fit_projection(train: dict[str, np.ndarray], kind: str, seed: int, epochs: int, batch_size: int, lr: float):
    """Latih hanya proyektor di embedding train dengan loss kontrastif simetris.

    Args:
        train: Embedding beku train (bukan val).
        kind: linear atau mlp.
        seed: Seed inisialisasi/urutan batch.
        epochs: Jumlah epoch positif yang disetujui.
        batch_size: Ukuran batch positif.
        lr: Learning rate positif.
    Returns:
        Modul terlatih dan jumlah parameter yang dilatih.
    Raises:
        ValueError: Konfigurasi/embedding tidak valid.
    """
    import torch
    import torch.nn.functional as functional

    if epochs < 1 or batch_size < 2 or lr <= 0:
        raise ValueError("epochs > 0, batch_size >= 2 dan lr > 0 diperlukan")
    dim = train["img"].shape[1]
    if any(array.shape != train["img"].shape for array in train.values()):
        raise ValueError("Embedding train harus sejajar pada ketiga modalitas")
    torch.manual_seed(seed)
    projector = build_projector(kind, dim)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    tensors = {key: torch.from_numpy(value) for key, value in train.items()}
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for indices in torch.randperm(len(tensors["img"]), generator=generator).split(batch_size):
            if len(indices) < 2:
                continue
            image = functional.normalize(projector(tensors["img"][indices]), dim=-1)
            loss = 0
            for language in ("ti", "te"):
                text = functional.normalize(projector(tensors[language][indices]), dim=-1)
                logits = image @ text.T / 0.07
                labels = torch.arange(len(indices))
                loss = loss + (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)) / 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return projector.eval(), sum(parameter.numel() for parameter in projector.parameters())


def transform_embeddings(projector, embeddings: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Terapkan proyektor bersama dan normalisasi ulang tiap modalitas."""
    import torch

    with torch.inference_mode():
        return {key: unit_rows(projector(torch.from_numpy(value)).numpy()) for key, value in embeddings.items()}


def evaluate(embeddings: dict[str, np.ndarray], variant: str) -> list[dict[str, object]]:
    """Hitung empat arah memakai fungsi metrik M1 tanpa mengubah baseline."""
    image, id_text, en_text = (embeddings[key] for key in ("img", "ti", "te"))
    if image.shape != id_text.shape or image.shape != en_text.shape:
        raise ValueError("Embedding val harus berukuran sama pada empat arah")
    pairs = ((image, id_text), (image, en_text), (id_text, image), (en_text, image))
    return [{"varian": variant, "arah": direction, **metrik(query, gallery)[0]}
            for direction, (query, gallery) in zip(ARAH, pairs)]


def select_unique_train(table: pd.DataFrame) -> np.ndarray:
    """Pilih pasangan train unik untuk mengurangi negatif palsu in-batch."""
    used_images, used_id, used_en = set(), set(), set()
    selected = []
    for index, row in table.reset_index(drop=True).iterrows():
        image = row["file"]
        caption_id = row["caption_id"].strip().casefold()
        caption_en = row["caption_en"].strip().casefold()
        if image in used_images or caption_id in used_id or caption_en in used_en:
            continue
        selected.append(index)
        used_images.add(image)
        used_id.add(caption_id)
        used_en.add(caption_en)
    if len(selected) < 2:
        raise ValueError("Kurang dari dua pasangan train unik untuk loss kontrastif")
    return np.asarray(selected, dtype=np.int64)


def compare_baseline(rows: list[dict[str, object]], baseline_path: Path) -> pd.DataFrame:
    """Gabung tabel M1 baca-saja, selisih R@1, dan gap tiap varian."""
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    baseline = pd.read_csv(baseline_path, index_col=0)
    metrics = ("Recall@1", "Recall@5", "Recall@10", "mAP@10")
    if set(baseline.index) != set(ARAH) or not set(metrics).issubset(baseline.columns):
        raise ValueError("Format tabel M1 tidak cocok dengan empat arah M2")
    baseline_rows = [{"varian": "B0", "arah": direction,
                      **{metric: float(baseline.loc[direction, metric]) for metric in metrics}}
                     for direction in ARAH]
    result = pd.DataFrame(baseline_rows + rows)
    result["delta_R1_vs_B0"] = result.apply(
        lambda row: row["Recall@1"] - float(baseline.loc[row["arah"], "Recall@1"]), axis=1)
    for variant in result["varian"].unique():
        subset = result[result["varian"] == variant].set_index("arah")
        for language, english, name in ((ARAH[0], ARAH[1], "gambar->teks"),
                                        (ARAH[2], ARAH[3], "teks->gambar")):
            result.loc[(result["varian"] == variant) & (result["arah"] == language),
                       "gap_EN_minus_ID_R1"] = subset.loc[english, "Recall@1"] - subset.loc[language, "Recall@1"]
    return result


def main() -> None:
    """CLI eksplisit; tidak mengunduh, melatih, atau mengevaluasi saat import."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--e1-revision", required=True)
    parser.add_argument("--e2-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=("E1", "E2"), default=["E1", "E2"])
    parser.add_argument("--skip-projection", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Folder --out harus baru/kosong agar hasil lama tidak tertimpa")
    train, val = (load_split(args.data, split) for split in ("train", "val"))
    if set(train["file"]) & set(val["file"]):
        raise ValueError("Gambar train/val beririsan")
    # Validasi format M1 sebelum unduhan besar; tabel B0 tidak pernah ditulis ulang.
    compare_baseline([], args.baseline)
    train_indices = select_unique_train(train)
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, costs = [], []
    for code, revision in (("E1", args.e1_revision), ("E2", args.e2_revision)):
        if code not in args.models:
            continue
        started = time.perf_counter()
        val_emb, frozen_count = encode_split(code, revision, val, args.batch_size)
        embed_val_seconds = time.perf_counter() - started
        all_rows.extend(evaluate(val_emb, code))
        costs.append({"varian": code, "embedding_val_s": embed_val_seconds,
                      "frozen_parameters": frozen_count, "trained_parameters": 0})
        if code == "E1" and not args.skip_projection:
            started = time.perf_counter()
            train_emb, _ = encode_split(code, revision, train, args.batch_size)
            embed_train_seconds = time.perf_counter() - started
            unique_train = {key: value[train_indices] for key, value in train_emb.items()}
            for kind, label in (("linear", "E1+P-L"), ("mlp", "E1+P-M")):
                started = time.perf_counter()
                projector, count = fit_projection(unique_train, kind, args.seed, args.epochs, args.batch_size, args.lr)
                training_seconds = time.perf_counter() - started
                all_rows.extend(evaluate(transform_embeddings(projector, val_emb), label))
                costs.append({"varian": label, "embedding_val_s": embed_val_seconds,
                              "embedding_train_s": embed_train_seconds, "training_s": training_seconds,
                              "frozen_parameters": frozen_count, "trained_parameters": count})
        # Simpan hasil yang sudah selesai agar kegagalan model berikutnya tidak menghilangkannya.
        compare_baseline(all_rows, args.baseline).to_csv(args.out / "m2_val.csv", index=False)
        (args.out / "costs.json").write_text(json.dumps(costs, indent=2), encoding="utf-8")
        (args.out / "manifest.json").write_text(json.dumps({"models": MODEL_IDS, "revisions": {
            "E1": args.e1_revision, "E2": args.e2_revision}, "completed_models": [item["varian"] for item in costs],
            "seed": args.seed, "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "cpu": os.cpu_count(), "train_rows": len(train), "train_unique_rows": len(train_indices),
            "val_rows": len(val)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
