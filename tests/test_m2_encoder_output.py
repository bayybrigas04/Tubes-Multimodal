"""Kontrak embedding M2 untuk keluaran Transformers dan OpenCLIP."""

import sys
from pathlib import Path

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from M2_Encoders import pooled_features  # noqa: E402


def test_pooled_features_model_output_returns_pooled_tensor() -> None:
    """Jangan kirim objek keluaran SigLIP 2 ke detach tanpa mengambil pooler."""
    pooled = torch.tensor([[1.0, 2.0]])
    output = BaseModelOutputWithPooling(
        last_hidden_state=torch.zeros(1, 3, 2), pooler_output=pooled
    )
    assert torch.equal(pooled_features(output), pooled)


def test_pooled_features_tensor_passthrough() -> None:
    """Embedding Tensor dari OpenCLIP tetap diterima."""
    values = torch.tensor([[1.0, 2.0]])
    assert torch.equal(pooled_features(values), values)


def test_pooled_features_missing_pooler_raises() -> None:
    """Output tanpa vektor gabungan tidak dipakai untuk retrieval."""
    output = BaseModelOutputWithPooling(last_hidden_state=torch.zeros(1, 3, 2))
    with pytest.raises(ValueError, match="pooler_output"):
        pooled_features(output)
