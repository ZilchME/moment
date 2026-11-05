"""Small dependency/serialization checks; no network, pretrained downloads or training."""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest
import torch

from momentfm import MOMENTPipeline
from momentfm.utils.device import resolve_device
from scripts.check_environment import tiny_model_config


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(
    "task,field,shape,kwargs",
    [
        ("reconstruction", "reconstruction", (1, 2, 512), {}),
        ("embedding", "embeddings", (1, 16), {}),
        ("embedding", "embeddings", (1, 2, 64, 16), {"reduction": "none"}),
        ("forecasting", "forecast", (1, 2, 24), {}),
        ("classification", "logits", (1, 3), {}),
    ],
)
def test_task_forward(task, field, shape, kwargs, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Apple MPS is unavailable")
    model = MOMENTPipeline(
        tiny_model_config(),
        model_kwargs={"task_name": task, "n_channels": 2, "num_class": 3, "forecast_horizon": 24},
    )
    model.init()
    model.eval().to(device)
    with torch.inference_mode():
        output = getattr(model(x_enc=torch.randn(1, 2, 512).to(device), **kwargs), field)
    assert tuple(output.shape) == shape
    assert torch.isfinite(output).all().item()


def test_local_pretrained_roundtrip(tmp_path):
    model = MOMENTPipeline(tiny_model_config(), model_kwargs={}).eval()
    model.save_pretrained(tmp_path)
    loaded = MOMENTPipeline.from_pretrained(str(tmp_path), local_files_only=True).eval()
    x = torch.randn(1, 1, 512)
    with torch.inference_mode():
        torch.testing.assert_close(model(x_enc=x).reconstruction, loaded(x_enc=x).reconstruction)


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device().type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device().type == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device().type == "cuda"
