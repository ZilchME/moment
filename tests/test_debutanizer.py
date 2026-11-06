"""Offline adaptation checks, authored without executing them during implementation."""

from argparse import Namespace

import numpy as np
import pytest
import torch

from momentfm.data.debutanizer import (
    CAUSAL13_NAMES, DebutanizerWindows, Standardizer, chronological_rows,
    load_debutanizer, make_features,
)
from momentfm.models.soft_sensor import MOMENTSoftSensor
from scripts.check_environment import tiny_model_config
from scripts.debutanizer import regression_metrics, restore_checkpoint, save_checkpoint


@pytest.fixture
def raw():
    return np.arange(24 * 8, dtype=np.float64).reshape(24, 8) / 200


def test_original_sae_target_boundaries():
    splits = chronological_rows(2394)
    assert [(len(rows), int(rows[0]), int(rows[-1])) for rows in splits.values()] == [
        (800, 4, 803), (200, 804, 1003), (1390, 1004, 2393),
    ]


def test_causal_features_do_not_see_current_or_future_y(raw):
    features, names = make_features(raw, "causal13")
    assert names == CAUSAL13_NAMES
    assert np.isnan(features[:4]).all()
    np.testing.assert_array_equal(features[10, -4:], raw[[9, 8, 7, 6], 7])
    changed = raw.copy()
    changed[10, 7] = 999
    changed[11:] = 999
    updated, _ = make_features(changed, "causal13")
    np.testing.assert_array_equal(features[:11], updated[:11])
    process, _ = make_features(raw, "process")
    changed[:, 7] = 999
    np.testing.assert_array_equal(process[:11], make_features(changed, "process")[0][:11])


def test_windows_preserve_causality_and_padding(raw):
    x, _ = make_features(raw, "process")
    data = DebutanizerWindows(x, raw[:, 7], np.array([4, 10]), context_length=8)
    sample = data[0]
    assert sample["x"].shape == (7, 512)
    assert sample["input_mask"].sum() == 8
    assert sample["observed_mask"].sum() == 5
    np.testing.assert_allclose(sample["x"][:, -5:], x[:5].T)
    np.testing.assert_allclose(sample["x"][:, -8:-5], np.repeat(x[0, :, None], 3, axis=1))
    assert not sample["x"][:, :-8].any()
    changed = x.copy()
    changed[5:] = 999
    updated = DebutanizerWindows(changed, raw[:, 7], np.array([4]), context_length=8)[0]
    np.testing.assert_array_equal(sample["x"], updated["x"])
    # Validation may use earlier training observations, never later observations.
    np.testing.assert_allclose(data[1]["x"][:, -8:], x[3:11].T)


def test_unknown_y_and_missing_history_do_not_become_inputs(tmp_path, raw):
    path = tmp_path / "data.csv"
    rows = [",".join(map(str, row)) for row in raw[:6]]
    rows[-1] = ",".join(map(str, raw[5, :7])) + ","
    path.write_text("u1,u2,u3,u4,u5,u6,u7,y\n" + "\n".join(rows), encoding="utf-8")
    loaded = load_debutanizer(path, allow_missing_y=True)
    assert np.isnan(loaded[5, 7])
    x, _ = make_features(loaded, "causal13")
    assert np.isfinite(x[5]).all()
    with pytest.raises(ValueError, match="finite y"):
        load_debutanizer(path)
    loaded[4, 7] = np.nan
    x, _ = make_features(loaded, "causal13")
    with pytest.raises(ValueError, match="missing"):
        DebutanizerWindows(x, loaded[:, 7], np.array([5]), context_length=8)


def test_training_scaler_and_constant_column():
    scaler = Standardizer.fit(np.array([[1.0, 2.0], [3.0, 2.0]]))
    np.testing.assert_array_equal(scaler.mean, [2.0, 2.0])
    np.testing.assert_array_equal(scaler.scale, [1.0, 1.0])
    np.testing.assert_array_equal(scaler.transform(np.array([[100.0, 2.0]])), [[98.0, 0.0]])
    restored = Standardizer.from_state(scaler.state())
    np.testing.assert_array_equal(restored.mean, scaler.mean)


@pytest.mark.parametrize("mode", ["probe", "finetune"])
def test_trainable_parameters_and_channel_preserving_head(mode):
    model = MOMENTSoftSensor.from_config(tiny_model_config(), 7, mode=mode)
    model.train()
    x = torch.randn(2, 7, 512)
    mask = torch.zeros(2, 512, dtype=torch.long)
    mask[:, -8:] = 1
    prediction = model(x, mask, mask)
    assert prediction.shape == (2,)
    assert model.head.in_features == 7 * (16 + 3)
    prediction.square().mean().backward()
    assert model.head.weight.grad is not None
    if mode == "probe":
        assert not model.backbone.training
        assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    else:
        assert any(p.grad is not None for p in model.backbone.encoder.parameters())


def test_statistics_use_only_real_observations(raw):
    x, _ = make_features(raw, "process")
    sample = DebutanizerWindows(x, raw[:, 7], np.array([4]), context_length=8)[0]
    batch = {key: torch.from_numpy(sample[key]).unsqueeze(0)
             for key in ("x", "input_mask", "observed_mask")}
    model = MOMENTSoftSensor.from_config(tiny_model_config(), 7).eval()
    with torch.no_grad():
        encoded = model.encode(**batch).numpy()
    expected = np.concatenate([x[:5].mean(axis=0), x[:5].std(axis=0), x[4]])
    np.testing.assert_allclose(encoded[0, -21:], expected, atol=1e-6)


def test_checkpoint_is_self_contained(tmp_path):
    model = MOMENTSoftSensor.from_config(tiny_model_config(), 7).eval()
    scaler = Standardizer.fit(np.array([[0.0], [1.0]]))
    x_scaler = Standardizer.fit(np.array([[0.0] * 7, [1.0] * 7]))
    path = tmp_path / "best.pt"
    args = Namespace(mode="probe", no_window_stats=False, data=tmp_path / "data.csv")
    save_checkpoint(path, model, args, x_scaler, scaler, tuple(f"u{i}" for i in range(1, 8)), 1, 0.5, "fixture")
    loaded, saved, _, _ = restore_checkpoint(path, torch.device("cpu"))
    assert saved["best_epoch"] == 1
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[name])
    x = torch.randn(1, 7, 512)
    mask = torch.zeros(1, 512, dtype=torch.long)
    mask[:, -8:] = 1
    with torch.no_grad():
        torch.testing.assert_close(model(x, mask, mask), loaded(x, mask, mask))


def test_regression_metrics():
    scores = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert scores["rmse"] == 0 and scores["r2"] == 1
    assert regression_metrics([1.0, 1.0], [1.0, 1.0])["r2"] is None
