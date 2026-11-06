"""Offline end-to-end check for training, checkpoint inference and plot artifacts."""

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from momentfm.models.soft_sensor import MOMENTSoftSensor
from scripts import debutanizer
from scripts.check_environment import tiny_model_config


class TrainingArtifactsTest(unittest.TestCase):
    def test_training_reloads_checkpoint_and_saves_plots(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / "data.csv"
            raw = np.random.default_rng(42).normal(size=(24, 8))
            np.savetxt(data, raw, delimiter=",", header="u1,u2,u3,u4,u5,u6,u7,y", comments="")
            args = Namespace(
                data=data, output=directory / "run", first_row=4, train_size=8, val_size=4,
                features="process", mode="probe", context_length=8, seed=42, device="cpu",
                model="offline-tiny-model", no_window_stats=False, revision=None,
                local_files_only=True, gradient_checkpointing=False, batch_size=4,
                workers=0, lr=1e-3, backbone_lr=1e-5, weight_decay=0.01,
                epochs=1, patience=1,
            )
            with patch.object(MOMENTSoftSensor, "from_pretrained", side_effect=lambda *a, **kw:
                              MOMENTSoftSensor.from_config(tiny_model_config(), 7)), \
                 patch.object(debutanizer, "restore_checkpoint", wraps=debutanizer.restore_checkpoint) as restore:
                debutanizer.train(args)
                restore.assert_called_once()

            validation = args.output / "inference_validation"
            consistency = json.loads((validation / "consistency.json").read_text())
            self.assertTrue(consistency["checkpoint_reloaded"])
            self.assertTrue(consistency["matches_training_evaluation"])
            self.assertEqual(consistency["n_predictions"], 8)
            original = json.loads((args.output / "metrics.json").read_text())
            reloaded = json.loads((validation / "metrics.json").read_text())
            self.assertEqual(original["metrics"], reloaded["metrics"])
            with (validation / "predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["sample_index"]) for row in rows], list(range(16, 24)))
            np.testing.assert_allclose([float(row["y_true"]) for row in rows], raw[16:, 7])
            for image in [args.output / "learning_curve.png", *[
                validation / name for name in (
                    "prediction_comparison.png", "prediction_scatter.png", "residuals.png")
            ]]:
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertGreater(image.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
