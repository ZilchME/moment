"""Report the runtime; optionally run a tiny offline forward pass, never training."""

import argparse
import json
import platform
from importlib.metadata import version

import torch

from momentfm import MOMENTPipeline
from momentfm.utils.device import resolve_device


def tiny_model_config():
    """One small T5 layer initialized locally, without downloading any weights."""
    return {
        "task_name": "reconstruction",
        "seq_len": 512,
        "patch_len": 8,
        "patch_stride_len": 8,
        "d_model": 16,
        "transformer_backbone": "google/flan-t5-small",
        "transformer_type": "encoder_only",
        "enable_gradient_checkpointing": False,
        "t5_config": {
            "d_model": 16,
            "d_kv": 8,
            "d_ff": 32,
            "num_layers": 1,
            "num_heads": 2,
            "vocab_size": 32,
            "dropout_rate": 0.0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny random model offline.")
    args = parser.parse_args()
    device = resolve_device(args.device)
    report = {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": str(device),
    }
    if args.smoke_test:
        torch.manual_seed(42)
        model = MOMENTPipeline(tiny_model_config(), model_kwargs={})
        model.init()
        model = model.eval().to(device=device, dtype=torch.float32)
        x = torch.sin(torch.linspace(0, 8 * torch.pi, 512)).reshape(1, 1, 512).to(device)
        with torch.inference_mode():
            output = model(x_enc=x).reconstruction
        if output.shape != x.shape or not torch.isfinite(output).all().item():
            raise RuntimeError("Offline forward pass produced an invalid reconstruction.")
        report["smoke_test"] = {
            "status": "passed",
            "output_shape": list(output.shape),
            "pretrained": False,
            "training": False,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
