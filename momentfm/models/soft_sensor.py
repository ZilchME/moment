"""MOMENT features plus a channel-preserving scalar regression head."""

from contextlib import nullcontext
from copy import deepcopy

import torch
from torch import nn

from momentfm.models.moment import MOMENTPipeline
from momentfm.utils.masking import Masking


class MOMENTSoftSensor(nn.Module):
    def __init__(self, backbone: MOMENTPipeline, n_channels: int, *,
                 mode: str = "probe", window_stats: bool = True):
        super().__init__()
        if mode not in {"probe", "finetune"} or n_channels < 1:
            raise ValueError("Expected probe/finetune mode and a positive channel count.")
        if backbone.task_name != "embedding":
            raise ValueError("Soft sensing requires an initialized embedding backbone.")
        self.backbone = backbone
        self.n_channels = n_channels
        self.frozen = mode == "probe"
        self.window_stats = window_stats
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self.frozen)
        # Mean pooling is over patches, not channels. Keep each variable distinct.
        feature_dim = n_channels * backbone.config.d_model
        if window_stats:
            feature_dim += 3 * n_channels
        self.head = nn.Linear(feature_dim, 1)

    @classmethod
    def from_pretrained(cls, model_id: str, n_channels: int, *, mode: str = "probe",
                        window_stats: bool = True, revision: str | None = None,
                        local_files_only: bool = False, gradient_checkpointing: bool = False):
        kwargs = {
            "model_kwargs": {
                "task_name": "embedding",
                "freeze_encoder": mode == "probe",
                "freeze_embedder": mode == "probe",
                "enable_gradient_checkpointing": gradient_checkpointing and mode == "finetune",
            },
            "local_files_only": local_files_only,
        }
        if revision is not None:
            kwargs["revision"] = revision
        backbone = MOMENTPipeline.from_pretrained(model_id, **kwargs)
        backbone.init()
        return cls(backbone, n_channels, mode=mode, window_stats=window_stats)

    @classmethod
    def from_config(cls, config: dict, n_channels: int, *, mode: str = "probe",
                    window_stats: bool = True):
        """Rebuild locally from a saved config; no Hub requests or base checkpoint needed."""
        backbone = MOMENTPipeline(deepcopy(config), model_kwargs={"task_name": "embedding"})
        backbone.init()
        return cls(backbone, n_channels, mode=mode, window_stats=window_stats)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            # A frozen backbone must also disable dropout during head training/caching.
            self.backbone.eval()
        return self

    def encode(self, x: torch.Tensor, input_mask: torch.Tensor,
               observed_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError("Expected [batch, n_channels, context] input.")
        gradient_context = torch.no_grad() if self.frozen else nullcontext()
        with gradient_context:
            tokens = self.backbone(x_enc=x, input_mask=input_mask, reduction="none").embeddings
        patch_mask = Masking.convert_seq_to_patch_view(input_mask, self.backbone.patch_len)
        weights = patch_mask[:, None, :, None].to(tokens.dtype)
        pooled = (tokens * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
        features = pooled.flatten(start_dim=1)
        if self.window_stats:
            # MOMENT's RevIN removes each window's absolute level and scale. Retain
            # these for soft sensing, using real observations rather than pad values.
            observed = observed_mask[:, None, :].to(x.dtype)
            count = observed.sum(dim=-1).clamp_min(1)
            mean = (x * observed).sum(dim=-1) / count
            variance = ((x - mean.unsqueeze(-1)).square() * observed).sum(dim=-1) / count
            std = variance.clamp_min(0).sqrt()
            features = torch.cat((features, mean, std, x[:, :, -1]), dim=-1)
        return features

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor,
                observed_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x, input_mask, observed_mask)).squeeze(-1)
