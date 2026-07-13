from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import ECGClassifier


class LeadGroupBranch(nn.Module):
    def __init__(self, in_channels: int, feature_dim: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.feature_dim = feature_dim

        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.cnn_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.cnn_block4 = nn.Sequential(
            nn.Conv1d(256, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)

        x = self.cnn_block3(x)
        x = self.cnn_block4(x)

        return x.mean(dim=-1)


class GroupedLeadCNNClassifier(ECGClassifier):
    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 2,
        num_classes: int = 2,
        ffn_dropout_p: float = 0.1,
        clf_dropout_p: float = 0.3,
        class_names: list[str] = ["Non-Chagas", "Chagas"],
        device: str = "cpu",
    ) -> None:
        super().__init__(class_names=class_names)

        self.num_classes = num_classes

        self.limb_branch = LeadGroupBranch(
            in_channels=6,
            feature_dim=feature_dim,
        )
        self.r_precordial_branch = LeadGroupBranch(
            in_channels=3,
            feature_dim=feature_dim,
        )
        self.l_precordial_branch = LeadGroupBranch(
            in_channels=3,
            feature_dim=feature_dim,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout_p),
            nn.Linear(feature_dim * 4, feature_dim),
        )

        self.ffn_norm = nn.LayerNorm(feature_dim)

        self.classifier_head = nn.Sequential(
            nn.Linear(3 * feature_dim, 256),
            nn.GELU(),
            nn.Dropout(clf_dropout_p),
            nn.Linear(256, num_classes),
        )

    def _format_data(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {x.shape}")
        if x.shape == (12, 734):
            return x.unsqueeze(0)
        if x.shape == (734, 12):
            return x.T.unsqueeze(0)
        raise ValueError(f"Unexpected shape: {x.shape}")

    def _split_groups(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        limb_group = x[:, :6, :]
        r_prec_group = x[:, 6:9, :]
        l_prec_group = x[:, 9:, :]

        return limb_group, r_prec_group, l_prec_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        limb_group, r_prec_group, l_prec_group = self._split_groups(x)

        f_limb = self.limb_branch(limb_group)
        f_r_precordial = self.r_precordial_branch(r_prec_group)
        f_l_precordial = self.l_precordial_branch(l_prec_group)

        f_grouped = torch.stack([f_limb, f_r_precordial, f_l_precordial], dim=1)

        normed = self.attn_norm(f_grouped)
        attended, _ = self.cross_attn(normed, normed, normed)
        attended = attended + f_grouped

        attended = attended + self.ffn(self.ffn_norm(attended))

        out = attended.flatten(start_dim=1)

        return self.classifier_head(out)
