import torch
import torch.nn as nn

from src.models.base import ECGClassifier


class DynamicPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        dropout: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        position = torch.arange(
            0,
            T,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, D, 2, dtype=torch.float, device=self.device)
            * (-torch.log(torch.tensor(10000.0, device=self.device)) / D)
        )

        pe = torch.zeros(T, D, device=self.device)

        pe[:, 0::2] = torch.sin(position * div_term)
        if D % 2:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        x = x + pe.unsqueeze(0)

        return self.dropout(x)


class MultiScaleCNNEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 128,
        num_blocks: int = 3,
        downsample_factor: int = 2,
        kernel_sizes: list[int] = [3, 7, 15],
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        num_downsamples = {1: 0, 2: 1, 4: 2}[downsample_factor]

        if downsample_factor not in [1, 2, 4]:
            raise ValueError("downsample_factor must be 1, 2, or 4.")

        if num_blocks < num_downsamples:
            raise ValueError(
                f"num_blocks={num_blocks} must be >= {num_downsamples} "
                f"for downsample_factor={downsample_factor}"
            )

        pool_strides = [2] * num_downsamples + [1] * (num_blocks - num_downsamples)

        num_channels = len(kernel_sizes)
        base = feature_dim // num_channels
        remainder = feature_dim % num_channels
        branch_channels = [
            base + (1 if i < remainder else 0) for i in range(num_channels)
        ]

        self.branches = nn.ModuleList()

        for branch_idx, kernel_size in enumerate(kernel_sizes):
            blocks = []
            current_channels = in_channels

            for stride in pool_strides:
                out_channels = branch_channels[branch_idx]
                blocks.append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=current_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            stride=1,
                        ),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.MaxPool1d(kernel_size=2, stride=stride),
                    )
                )
                current_channels = out_channels

            self.branches.append(nn.Sequential(*blocks))

        self.fusion = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = []

        for branch in self.branches:
            branch_outputs.append(branch(x))

        concat = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(concat)

        return fused


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionAttention(nn.Module):
    def __init__(self, d_model: int = 128) -> None:
        super().__init__()

        self.attention_weights = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, cnn_features: torch.Tensor, transformer_features: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([cnn_features, transformer_features], dim=-1)
        weights = self.attention_weights(combined)

        fused = (
            weights[:, :, 0:1] * cnn_features
            + weights[:, :, 1:2] * transformer_features
        )

        return fused


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)

        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled


class CNNBertClassifier(ECGClassifier):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 8,
        num_leads: int = 12,
        num_classes: int = 2,
        num_cnn_blocks: int = 3,
        num_transformer_layers: int = 4,
        cnn_downsample_factor: int = 2,
        device: str | torch.device = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__(class_names or ["Normal", "Chagas", "Structural"])
        self.device = device

        self.cnn_encoder = MultiScaleCNNEncoder(
            in_channels=num_leads,
            feature_dim=d_model,
            num_blocks=num_cnn_blocks,
            downsample_factor=cnn_downsample_factor,
        )

        self.positional_encoding = DynamicPositionalEncoding(
            d_model=d_model,
            dropout=0.1,
            device=device,
        )

        self.transformer_encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            num_layers=num_transformer_layers,
        )

        self.fusion_attention = FusionAttention(d_model=d_model)

        self.attention_pool = AttentionPooling(d_model=d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def _format_data(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        if x.ndim != 2:
            raise ValueError(f"Expected 2D ECG tensor, got {x.shape}")

        if x.shape == (12, 734):
            return x.unsqueeze(0)

        if x.shape == (734, 12):
            return x.T.unsqueeze(0)

        raise ValueError(f"Unexpected ECG shape: {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn_encoder(x)
        cnn_out = cnn_out.transpose(1, 2)

        cnn_features = cnn_out.clone()

        cnn_out = self.positional_encoding(cnn_out)

        transformer_out = self.transformer_encoder(cnn_out)

        fused = self.fusion_attention(cnn_features, transformer_out)

        attn_pooled = self.attention_pool(fused)
        max_pooled = torch.max(fused, dim=1)[0]

        combined = torch.cat([attn_pooled, max_pooled], dim=1)

        logits = self.classifier(combined)

        return logits
