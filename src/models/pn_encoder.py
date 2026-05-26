"""
Contains PreNormTransformerEncoderLayer, and ECGEncoderClassifier.
"""

import torch
import torch.nn as nn

from src.models.base import ECGClassifier


class PreNormTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with pre-normalization and GELU activation

    Args:
        d_model (int): Size of embedding dimension.
        nhead (int): Number of self-attention heads.
        dim_feedforward (int): Size of dimension in feedforward layer.
        dropout (float, optional): p value for Dropout layer. Defaults to 0.1.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed-forward network with GELU
        self.linear_1 = nn.Linear(d_model, dim_feedforward)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()

        # Layer normalization
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass in Pre-Norm Transformer Encoder

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor | None, optional): Mask for input tensor. Defaults to None.
            src_key_padding_mask (torch.Tensor | None, optional): Mask for tensor's padding. Defaults to None.

        Returns:
            torch.Tensor: Encoder's output.
        """
        # Pre-norm
        src_norm = self.norm_1(src)
        attn_output, _ = self.self_attn(
            src_norm,
            src_norm,
            src_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )

        # Dropout
        src = src + self.dropout_1(attn_output)

        # Pre-norm ff layer
        src_norm = self.norm_2(src)
        ff_output = self.linear_2(
            self.dropout(self.activation(self.linear_1(src_norm)))
        )

        # Dropout
        src = src + self.dropout_2(ff_output)

        return src


class PreNormEncoderClassifier(ECGClassifier):
    """Transformer Encoder for ECG Classification

    Args:
        num_leads (int): Number of leads in ECG signal.
        seq_len (int): Size of ECG sample.
        embed_dim (int): Size of embedding dimension.
        num_layers (int): Number of Encoder layers.
        num_heads (int): Number of heads in Encoder.
        num_classes (int): Number of classes to output in classification
        device (torch.device | str): Accelerator device
        dropout (float, optional): p value for Dropout layer. Defaults to 0.1.
        ff_multiplier (int, optional): Multiplier for feed-forward layer. Defaults to 4.
    """

    def __init__(
        self,
        num_leads: int,
        seq_len: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        device: torch.device | str,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__(class_names or ["non-Chagas", "Chagas"])

        self.device = device
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Token Embedding
        self.tok_emb = nn.Linear(num_leads, embed_dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.zeros_(self.tok_emb.bias)

        # Positional Embedding
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Input normalization
        self.input_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Enhanced transformer encoder layers
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                PreNormTransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * ff_multiplier,
                    dropout=dropout,
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_layers)

        # Final normalization before pooling
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim, num_heads=min(4, num_heads), batch_first=True
        )
        self.cls_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.cls_query, std=0.02)

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.normal_(param, std=0.02)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass in Encoder Classifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Class logits.
        """
        batch_size, seq_len, num_leads = x.shape

        # Token embeddings
        x = self.tok_emb(x)

        # Positional embeddings
        positions = (
            torch.arange(seq_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_emb = self.pos_emb(positions)
        x = x + pos_emb

        # Input normalization
        x = self.input_norm(x)

        # Transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Final normalization
        x = self.final_norm(x)

        # Pooling
        batch_size = x.size(0)
        cls_queries = self.cls_query.expand(batch_size, -1, -1)
        pooled, _ = self.attention_pooling(cls_queries, x, x)
        pooled = pooled.squeeze(1)

        # Classification with dropout
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits
