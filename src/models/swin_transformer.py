from typing import Tuple

import torch
import torch.nn as nn

from einops import rearrange


from src.models.base import ECGClassifier


def create_mask(
    window_size: int, displacement: int, upper_lower: bool, left_right: bool
) -> torch.Tensor:
    """Creates masks for attention scores.

    Args:
        window_size: Size of windows in input tensor (M).
        displacement: Displacement for shifted windows (M//2).
        upper_lower: If True, applies upper_lower mask.
        left_right: If True, applies left_right mask.

    Returns:
        Attention mask of shape (M^2, M^2) where M = window_size.
        Contains 0.0 for allowed attention, -inf for blocked.
    """
    mask = torch.zeros(window_size**2, window_size**2)

    if upper_lower:
        mask[-displacement * window_size :, : -displacement * window_size] = float(
            '-inf'
        )
        mask[: -displacement * window_size, -displacement * window_size :] = float(
            '-inf'
        )
    if left_right:
        mask = rearrange(
            mask, "(h1 w1) (h2 w2) -> h1 w1 h2 w2", h1=window_size, h2=window_size
        )
        mask[:, -displacement:, :, :-displacement] = float("-inf")
        mask[:, :-displacement, :, -displacement:] = float("-inf")
        mask = rearrange(mask, "h1 w1 h2 w2 -> (h1 w1) (h2 w2)")

    return mask


def get_relative_distances(window_size: int) -> torch.Tensor:
    """Creates indexes for relative positions of windows.

    Args:
        window_size: Size of windows in input tensor (M).

    Returns:
        Tensor with distances between different indexes.
        Shape (M^2, M^2, 2), where M is window_size.

        distances[i, j] is the relative position of j from position i.
    """
    indexes = torch.tensor(
        [[x, y] for x in range(window_size) for y in range(window_size)]
    )
    distances = indexes[None, :, :] - indexes[:, None, :]
    return distances


class CyclicShift(nn.Module):
    """Shifts the input for shifted-window attention computation.

    Args:
        displacement: Displacement for shifted windows (M//2).
    """

    def __init__(self, displacement: int) -> None:
        super().__init__()
        self.displacement = displacement

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, shifts=[self.displacement, self.displacement], dims=(1, 2))


class WindowAttention(nn.Module):
    """Computation of WindowAttention for SwinTransformer.

    Args:
        dim: Number of channels of input.
        heads: Number of attention heads.
        shifted: Whether the block is for W-MSA or SW-MSA
        window_size: Size of attention window.
        relative_pos_embedding: Whether to use relative position embedding or positional embedding.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        shifted: bool,
        window_size: int,
        relative_pos_embedding: bool,
    ) -> None:
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim**-0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        # Shifted window setup
        if self.shifted:
            displacement = window_size // 2

            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=True,
                    left_right=False,
                ),
                requires_grad=False,
            )
            self.left_right_mask = nn.Parameter(
                create_mask(
                    window_size=window_size,
                    displacement=displacement,
                    upper_lower=False,
                    left_right=True,
                ),
                requires_grad=False,
            )

        # Linear projection of input
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Relative or positional embedding
        if self.relative_pos_embedding:
            self.relative_indexes = (
                get_relative_distances(window_size) + window_size - 1
            )
            self.pos_embedding = nn.Parameter(
                torch.randn(2 * window_size - 1, 2 * window_size - 1)
            )
        else:
            self.pos_embedding = nn.Parameter(
                torch.randn(window_size**2, window_size**2)
            )

        # Output projection
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies window attention.

        Args:
            x: Input tensor.

        Returns:
            Tensor after application of windowed or shifted window attention.
        """
        # Apply cyclic shifting
        if self.shifted:
            x = self.cyclic_shift(x)

        _, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        # Dividing qkv into different heads
        q, k, v = map(
            lambda t: rearrange(
                t,
                "b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d",
                h=h,
                w_h=self.window_size,
                w_w=self.window_size,
                nw_h=nw_h,
                nw_w=nw_w,
            ),
            qkv,
        )

        # Matmul between q an k^t
        dots = torch.einsum("b h w i d, b h w j d -> b h w i j", q, k) * self.scale

        # Applying relative/positional embedding
        if self.relative_pos_embedding:
            dots += self.pos_embedding[
                self.relative_indexes[:, :, 0], self.relative_indexes[:, :, 1]
            ]
        else:
            dots += self.pos_embedding

        # Applying masks if SW-MSA
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1 :: nw_w] += self.left_right_mask

        # Getting attention scores
        attn = dots.softmax(dim=-1)

        # Multiplying attention scores with values
        out = torch.einsum("b h w i j, b h w j d -> b h w i d", attn, v)

        # Joining heads
        out = rearrange(
            out,
            "b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)",
            h=h,
            w_h=self.window_size,
            w_w=self.window_size,
            nw_h=nw_h,
            nw_w=nw_w,
        )

        # Final linear projection
        out = self.to_out(out)

        # Un-cycling windows if shifted
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    """Entire Swin Block from SwinTransformer with LayerNorm, WindowAttention and MLP.

    Args:
        dim: Number of channels of input.
        heads: Number of attention heads.
        head_dim: Dimensions of attention heads.
        mlp_dim: Internal dimension of MLP block.
        shifted: Whether the block is for W-MSA or SW-MSA
        window_size: Size of attention window.
        relative_pos_embedding: Whether to use relative position embedding or positional embedding.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        mlp_dim: int,
        shifted: bool,
        window_size: int,
        relative_pos_embedding: bool,
    ) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)

        self.attention = WindowAttention(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
            shifted=shifted,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=dim, out_features=mlp_dim),
            nn.GELU(),
            nn.Linear(in_features=mlp_dim, out_features=dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through SwinBlock.

        Inputs pass through a Layer Normalization block before going into the WindowAttention and MLP blocks.
        They are then combined with their inputs to form a residual connection.

        Args:
            x: Input tensor.

        Returns:
            Tensor with output of SwinBlock.
        """
        # Residual connection with attention output
        x = x + self.attention(self.layer_norm_1(x))
        x = x + self.mlp(self.layer_norm_2(x))

        return x


class PatchMerging(nn.Module):
    """Patch Merges input.

    Args:
        in_channels: Number of channels of input.
        out_channels: Number of channels desired for output.
        downscaling_factor: Factor to downsample spatial dimensions in input.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscaling_factor: int,
    ) -> None:
        super().__init__()
        self.downscaling_factor = downscaling_factor
        merged_dim = (downscaling_factor**2) * in_channels

        self.linear = nn.Linear(merged_dim, out_channels, bias=False)
        self.norm = nn.LayerNorm(merged_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies Patch Merging followed by layer norm and a linear projection.

        Args:
            x: Input tensor

        Returns:
            Tensor with patches merged by the downsampling factor.

        """
        x = x.permute(0, 2, 3, 1)
        x = rearrange(
            x,
            "b (h s1) (w s2) c -> b h w (s1 s2 c)",
            s1=self.downscaling_factor,
            s2=self.downscaling_factor,
        )
        x = self.norm(x)
        x = self.linear(x)
        return x


class StageModule(nn.Module):
    """Module for a Stage in the Swin Transformer.

    Args:
        in_channels: Number of channels of input.
        hidden_dimensions: Size of hidden dimension for stage.
        layers: Number of Swin Blocks in model. Must be greater or equal to 2, and divisible by 2.
        downscaling_factor: Factor to downsample spatial dimensions in input.
        num_heads: Number of attention heads.
        window_size: Size of attention window.
        relative_pos_embedding: Whether to use relative position embedding or positional embedding.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dimension: int,
        num_layers: int,
        downscaling_factor: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        relative_pos_embedding: bool,
    ) -> None:
        super().__init__()
        assert (
            num_layers % 2 == 0
        ), "Stage layers need to be divisible by 2 for regular and shifted block."

        self.patch_partition = PatchMerging(
            in_channels=in_channels,
            out_channels=hidden_dimension,
            downscaling_factor=downscaling_factor,
        )

        self.layers = nn.ModuleList([])
        for _ in range(num_layers // 2):
            self.layers.append(
                SwinBlock(
                    dim=hidden_dimension,
                    heads=num_heads,
                    head_dim=head_dim,
                    mlp_dim=hidden_dimension * 4,
                    shifted=False,
                    window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding,
                ),
            )
            self.layers.append(
                SwinBlock(
                    dim=hidden_dimension,
                    heads=num_heads,
                    head_dim=head_dim,
                    mlp_dim=hidden_dimension * 4,
                    shifted=True,
                    window_size=window_size,
                    relative_pos_embedding=relative_pos_embedding,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a stage of SwinTransformer to input.

        Args:
            x: Input tensor.

        Returns:
            Tensor rearranged after passing through patch merging and at least a pair of W-MSA and SW-MSA
        """
        x = self.patch_partition(x)
        for block in self.layers:
            x = block(x)
        return x.permute(0, 3, 1, 2)


class SwinTransformer(ECGClassifier):
    """SwinTransformer with 4 stages.

    Args:
        hidden_dim: Size of hidden dimension for stage.
        layers: Tuple with number of Swin Blocks in model, one for each of the 4 stages.
        heads: Tuple with number of attention heads, one for each of the 4 stages.
        channels: Number of channels of input. Default 3.
        num_classes: Number of classes to predict. Default 2.
        head_dim: Dimensions of attention heads. Default 32.
        window_size: Size of attention window. Default 7.
        downscaling_factors: Factor to downsample spatial dimensions in input. Default `(4, 2, 2, 2)`.
        relative_pos_embedding: Whether to use relative position embedding or positional embedding. Default True.
    """

    def __init__(
        self,
        hidden_dim: int,
        layers: Tuple[int, int, int, int],
        heads: Tuple[int, int, int, int],
        channels: int = 3,
        num_classes: int = 1000,
        head_dim: int = 32,
        window_size: int = 7,
        downscaling_factors: Tuple[int, int, int, int] = (4, 2, 2, 2),
        relative_pos_embedding: bool = True,
        class_names: list[str] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(class_names or ["non-Chagas", "Chagas"])
        self.device = "cpu"
        self.layers = layers

        self.stage_1 = StageModule(
            in_channels=channels,
            hidden_dimension=hidden_dim,
            num_layers=layers[0],
            downscaling_factor=downscaling_factors[0],
            num_heads=heads[0],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage_2 = StageModule(
            in_channels=hidden_dim,
            hidden_dimension=hidden_dim * 2,
            num_layers=layers[1],
            downscaling_factor=downscaling_factors[1],
            num_heads=heads[1],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage_3 = StageModule(
            in_channels=hidden_dim * 2,
            hidden_dimension=hidden_dim * 4,
            num_layers=layers[2],
            downscaling_factor=downscaling_factors[2],
            num_heads=heads[2],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )
        self.stage_4 = StageModule(
            in_channels=hidden_dim * 4,
            hidden_dimension=hidden_dim * 8,
            num_layers=layers[3],
            downscaling_factor=downscaling_factors[3],
            num_heads=heads[3],
            head_dim=head_dim,
            window_size=window_size,
            relative_pos_embedding=relative_pos_embedding,
        )

        active_stages = sum(1 for layer in layers if layer > 0)
        if active_stages == 1:
            final_dim = hidden_dim
        elif active_stages == 2:
            final_dim = hidden_dim * 2
        elif active_stages == 3:
            final_dim = hidden_dim * 4
        else:  # 4 stages
            final_dim = hidden_dim * 8

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(final_dim),  # MODIFIED: Dynamic dimension
            nn.Linear(final_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes image through SwinTransformer Architecture.

        Args:
            x: Input image. Shape: (B, C, H, W)

        Returns:
            Logits of image class. Shape: (B, num_classes)
        """
        x = self.stage_1(x)

        # MODIFIED: Conditional stages (comment out unused stages)
        if self.layers[1]:
            if (
                self.stage_2.patch_partition.downscaling_factor > 1
            ):  # Only if stage 2 active
                x = self.stage_2(x)

        if self.layers[2]:
            x = self.stage_3(x)  # MODIFIED: Commented out (set layers[2]=0)
        if self.layers[3]:
            x = self.stage_4(x)  # MODIFIED: Commented out (set layers[3]=0)

        x = x.mean(dim=[2, 3])  # Global average pooling
        return self.mlp_head(x)
