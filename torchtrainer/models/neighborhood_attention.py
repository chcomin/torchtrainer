"""Local attention using the natten library"""
import logging

from natten import NeighborhoodAttention2D, use_fused_na
from natten.types import CausalArg2DTypeOrDed, Dimension2DTypeOrDed

# Suppress warnings from natten
logging.getLogger("natten.context").setLevel(logging.ERROR)

#kv_parallel=True uses more memory but is faster
#use_fused_na(True, kv_parallel=True, use_flex_attention=False)
use_fused_na(False)

class NeighborhoodAttention(NeighborhoodAttention2D):
    """Adapter class for NeighborhoodAttention2D to work with 2D sequences that have already
    been transformed to 1D sequences.
    """

    def __init__(
        self,
        num_patches: tuple[int, int] | int,
        dim: int,
        num_heads: int,
        kernel_size: Dimension2DTypeOrDed,
        dilation: Dimension2DTypeOrDed = 1,
        is_causal: CausalArg2DTypeOrDed = False,
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_experimental_ops: bool = False,
    ):
        """The parameters are the same as for NeighborhoodAttention2D, except for num_patches.
        
        Parameters
        ----------
        num_patches
            Number of row and column patches in the input sequence. This is usually given by 
            image_size // patch_size. If an integer is provided, the same value is used for both
            dimensions.
        ...
        """

        super().__init__(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_experimental_ops=use_experimental_ops,
        )

        if isinstance(num_patches, int):
            num_patches = (num_patches, num_patches)

        self.num_patches = num_patches

    def forward(self, x):

        num_patches_r, num_patches_c = self.num_patches

        bs, seq_len, dim = x.shape
        if seq_len != num_patches_r * num_patches_c:
            raise ValueError(
                f"Input sequence length {seq_len} does not match expected "
                f"{num_patches_r} * {num_patches_c}"
            )

        # Reshape the input to match the expected input shape of NeighborhoodAttention2D
        # (bs, seq_len, dim) -> (bs, num_patches_r, num_patches_c, dim)
        x = x.permute(0, 2, 1).reshape(bs, dim, num_patches_r, num_patches_c).permute(0, 2, 3, 1)
        x = super().forward(x)
        # Reshape the output back to (bs, seq_len, dim)
        x = x.permute(0, 3, 1, 2).reshape(bs, dim, -1).permute(0, 2, 1)
        
        return x
