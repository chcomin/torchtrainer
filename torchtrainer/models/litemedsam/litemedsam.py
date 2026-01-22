from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..medsam.segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from .tiny_vit_sam import TinyViT

MODEL_CKPT = "lite_medsam.pth"
    
class LiteMedSAM(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                img_size: tuple[int, int] = (256, 256),
                freeze_image_encoder: bool = False,
                ):
        super().__init__()

        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        if freeze_image_encoder:
            self.freeze_image_encoder()
        else:
            self.unfreeze_image_encoder()
    
        for param in prompt_encoder.parameters():
            param.requires_grad = False

        # Precompute embedding for full image box prompt
        bbox = torch.tensor([[0, 0, img_size[1]-1, img_size[0]-1]]).float()
        with torch.no_grad():
            sparse_prompt_embedding, dense_prompt_embedding = prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )

        self.register_buffer("sparse_prompt_embedding", sparse_prompt_embedding) # (1, 2, 256)
        self.register_buffer("dense_prompt_embedding", dense_prompt_embedding) # (1, 256, 64, 64)

        
    def forward(self, images):

        ctx = torch.no_grad() if self.frozen_image_encoder else torch.enable_grad()
        with ctx:
            image_embeddings = self.image_encoder(images)  # (B, 256, 64, 64)

        # Expand precomputed prompt embeddings to batch size
        sparse_prompt_embeddings = self.sparse_prompt_embedding.expand(images.shape[0], -1, -1) 
        dense_prompt_embeddings = self.dense_prompt_embedding.expand(images.shape[0], -1, -1, -1)

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_prompt_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_prompt_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(images.shape[2], images.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """Do cropping and resizing"""
        # Crop
        masks = masks[:, :, :new_size[0], :new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
    
    def freeze_image_encoder(self):
        """Freeze the image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.frozen_image_encoder = True

    def unfreeze_image_encoder(self):
        """Unfreeze the image encoder parameters."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        self.frozen_image_encoder = False

# %%
medsam_lite_image_encoder = TinyViT(
    img_size=256,
    in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

medsam_lite_mask_decoder = MaskDecoder(
    num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
)

def get_model(img_size = (256, 256), freeze_image_encoder = False) -> LiteMedSAM:
    """Get the LiteMedSAM model.

    Parameters
    ----------
    img_size
        Input image size.
    freeze_image_encoder
        Whether to freeze the image encoder weights during training. Default is False.

    Returns:
    -------
      medsam_lite_model: The model.
    """

    script_directory = Path(__file__).parent.resolve()
    ckpt_path = script_directory / MODEL_CKPT

    if not ckpt_path.exists():
        raise FileNotFoundError(
            "LiteMedSAM checkpoint not found. Please download file lite_medsam.pth from "
            "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN "
            f"and place it in {script_directory}"
        )

    medsam_lite_model = LiteMedSAM(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder,
        img_size=img_size,
        freeze_image_encoder=freeze_image_encoder,
    )

    medsam_lite_ckpt = torch.load(
        ckpt_path,
        map_location="cpu"
    )
    medsam_lite_model.load_state_dict(medsam_lite_ckpt, strict=False)

    return medsam_lite_model

