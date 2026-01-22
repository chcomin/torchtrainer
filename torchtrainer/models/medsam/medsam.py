"""MedSAM model definition and loading function."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .segment_anything import sam_model_registry

MEDSAM_CKPT = "medsam_vit_b.pth"


class MedSAM(nn.Module):
    """MedSAM model for medical image segmentation."""

    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        img_size: tuple[int, int] = (1024, 1024),
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
    
    """
    def parameters(
            self, 
            include_prompt: bool = False,
            recurse: bool = True,
            ) -> Iterator[Parameter]:
        
        components = [self.mask_decoder]
        if not self.frozen_image_encoder:
            components.append(self.image_encoder)
        if include_prompt:
            components.append(self.prompt_encoder)

        for component in components:
            yield from component.parameters(recurse=recurse)

    def named_parameters(
        self, 
        include_prompt: bool = False, 
        prefix: str = "", 
        recurse: bool = True, 
        remove_duplicate: bool = True
    ) -> Iterator[(str, Parameter)]:
        
        components = [self.mask_decoder]
        if not self.frozen_image_encoder:
            components.append(self.image_encoder)
        if include_prompt:
            components.append(self.prompt_encoder)

        for component in components:
            yield from component.named_parameters(
                prefix=prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate
            )
    """

def get_model(img_size = (1024, 1024), freeze_image_encoder = False) -> MedSAM:
    """Get a MedSAM model with a ViT-B image encoder.
    The model is initialized with pretrained weights on the SA-1B dataset.

    Parameters
    ----------
    img_size
        Input image size. Default is (1024, 1024).
    freeze_image_encoder
        Whether to freeze the image encoder weights during training. Default is False.

    Returns:
    -------
      medsam_model: The model.
    """

    script_directory = Path(__file__).parent.resolve()
    ckpt_path = script_directory / MEDSAM_CKPT

    if not ckpt_path.exists():
        raise FileNotFoundError(
            "MedSAM checkpoint not found. Please download file medsam_vit_b.pth from "
            "https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN "
            f"and place it in {script_directory}"
        )

    sam_model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        img_size=img_size,
        freeze_image_encoder=freeze_image_encoder,
    )

    return medsam_model
