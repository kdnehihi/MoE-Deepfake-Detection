"""Vision Transformer backbone wrapper used by the detector."""

from __future__ import annotations

from torch import Tensor, nn

from utils.config import BackboneConfig


class FrozenViTBackbone(nn.Module):
    """
    Frozen ViT wrapper built on top of timm.

    The backbone exposes token-level features so later steps can insert
    parameter-efficient expert modules around the transformer blocks while
    keeping the pretrained ViT weights frozen.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.model = self._build_model(config)
        self.embed_dim = getattr(self.model, "num_features", config.embed_dim)
        if config.freeze:
            self.freeze_parameters()

    @staticmethod
    def _build_model(config: BackboneConfig) -> nn.Module:
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "timm is required to build the ViT backbone. Install it before running Step 2."
            ) from error

        model = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            img_size=config.image_size,
            num_classes=0,
        )
        return model

    def freeze_parameters(self) -> None:
        """Freeze all backbone weights while preserving feature gradients."""
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

    def embed_patches(self, images: Tensor) -> Tensor:
        """Convert images into ViT tokens before the transformer blocks."""
        tokens = self.model.patch_embed(images)
        if hasattr(self.model, "_pos_embed"):
            tokens = self.model._pos_embed(tokens)
        if hasattr(self.model, "patch_drop"):
            tokens = self.model.patch_drop(tokens)
        if hasattr(self.model, "norm_pre"):
            tokens = self.model.norm_pre(tokens)
        return tokens

    def forward_blocks(self, tokens: Tensor) -> Tensor:
        """Run the frozen pretrained transformer blocks."""
        if hasattr(self.model, "blocks"):
            tokens = self.model.blocks(tokens)
        if hasattr(self.model, "norm"):
            tokens = self.model.norm(tokens)
        return tokens

    def forward_features(self, images: Tensor) -> Tensor:
        tokens = self.embed_patches(images)
        return self.forward_blocks(tokens)

    def forward(self, images: Tensor) -> Tensor:
        return self.forward_features(images)
