import torch
import torch.nn as nn

from models.smp_unet_student import SMPUnetKDWrapper

try:
    import segmentation_models_pytorch as smp  # 존재 확인용(선택)
except Exception:
    smp = None

from transformers import SegformerForSemanticSegmentation
from models.SegFormerBGBD_final import Segb5_RGBD_FuseStageSelective_EncUpdate


class SegformerWrapper(nn.Module):
    def __init__(self, hf_segformer_model):
        super().__init__()
        self.model = hf_segformer_model

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, dict):
            return outputs["logits"]
        return outputs


def make_model(
    model_name: str,
    encoder_name: str | None = None,
    encoder_weights: str | None = None,
    in_channels: int | None = None,
    depth_channels: int | None = None,
    n_classes: int | None = None,
    patch_sizes: list[int] | None = None,
    fuse_stages: list[int] | None = None,
    attn_variant: str = 'base',
    cp_rank: int = 8,
    device: str = 'cuda',
    **kwargs
) -> nn.Module:
    name = model_name.lower()

    if model_name.lower() in ("smp_unet_student_kd", "unet_student_kd"):
        if n_classes is None:
            raise ValueError("`n_classes` must be specified for smp_unet_student_kd.")
        if smp is None:
            raise ImportError("segmentation_models_pytorch is not installed.")
        encoder_name = kwargs.get("encoder_name", "resnet50")
        encoder_weights = kwargs.get("encoder_weights", "imagenet")  # None or "imagenet"
        model = SMPUnetKDWrapper(
            num_classes=n_classes,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights
        )
        return model

    elif model_name == 'segb5_rgbd_fuse_stageselective_enc_update':
        assert n_classes is not None, "n_classes를 지정해야 합니다."
        assert patch_sizes is not None, "patch_sizes를 지정해야 합니다."
        assert fuse_stages is not None, "fuse_stages를 지정해야 합니다."

        model = Segb5_RGBD_FuseStageSelective_EncUpdate(
            n_classes     = n_classes,
            patch_sizes   = patch_sizes,
            fuse_stages   = fuse_stages,
        )
        try:
            model.fuse_stages_list = list(fuse_stages)
        except Exception:
            pass 

    elif name == 'segformer_b5':
        if n_classes is None:
            raise ValueError("`n_classes` must be specified for SegFormer B5.")
        hf = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            num_labels=n_classes,
            ignore_mismatched_sizes=True
        )
        model = SegformerWrapper(hf)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)
