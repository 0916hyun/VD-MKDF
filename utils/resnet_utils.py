import torch.nn as nn

def make_resnet_encoder_blocks(encoder: nn.Module):
    """
    ResNet encoder를 'stem'과 'stages'로 분리하여 반환합니다.

    Args:
        encoder: segmentation_models_pytorch의 ResNetEncoder 인스턴스

    Returns:
        stem: nn.Sequential(conv1, bn1, relu, maxpool)
        stages: list of (layer1, layer2, layer3, layer4)
    """
    # 초기 stem: conv1 -> bn1 -> relu -> maxpool
    stem = nn.Sequential(
        encoder.conv1,
        encoder.bn1,
        encoder.relu,     # <— 변경: act -> relu
        encoder.maxpool,  # <— 변경: pool -> maxpool
    )
    # ResNet의 주요 블록들
    stages = [
        encoder.layer1,
        encoder.layer2,
        encoder.layer3,
        encoder.layer4,
    ]
    return stem, stages