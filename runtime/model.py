from __future__ import annotations
from typing import Any
import torch
from torch import nn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import nms

FDI_NUMBERS = tuple(
    quadrant * 10 + position
    for quadrant in (1, 2, 3, 4)
    for position in range(1, 9)
)


FDI_TO_CLASS = {fdi: index + 1 for index, fdi in enumerate(FDI_NUMBERS)}


CLASS_TO_FDI = {class_index: fdi for fdi, class_index in FDI_TO_CLASS.items()}


def build_instance_model(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 512,
    max_size: int = 1024,
    num_classes: int = 2,
) -> nn.Module:
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=trainable_backbone_layers if pretrained else None,
    )
    layers = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_backbone_layers]
    if trainable_backbone_layers == 5:
        layers.append("bn1")
    for name, parameter in model.backbone.body.named_parameters():
        parameter.requires_grad = any(name.startswith(layer) for layer in layers)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def prepare_prediction(
    prediction: dict[str, torch.Tensor], score_threshold: float = 0.05
) -> dict[str, torch.Tensor]:
    keep = prediction["scores"].detach().cpu() >= score_threshold
    result = {
        "scores": prediction["scores"].detach().cpu()[keep],
        "boxes": prediction["boxes"].detach().cpu()[keep],
        "masks": (prediction["masks"].detach().cpu()[keep, 0] >= 0.5),
    }
    if "labels" in prediction:
        result["labels"] = prediction["labels"].detach().cpu()[keep]
    return result


def postprocess_fdi_prediction(
    prediction: dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    nms_threshold: float | None = 0.5,
    unique_fdi: bool = True,
) -> dict[str, torch.Tensor]:
    """Elimina duplicados entre clases y limita cada número FDI a una aparición."""
    result = prepare_prediction(prediction, score_threshold)
    if len(result["scores"]) == 0:
        return result
    keep = torch.arange(len(result["scores"]), dtype=torch.int64)
    if nms_threshold is not None:
        keep = keep[nms(result["boxes"], result["scores"], nms_threshold)]
    if unique_fdi:
        ordered = keep[torch.argsort(result["scores"][keep], descending=True)]
        seen: set[int] = set()
        unique: list[int] = []
        for index in ordered.tolist():
            label = int(result["labels"][index])
            if label not in seen:
                seen.add(label)
                unique.append(index)
        keep = torch.tensor(unique, dtype=torch.int64)
    return {key: value[keep] for key, value in result.items()}


def checkpoint_model_args(checkpoint: dict[str, Any]) -> dict[str, Any]:
    saved = checkpoint.get("args", {})
    return {
        "pretrained": False,
        "trainable_backbone_layers": int(saved.get("trainable_backbone_layers", 3)),
        "min_size": int(saved.get("height", 512)),
        "max_size": int(saved.get("width", 1024)),
        "num_classes": int(checkpoint.get("num_classes", saved.get("num_classes", 2))),
    }
