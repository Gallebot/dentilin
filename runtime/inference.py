from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from .model import CLASS_TO_FDI, build_instance_model, checkpoint_model_args as instance_checkpoint_model_args, postprocess_fdi_prediction

FDI_QUADRANT_COLORS = {
    1: (244, 67, 54),
    2: (33, 150, 243),
    3: (76, 175, 80),
    4: (255, 152, 0),
}


FDI_TOOTH_TYPES = {
    1: "Incisivo central",
    2: "Incisivo lateral",
    3: "Canino",
    4: "Primer premolar",
    5: "Segundo premolar",
    6: "Primer molar",
    7: "Segundo molar",
    8: "Tercer molar",
}


FDI_QUADRANT_NAMES = {
    1: "superior derecho",
    2: "superior izquierdo",
    3: "inferior izquierdo",
    4: "inferior derecho",
}


def fdi_to_spanish_name(fdi: int) -> str:
    quadrant, position = divmod(int(fdi), 10)
    if quadrant not in FDI_QUADRANT_NAMES or position not in FDI_TOOTH_TYPES:
        return "Pieza FDI desconocida"
    return f"{FDI_TOOTH_TYPES[position]} {FDI_QUADRANT_NAMES[quadrant]}"


def preprocess_image(
    source: Image.Image,
    model_size: tuple[int, int],
) -> tuple[torch.Tensor, Image.Image, dict[str, float | int]]:
    original = ImageOps.exif_transpose(source).convert("RGB")
    gray = original.convert("L")
    original_width, original_height = gray.size
    model_width, model_height = model_size
    scale = min(model_width / original_width, model_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = gray.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    offset_x = (model_width - resized_width) // 2
    offset_y = (model_height - resized_height) // 2
    canvas = Image.new("L", model_size, 0)
    canvas.paste(resized, (offset_x, offset_y))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    geometry: dict[str, float | int] = {
        "original_width": original_width,
        "original_height": original_height,
        "model_width": model_width,
        "model_height": model_height,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }
    return tensor, original, geometry


def load_fdi_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[int, int]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved = checkpoint.get("args", {})
    model_size = (int(saved.get("width", 1024)), int(saved.get("height", 512)))
    model = build_instance_model(**instance_checkpoint_model_args(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_size


def _mask_to_original(
    mask: torch.Tensor,
    geometry: dict[str, float | int],
) -> np.ndarray:
    offset_x = int(geometry["offset_x"])
    offset_y = int(geometry["offset_y"])
    resized_width = int(geometry["resized_width"])
    resized_height = int(geometry["resized_height"])
    original_width = int(geometry["original_width"])
    original_height = int(geometry["original_height"])
    cropped = mask[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ].numpy()
    mask_image = Image.fromarray(cropped.astype(np.uint8) * 255, mode="L")
    restored = mask_image.resize(
        (original_width, original_height),
        Image.Resampling.NEAREST,
    )
    return np.asarray(restored, dtype=np.uint8) > 0


@torch.inference_mode()
def run_fdi_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    geometry: dict[str, float | int],
    device: torch.device,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    raw_prediction = model([image_tensor.to(device)])[0]
    prediction = postprocess_fdi_prediction(
        raw_prediction,
        score_threshold=score_threshold,
        nms_threshold=None,
        unique_fdi=True,
    )
    scale = float(geometry["scale"])
    offset_x = int(geometry["offset_x"])
    offset_y = int(geometry["offset_y"])
    original_width = int(geometry["original_width"])
    original_height = int(geometry["original_height"])
    teeth: list[dict[str, Any]] = []
    for mask, box_tensor, label_tensor, score_tensor in zip(
        prediction["masks"],
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"],
    ):
        class_id = int(label_tensor) - 1
        x1 = max(0.0, min(original_width, (float(box_tensor[0]) - offset_x) / scale))
        y1 = max(0.0, min(original_height, (float(box_tensor[1]) - offset_y) / scale))
        x2 = max(0.0, min(original_width, (float(box_tensor[2]) - offset_x) / scale))
        y2 = max(0.0, min(original_height, (float(box_tensor[3]) - offset_y) / scale))
        if x2 <= x1 or y2 <= y1:
            continue
        fdi = CLASS_TO_FDI[int(label_tensor)]
        teeth.append(
            {
                "class_id": class_id,
                "fdi": fdi,
                "score": float(score_tensor),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "mask": _mask_to_original(mask, geometry),
            }
        )
    return sorted(teeth, key=lambda row: int(row["fdi"]))


def draw_teeth_layer(
    image: Image.Image,
    teeth: list[dict[str, Any]],
    opacity: float = 0.38,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for tooth in teeth:
        mask = np.asarray(tooth["mask"], dtype=bool)
        color = np.asarray(
            FDI_QUADRANT_COLORS[int(tooth["fdi"]) // 10],
            dtype=np.float32,
        )
        base[mask] = base[mask] * (1.0 - opacity) + color * opacity
    result = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    line_width = max(2, round(max(result.size) / 700))
    for tooth in teeth:
        fdi = int(tooth["fdi"])
        color = FDI_QUADRANT_COLORS[fdi // 10]
        box = tuple(float(tooth[key]) for key in ("x1", "y1", "x2", "y2"))
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{fdi} · {float(tooth['score']):.0%}"
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = max(0.0, box[0])
        y = max(0.0, box[1] - text_height - 6)
        draw.rectangle((x, y, x + text_width + 6, y + text_height + 6), fill="black")
        draw.text(
            (x + 3, y + 3),
            label,
            fill=color,
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )
    return result


def highlight_teeth(
    image: Image.Image,
    teeth: list[dict[str, Any]],
    selected_fdi: set[int],
) -> Image.Image:
    if not selected_fdi:
        return image.copy()
    result = image.copy().convert("RGB")
    array = np.asarray(result, dtype=np.uint8).copy()
    image_scale = max(result.size) / 1200
    filter_size = max(5, round(7 * image_scale))
    if filter_size % 2 == 0:
        filter_size += 1
    selected = [tooth for tooth in teeth if int(tooth["fdi"]) in selected_fdi]
    for tooth in selected:
        mask = np.asarray(tooth["mask"], dtype=bool)
        expanded = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(
                ImageFilter.MaxFilter(filter_size)
            ),
            dtype=np.uint8,
        ) > 0
        boundary = expanded & ~mask
        array[mask] = (
            array[mask].astype(np.float32) * 0.55
            + np.asarray((255, 235, 59), dtype=np.float32) * 0.45
        ).astype(np.uint8)
        array[boundary] = np.asarray((255, 235, 59), dtype=np.uint8)
    result = Image.fromarray(array)
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    line_width = max(4, round(max(result.size) / 400))
    for tooth in selected:
        box = tuple(float(tooth[key]) for key in ("x1", "y1", "x2", "y2"))
        draw.rectangle(box, outline=(255, 235, 59), width=line_width)
        label = f"FDI {int(tooth['fdi'])} seleccionado"
        text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        x = max(0.0, box[0])
        y = max(0.0, box[1] - text_height - 8)
        draw.rectangle(
            (x, y, x + text_width + 8, y + text_height + 8),
            fill=(0, 0, 0),
        )
        draw.text(
            (x + 4, y + 4),
            label,
            fill=(255, 235, 59),
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    return result
