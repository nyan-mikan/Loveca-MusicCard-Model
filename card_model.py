import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    return tensor


class ImageTransform:
    """Callable transform that handles training or eval preprocessing."""

    def __init__(self, image_size: int = 224, training: bool = False):
        self.image_size = image_size
        self.training = training

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        if self.training:
            img = self._augment(img)
            img = _random_scale_and_pad(img, self.image_size)
        else:
            img = ImageOps.fit(
                img,
                (self.image_size, self.image_size),
                method=Image.BICUBIC,
                bleed=0.0,
                centering=(0.5, 0.5),
            )
        return _to_tensor(img)

    def _augment(self, img: Image.Image) -> Image.Image:
        img = img.copy()
        # Random rotation within +/- 20 degrees
        if random.random() < 0.9:
            angle = random.uniform(-20.0, 20.0)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

        # Random perspective-like shear via affine transform
        if random.random() < 0.5:
            w, h = img.size
            xshift = random.uniform(-0.1, 0.1) * w
            yshift = random.uniform(-0.1, 0.1) * h
            data = (
                1,
                xshift / h,
                0,
                yshift / w,
                1,
                0,
            )
            img = img.transform(img.size, Image.AFFINE, data, resample=Image.BICUBIC)

        # Random horizontal flip
        if random.random() < 0.5:
            img = ImageOps.mirror(img)

        # Random color jitter
        if random.random() < 0.8:
            brightness = ImageEnhance.Brightness(img)
            img = brightness.enhance(random.uniform(0.7, 1.4))

        if random.random() < 0.8:
            contrast = ImageEnhance.Contrast(img)
            img = contrast.enhance(random.uniform(0.7, 1.4))

        if random.random() < 0.8:
            color = ImageEnhance.Color(img)
            img = color.enhance(random.uniform(0.7, 1.4))

        if random.random() < 0.6:
            img = _apply_specular_glare(img)

        if random.random() < 0.5:
            img = _mask_text_band(img)

        # Random crop keeping most of the card
        w, h = img.size
        min_scale = 0.7
        scale = random.uniform(min_scale, 1.0)
        crop_w = max(int(w * scale), 1)
        crop_h = max(int(h * scale), 1)
        if crop_w < w or crop_h < h:
            left = random.randint(0, w - crop_w)
            top = random.randint(0, h - crop_h)
            img = img.crop((left, top, left + crop_w, top + crop_h))

        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        img = _random_erasing(img)

        return img


class CardDataset(Dataset):
    """Dataset that repeats card images to enable augmentation-heavy training."""

    def __init__(
        self,
        cards_dirs: Sequence[Path] | Path,
        repeats: int = 32,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        label_map: Optional[Dict[str, str]] = None,
        dir_weights: Optional[Sequence[int]] = None,
    ):
        directories = _normalize_directories(cards_dirs)
        weights = _normalize_dir_weights(dir_weights, len(directories))

        self.label_map = {str(key): str(value) for key, value in (label_map or {}).items()}
        self.transform = transform or ImageTransform(training=False)
        self.repeats = max(1, int(repeats))

        self.samples: List[Tuple[Path, str]] = []
        self.source_summary: List[Dict[str, object]] = []

        for dir_path, weight in zip(directories, weights):
            if not dir_path.exists():
                raise FileNotFoundError(f"Cards directory not found: {dir_path}")

            image_paths = sorted(
                [
                    path
                    for path in dir_path.iterdir()
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
            )

            if not image_paths:
                continue

            for path in image_paths:
                label = self._resolve_label(path)
                for _ in range(weight):
                    self.samples.append((path, label))

            self.source_summary.append(
                {
                    "directory": str(dir_path),
                    "images": len(image_paths),
                    "weight": weight,
                }
            )

        if not self.samples:
            raise ValueError(
                f"No images found across directories: {', '.join(str(d) for d in directories)}"
            )

        self.label_names = sorted({label for _, label in self.samples})
        self.label_to_index = {label: idx for idx, label in enumerate(self.label_names)}

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        base_idx = idx % len(self.samples)
        path, label_name = self.samples[base_idx]
        with Image.open(path) as img:
            tensor = self.transform(img)
        label_index = self.label_to_index[label_name]
        return tensor, label_index

    def _resolve_label(self, path: Path) -> str:
        if self.label_map:
            if path.name in self.label_map:
                return self.label_map[path.name]
            if path.stem in self.label_map:
                return self.label_map[path.stem]
        label = path.stem
        if "_deg" in label:
            label = label.split("_deg")[0]
        if self.label_map and label in self.label_map:
            return self.label_map[label]
        return label


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CardNet(nn.Module):
    """Compact CNN tailored for card recognition on CPU."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x


@dataclass
class ModelArtifacts:
    model_path: Path
    label_index_path: Path

    def save_labels(self, labels: Sequence[str]) -> None:
        payload = {"labels": list(labels)}
        with self.label_index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_labels(self) -> List[str]:
        with self.label_index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "labels" not in payload:
            raise KeyError(f"labels key missing in {self.label_index_path}")
        return list(payload["labels"])


def create_artifacts(output_dir: Path) -> ModelArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "card_classifier.pt"
    label_index_path = output_dir / "label_index.json"
    return ModelArtifacts(model_path=model_path, label_index_path=label_index_path)


def load_model(
    model_path: Path,
    num_classes: int,
    map_location: Optional[torch.device] = None,
) -> CardNet:
    model = CardNet(num_classes=num_classes)
    state = torch.load(model_path, map_location=map_location or "cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _apply_specular_glare(img: Image.Image) -> Image.Image:
    """Overlay synthetic glare to improve robustness to reflective highlights."""
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return img

    h, w, _ = arr.shape
    glare_layers = random.randint(1, 2)
    for _ in range(glare_layers):
        center_x = random.uniform(0.1, 0.9) * w
        center_y = random.uniform(0.1, 0.9) * h
        radius_x = random.uniform(0.2, 0.6) * w
        radius_y = random.uniform(0.2, 0.6) * h
        strength = random.uniform(0.3, 0.9) * 255.0

        y_coords, x_coords = np.ogrid[:h, :w]
        norm = ((x_coords - center_x) / (radius_x + 1e-6)) ** 2 + (
            (y_coords - center_y) / (radius_y + 1e-6)
        ) ** 2
        mask = np.exp(-norm * random.uniform(2.0, 5.0))
        mask = mask[..., np.newaxis]

        tint = np.array(
            [
                random.uniform(0.9, 1.0),
                random.uniform(0.9, 1.0),
                random.uniform(0.9, 1.0),
            ],
            dtype=np.float32,
        )
        arr += mask * strength * tint

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _random_scale_and_pad(img: Image.Image, target_size: int) -> Image.Image:
    """Randomly scale card and place on a padded canvas to simulate zoomed-out shots."""
    scale = random.uniform(0.55, 1.0)
    max_side = int(target_size * scale)
    max_side = max(1, max_side)
    img_resized = ImageOps.contain(img, (max_side, max_side), method=Image.BICUBIC)

    background_color = tuple(int(x) for x in np.random.randint(0, 60, size=3))
    canvas = Image.new("RGB", (target_size, target_size), background_color)
    max_x = target_size - img_resized.width
    max_y = target_size - img_resized.height
    offset_x = random.randint(0, max(0, max_x))
    offset_y = random.randint(0, max(0, max_y))
    canvas.paste(img_resized, (offset_x, offset_y))
    return canvas


def _random_erasing(img: Image.Image, max_patches: int = 3) -> Image.Image:
    """Randomly occlude parts of the card so the model relies less on specific text."""
    arr = np.asarray(img).astype(np.float32)
    h, w, _ = arr.shape
    num_patches = random.randint(1, max_patches)
    for _ in range(num_patches):
        if random.random() > 0.6:
            continue
        erase_scale = random.uniform(0.02, 0.15)
        erase_w = int(w * erase_scale)
        erase_h = int(h * random.uniform(0.05, 0.25))
        if erase_w < 1 or erase_h < 1:
            continue
        x = random.randint(0, max(0, w - erase_w))
        y = random.randint(0, max(0, h - erase_h))
        fill_color = np.random.uniform(0, 255, size=(3,)).astype(np.float32)
        arr[y : y + erase_h, x : x + erase_w, :] = fill_color
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _mask_text_band(img: Image.Image) -> Image.Image:
    """Mask a horizontal band to reduce over-reliance on localized text."""
    arr = np.asarray(img).astype(np.float32)
    h, w, _ = arr.shape
    band_height = int(h * random.uniform(0.12, 0.25))
    top = int(h * random.uniform(0.55, 0.85))
    top = min(max(top, 0), h - band_height)
    color = np.random.uniform(0, 255, size=(3,)).astype(np.float32)
    alpha = random.uniform(0.4, 0.8)
    arr[top : top + band_height, :, :] = (
        alpha * color + (1 - alpha) * arr[top : top + band_height, :, :]
    )
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _normalize_directories(cards_dirs: Sequence[Path] | Path) -> List[Path]:
    if isinstance(cards_dirs, (str, Path)):
        directories = [Path(cards_dirs)]
    else:
        directories = [Path(directory) for directory in cards_dirs]
    if not directories:
        raise ValueError("At least one cards directory must be provided.")
    return directories


def _normalize_dir_weights(dir_weights: Optional[Sequence[int]], count: int) -> List[int]:
    if dir_weights is None:
        return [1] * count
    if len(dir_weights) != count:
        raise ValueError(
            f"dir_weights length ({len(dir_weights)}) must match cards directories ({count})."
        )
    weights: List[int] = []
    for weight in dir_weights:
        weight_int = int(weight)
        if weight_int < 1:
            raise ValueError("Directory weights must be >= 1.")
        weights.append(weight_int)
    return weights
