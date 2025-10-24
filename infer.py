import argparse
from pathlib import Path
import json
from typing import List, Dict, Union

import torch
from PIL import Image

from card_model import ImageTransform, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on a Loveca Music Card image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the input image to classify."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained model weights (e.g., .pth/.pt)."
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to labels file (.txt with one label per line, or .json list/dict)."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size expected by the model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run inference on."
    )
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_labels_any(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    if path.suffix.lower() == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        labels = [ln.strip() for ln in lines if ln.strip()]
        if not labels:
            raise ValueError(f"Empty labels file: {path}")
        return labels

    if path.suffix.lower() == ".json":
        data: Union[List[str], Dict[Union[str, int], str]] = json.loads(
            path.read_text(encoding="utf-8")
        )
        if isinstance(data, list):
            if not data:
                raise ValueError(f"Empty label list in JSON: {path}")
            return [str(x) for x in data]
        if isinstance(data, dict):
            try:
                items = sorted(((int(k), v) for k, v in data.items()), key=lambda x: x[0])
            except Exception as e:
                raise ValueError(
                    f"JSON labels dict must have integer-like keys. Error: {e}"
                )
            labels = [str(v) for _, v in items]
            if not labels:
                raise ValueError(f"Empty label dict in JSON: {path}")
            return labels
        raise ValueError(f"Unsupported JSON structure in labels file: {path}")

    raise ValueError(f"Unsupported labels file extension: {path.suffix}")


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    labels = load_labels_any(args.labels)

    model = load_model(args.weights, num_classes=len(labels))

    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    transform = ImageTransform(image_size=args.image_size, training=False)

    with Image.open(args.image) as img:
        tensor = transform(img)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    top_k = min(max(1, args.top_k), len(labels))
    probs, indices = torch.topk(probabilities, k=top_k, dim=1)

    print("Top predictions:")
    for rank in range(top_k):
        label_idx = indices[0, rank].item()
        label_name = labels[label_idx]
        probability = probs[0, rank].item()
        print(f"{rank + 1}. {label_name} ({probability:.2%})")

    best_label = labels[indices[0, 0].item()]
    print(f"\nPredicted card: {best_label}")


if __name__ == "__main__":
    main()
