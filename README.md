# Loveca MusicCard Model
Loveca MusicCard Model is a deep learning model for detecting and classifying Loveca Music Cards from images.
It supports direct image input and top-K prediction display for fast recognition and digital archiving.

## Project Structure
```
Loveca-MusicCard-Scanner/
├── card_model.py              # Model and transform definitions
├── infer.py                   # Inference entry point (this script)
├── artifacts/
│   ├── loveca.pt              # Pretrained model weights (default)
│   └── label_index.json       # Label index file (default)
└── requirements.txt
```

## Requirements
Install dependencies before running inference:
`pip install -r requirements.txt`

## Usage
Run inference on a single image:

`python infer.py --image path/to/your_image.jpg`

### Optional arguments:
| Argument      | Default                      | Description                                                   |
|----------------|------------------------------|---------------------------------------------------------------|
| `--image`      | *(required)*                 | Path to the input image file                                  |
| `--weights`    | `./artifacts/loveca.pt`      | Path to the trained model weights                             |
| `--labels`     | `./artifacts/label_index.json` | Path to the label file                                       |
| `--top-k`      | `3`                          | Number of top predictions to show                             |
| `--image-size` | `224`                        | Image size expected by the model                              |
| `--device`     | `auto`                       | Device to run inference on (`cpu`, `cuda`, `mps`, or `auto`)  |

## Default Artifacts
- Model weights: `./artifacts/loveca.pt`
- Label index: `./artifacts/label_index.json`

## License
This project is released under the MIT License.
