# Loveca MusicCard Model
Loveca MusicCard Model is a deep learning model for detecting and classifying Loveca Music Cards from images.
It supports direct image input and top-K prediction display for fast recognition and digital archiving.

## Supported cards
| Card ID | Card Name |
| :--- | :--- |
| LL-PR-004-PR | 愛♡スクリ～ム！ |
| PL!-bp3-019-L | 僕らのLIVE 君とのLIFE |
| PL!-bp3-020-L | Snow halation |
| PL!-bp3-021-L | 愛してるばんざーい! |
| PL!-bp3-022-L | ユメノトビラ |
| PL!-bp3-023-L | ミはμ'sicのミ |
| PL!-bp3-024-L | 夏色えがおで1,2,Jump! |
| PL!-bp3-025-L | タカラモノズ |
| PL!-bp3-026-L | Oh,Love&Peace! |
| PL!-pb1-028-L | WAO-WAO Powerful day! |
| PL!-pb1-029-L | 知らないLove＊教えてLove |
| PL!-pb1-030-L | Cutie Panther |
| PL!-pb1-032-L | SENTIMENTAL StepS |
| PL!-pb1-033-L | KiRa-KiRa Sensation! |
| PL!-sd1-019-SD | START:DASH!! |
| PL!-sd1-020-SD | きっと青春が聞こえる |
| PL!-sd1-021-SD | これからのSomeday |
| PL!-sd1-022-SD | 僕らは今のなかで |
| PL!HS-bp1-019-L | Dream Believers |
| PL!HS-bp1-020-L | 365 Days |
| PL!HS-bp1-021-L | Holiday∞Holiday |
| PL!HS-bp1-022-L | AWOKE |
| PL!HS-bp1-023-L | ド！ド！ド！ |
| PL!HS-bp2-019-L | Bloom the smile, Bloom the dream! |
| PL!HS-bp2-020-L | Link to the FUTURE |
| PL!HS-bp2-021-L | 眩耀夜行 |
| PL!HS-bp2-022-L | アオクハルカ |
| PL!HS-bp2-023-L | Mirage Voyage |
| PL!HS-bp2-024-L | レディバグ |
| PL!HS-bp2-025-L | ココン東西 |
| PL!HS-bp2-026-L | みらくりえーしょん |
| PL!HS-PR-010-PR | Reflection in the mirror |
| PL!HS-PR-011-PR | Sparkly Spot |
| PL!HS-PR-012-PR | アイデンティティ |
| PL!N-bp1-025-L | 虹色Passions！ |
| PL!N-bp1-026-L | Poppin' Up! |
| PL!N-bp1-027-L | Solitude Rain |
| PL!N-bp1-028-L | Butterfly |
| PL!N-bp1-029-L | Eutopia |
| PL!N-bp3-025-L | Awakening Promise |
| PL!N-bp3-026-L | サイコーハート |
| PL!N-bp3-027-L | La Bella Patria |
| PL!N-bp3-028-L | ツナガルコネクト |
| PL!N-bp3-029-L | 未来ハーモニー |
| PL!N-bp3-030-L | Love U my friends |
| PL!N-bp3-031-L | MONSTER GIRLS |
| PL!N-bp3-032-L | THE SECRET NiGHT |
| PL!N-sd1-025-SD | Colorful Dreams! Colorful Smiles! |
| PL!N-sd1-026-SD | 夢が僕らの太陽さ |
| PL!N-sd1-027-SD | Just Believe!!! |
| PL!N-sd1-028-SD | Dream with You |
| PL!S-bp2-019-L | WATER BLUE NEW WORLD |
| PL!S-bp2-020-L | DREAMY COLOR |
| PL!S-bp2-021-L | 未体験HORIZON |
| PL!S-bp2-022-L | 未熟DREAMER |
| PL!S-bp2-023-L | MY舞☆TONIGHT |
| PL!S-bp2-024-L | 君のこころは輝いてるかい？ |
| PL!S-bp2-025-L | 青空Jumping Heart |
| PL!S-bp2-026-L | ユメ語るよりユメ歌おう |
| PL!S-bp3-019-L | MIRACLE WAVE |
| PL!S-bp3-020-L | ダイスキだったらダイジョウブ！ |
| PL!S-bp3-021-L | 想いよひとつになれ |
| PL!S-bp3-022-L | Fantastic Departure! |
| PL!S-bp3-023-L | KOKORO Magic “A to Z” |
| PL!S-bp3-024-L | Deep Resonance |
| PL!S-bp3-025-L | SUKI for you, DREAM for you! |
| PL!S-pb1-019-L | 元気全開DAY！DAY！DAY！ |
| PL!S-pb1-020-L | トリコリコPLEASE!! |
| PL!S-pb1-021-L | Strawberry Trapper |
| PL!S-pb1-022-L | 逃走迷走メビウスループ |
| PL!S-pb1-023-L | Next SPARKLING!! |
| PL!S-pb1-024-L | 僕らの走ってきた道は・・・ |
| PL!S-PR-022-PR | HAPPY PARTY TRAIN |
| PL!S-PR-023-PR | 恋になりたいAQUARIUM |
| PL!S-PR-024-PR | 勇気はどこに?君の胸に! |
| PL!SP-bp1-023-L | START!! True dreams |
| PL!SP-bp1-024-L | Tiny Stars |
| PL!SP-bp1-025-L | Starlight Prologue |
| PL!SP-bp1-026-L | 未来予報ハレルヤ！ |
| PL!SP-bp1-027-L | Sing！Shine！Smile！ |
| PL!SP-bp2-023-L | Go!! リスタート |
| PL!SP-bp2-024-L | ビタミンSUMMER! |
| PL!SP-bp2-025-L | Bubble Rise |
| PL!SP-bp2-026-L | 笑顔のPromise |
| PL!SP-bp2-027-L | UNIVERSE!! |
| PL!SP-pb1-023-L | ディストーション |
| PL!SP-pb1-024-L | ニュートラル |
| PL!SP-pb1-025-L | Jellyfish |
| PL!SP-pb1-026-L | Jump Into the New World |
| PL!SP-sd1-023-SD | WE WILL!! |
| PL!SP-sd1-024-SD | シェキラ☆☆☆ |
| PL!SP-sd1-025-SD | 未来は風のように |
| PL!SP-sd1-026-SD | 私のSymphony 〜澁谷かのんVer.〜 |

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
This project is released under the Apache-2.0 License.
