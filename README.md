# Loveca MusicCard Model
Loveca MusicCard Model is a deep learning model for detecting and classifying Loveca Music Cards from images.
It supports direct image input and top-K prediction display for fast recognition and digital archiving.

## Supported cards
| Card ID | Card Name | Series Name |
|--------|-----------|-------------|
| LL-PR-004-PR | 愛♡スクリ～ム！ | AiScReam |
| PL!-bp3-019-L | 僕らのLIVE 君とのLIFE | ラブライブ！ |
| PL!-bp3-020-L | Snow halation | ラブライブ！ |
| PL!-bp3-020-L＋ | Snow halation | ラブライブ！ |
| PL!-bp3-021-L | 愛してるばんざーい! | ラブライブ！ |
| PL!-bp3-022-L | ユメノトビラ | ラブライブ！ |
| PL!-bp3-023-L | ミはμ'sicのミ | ラブライブ！ |
| PL!-bp3-024-L | 夏色えがおで1,2,Jump! | ラブライブ！ |
| PL!-bp3-025-L | タカラモノズ | ラブライブ！ |
| PL!-bp3-026-L | Oh,Love&Peace! | ラブライブ！ |
| PL!-bp4-019-L | Angelic Angel | ラブライブ！ |
| PL!-bp4-020-L | Love wing bell | ラブライブ！ |
| PL!-bp4-021-L | ?←HEARTBEAT | ラブライブ！ |
| PL!-bp4-022-L | No brand girls | ラブライブ！ |
| PL!-bp4-023-L | もぎゅっと"love"で接近中！ | ラブライブ！ |
| PL!-pb1-031-L | 輝夜の城で踊りたい | ラブライブ！ |
| PL!-pb1-032-L | SENTIMENTAL StepS | ラブライブ！ |
| PL!-pb1-033-L | KiRa-KiRa Sensation! | ラブライブ！ |
| PL!-sd1-019-SD | START:DASH!! | ラブライブ！ |
| PL!-sd1-020-SD | きっと青春が聞こえる | ラブライブ！ |
| PL!-sd1-021-SD | これからのSomeday | ラブライブ！ |
| PL!-sd1-022-SD | 僕らは今のなかで | ラブライブ！ |
| PL!-bp4-024-L | 小夜啼鳥恋詩 | ラブライブ！ |
| PL!-pb1-028-L | WAO-WAO Powerful day! | ラブライブ！ |
| PL!-bp4-026-L | ダイヤモンドプリンセスの憂鬱 | ラブライブ！ |
| PL!-pb1-030-L | Cutie Panther | ラブライブ！ |
| PL!-bp4-025-L | 微熱からMystery | ラブライブ！ |
| PL!-pb1-029-L | 知らないLove＊教えてLove | ラブライブ！ |
| PL!S-bp2-019-L | WATER BLUE NEW WORLD | ラブライブ！サンシャイン!! |
| PL!S-bp2-020-L | DREAMY COLOR | ラブライブ！サンシャイン!! |
| PL!S-bp2-021-L | 未体験HORIZON | ラブライブ！サンシャイン!! |
| PL!S-bp2-022-L | 未熟DREAMER | ラブライブ！サンシャイン!! |
| PL!S-bp2-023-L | MY舞☆TONIGHT | ラブライブ！サンシャイン!! |
| PL!S-bp2-024-L | 君のこころは輝いてるかい？ | ラブライブ！サンシャイン!! |
| PL!S-bp2-025-L | 青空Jumping Heart | ラブライブ！サンシャイン!! |
| PL!S-bp2-026-L | ユメ語るよりユメ歌おう | ラブライブ！サンシャイン!! |
| PL!S-bp3-019-L | MIRACLE WAVE | ラブライブ！サンシャイン!! |
| PL!S-bp3-020-L | ダイスキだったらダイジョウブ！ | ラブライブ！サンシャイン!! |
| PL!S-bp3-021-L | 想いよひとつになれ | ラブライブ！サンシャイン!! |
| PL!S-bp3-022-L | Fantastic Departure! | ラブライブ！サンシャイン!! |
| PL!S-bp3-023-L | KOKORO Magic “A to Z” | ラブライブ！サンシャイン!! |
| PL!S-bp3-024-L | Deep Resonance | ラブライブ！サンシャイン!! |
| PL!S-bp3-025-L | SUKI for you, DREAM for you! | ラブライブ！サンシャイン!! |
| PL!S-pb1-023-L | Next SPARKLING!! | ラブライブ！サンシャイン!! |
| PL!S-pb1-023-L＋ | Next SPARKLING!! | ラブライブ！サンシャイン!! |
| PL!S-pb1-024-L | 僕らの走ってきた道は・・・ | ラブライブ！サンシャイン!! |
| PL!S-PR-022-PR | HAPPY PARTY TRAIN | ラブライブ！サンシャイン!! |
| PL!S-PR-023-PR | 恋になりたいAQUARIUM | ラブライブ！サンシャイン!! |
| PL!S-PR-024-PR | 勇気はどこに?君の胸に! | ラブライブ！サンシャイン!! |
| PL!S-pb1-021-L | Strawberry Trapper | ラブライブ！サンシャイン!! |
| PL!S-pb1-020-L | トリコリコPLEASE!! | ラブライブ！サンシャイン!! |
| PL!S-pb1-022-L | 逃走迷走メビウスループ | ラブライブ！サンシャイン!! |
| PL!S-pb1-022-L＋ | 逃走迷走メビウスループ | ラブライブ！サンシャイン!! |
| PL!S-pb1-019-L | 元気全開DAY！DAY！DAY！ | ラブライブ！サンシャイン!! |
| PL!SP-bp1-023-L | START!! True dreams | ラブライブ！スーパースター!! |
| PL!SP-bp1-024-L | Tiny Stars | ラブライブ！スーパースター!! |
| PL!SP-bp1-025-L | Starlight Prologue | ラブライブ！スーパースター!! |
| PL!SP-bp1-025-L＋ | Starlight Prologue | ラブライブ！スーパースター!! |
| PL!SP-bp1-026-L | 未来予報ハレルヤ！ | ラブライブ！スーパースター!! |
| PL!SP-bp1-027-L | Sing！Shine！Smile！ | ラブライブ！スーパースター!! |
| PL!SP-bp2-023-L | Go!! リスタート | ラブライブ！スーパースター!! |
| PL!SP-bp2-024-L | ビタミンSUMMER! | ラブライブ！スーパースター!! |
| PL!SP-bp2-025-L | Bubble Rise | ラブライブ！スーパースター!! |
| PL!SP-bp2-026-L | 笑顔のPromise | ラブライブ！スーパースター!! |
| PL!SP-bp2-027-L | UNIVERSE!! | ラブライブ！スーパースター!! |
| PL!SP-bp4-023-L | Dazzling Game | ラブライブ！スーパースター!! |
| PL!SP-bp4-024-L | ノンフィクション!! | ラブライブ！スーパースター!! |
| PL!SP-bp4-025-L | Special Color | ラブライブ！スーパースター!! |
| PL!SP-bp4-026-L | Wish Song | ラブライブ！スーパースター!! |
| PL!SP-bp4-027-L | Chance Day, Chance Way! | ラブライブ！スーパースター!! |
| PL!SP-bp4-028-L | DAISUKI FULL POWER | ラブライブ！スーパースター!! |
| PL!SP-bp4-029-L | 追いかける夢の先で | ラブライブ！スーパースター!! |
| PL!SP-bp4-030-L | Second Sparkle | ラブライブ！スーパースター!! |
| PL!SP-pb1-026-L | Jump Into the New World | ラブライブ！スーパースター!! |
| PL!SP-pb1-026-L＋ | Jump Into the New World | ラブライブ！スーパースター!! |
| PL!SP-sd1-023-SD | WE WILL!! | ラブライブ！スーパースター!! |
| PL!SP-sd1-024-SD | シェキラ☆☆☆ | ラブライブ！スーパースター!! |
| PL!SP-sd1-025-SD | 未来は風のように | ラブライブ！スーパースター!! |
| PL!SP-sd1-026-SD | 私のSymphony 〜澁谷かのんVer.〜 | ラブライブ！スーパースター!! |
| PL!SP-pb1-023-L | ディストーション | ラブライブ！スーパースター!! |
| PL!SP-pb1-024-L | ニュートラル | ラブライブ！スーパースター!! |
| PL!SP-pb1-025-L | Jellyfish | ラブライブ！スーパースター!! |
| PL!N-bp1-025-L | 虹色Passions！ | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp1-026-L | Poppin' Up! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp1-027-L | Solitude Rain | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp1-028-L | Butterfly | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp1-029-L | Eutopia | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-025-L | Awakening Promise | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-026-L | サイコーハート | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-027-L | La Bella Patria | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-028-L | ツナガルコネクト | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-029-L | 未来ハーモニー | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-030-L | Love U my friends | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-025-L | VIVID WORLD | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-026-L | DIVE! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-027-L | EMOTION | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-028-L | stars we chase | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-029-L | Rise Up High! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-030-L | Daydream Mermaid | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-031-L | NEO SKY, NEO MAP! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-pb1-037-L | Cara Tesoro | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-pb1-038-L | PHOENIX | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-pb1-039-L | Stellar Stream | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-pb1-040-L | どこにいても君は君 | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-pb1-042-L | Eternalize Love!! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-sd1-025-SD | Colorful Dreams! Colorful Smiles! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-sd1-026-SD | 夢が僕らの太陽さ | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-sd1-027-SD | Just Believe!!! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-sd1-028-SD | Dream with You | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-032-L | THE SECRET NiGHT | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-032-L | Blue! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp4-032-L＋ | Blue! | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-pb1-041-L | PASTEL | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-pb1-041-L＋ | PASTEL | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!N-bp3-031-L | MONSTER GIRLS | ラブライブ！虹ヶ咲学園スクールアイドル同好会 |
| PL!HS-bp1-019-L | Dream Believers | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp1-020-L | 365 Days | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-019-L | Bloom the smile, Bloom the dream! | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-020-L | Link to the FUTURE | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-PR-010-PR | Reflection in the mirror | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-PR-011-PR | Sparkly Spot | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-PR-012-PR | アイデンティティ | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp1-021-L | Holiday∞Holiday | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-021-L | 眩耀夜行 | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-022-L | アオクハルカ | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp1-022-L | AWOKE | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-023-L | Mirage Voyage | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-024-L | レディバグ | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp1-023-L | ド！ド！ド！ | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-025-L | ココン東西 | 蓮ノ空女学院スクールアイドルクラブ |
| PL!HS-bp2-026-L | みらくりえーしょん | 蓮ノ空女学院スクールアイドルクラブ |

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
