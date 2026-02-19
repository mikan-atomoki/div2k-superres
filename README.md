# HAT x4 Super-Resolution Training System

DIV2Kデータセットを用いたHAT-L (Hybrid Attention Transformer) x4超解像モデルの学習システム。
basicsr非依存のスタンドアロン実装。8x NVIDIA A6000 (48GB) 向けに最適化。

## セットアップ

```bash
bash setup.sh                   # conda環境作成 + 依存インストール
conda activate hat-sr
bash scripts/download_div2k.sh  # DIV2Kダウンロード (~7GB)
```

## 学習

```bash
# 単一GPUテスト
python train.py --config configs/hat_x4.yaml --devices 1

# 8GPU分散学習
bash scripts/train.sh

# パラメータ上書き
python train.py --config configs/hat_x4.yaml training.lr=1e-4 data.batch_size=4

# チェックポイントから再開
bash scripts/train.sh --resume experiments/hat_l_x4_div2k/checkpoints/last.ckpt

# W&Bロガー使用
bash scripts/train.sh --wandb
```

## 推論

```bash
# 単一画像
python inference.py --checkpoint <ckpt> --input image.png --output results/

# ディレクトリ一括処理
python inference.py --checkpoint <ckpt> --input test_images/ --output results/

# 大画像向けタイル推論 (OOM回避)
python inference.py --checkpoint <ckpt> --input image.png --tile_size 256
```

## 学習設定 (8x A6000)

| 項目 | 値 |
|------|-----|
| 実効バッチサイズ | 256 (8GPU × 8/GPU × accum 4) |
| 精度 | bf16-mixed |
| Optimizer | Adam (lr=2e-4, betas=0.9/0.99) |
| Scheduler | Cosine + 5000step warmup |
| 総ステップ数 | 500,000 |
| 推定学習時間 | 約7-8日 |

## モデル: HAT-L

- **パラメータ数**: ~40.8M
- **構成**: 12 RHAG × (6 HAB + 1 OCAB)
- **Window Size**: 16, Overlap: 24
- **Embed Dim**: 180, Heads: 6

## プロジェクト構成

```
configs/hat_x4.yaml          # ハイパーパラメータ
src/models/hat.py             # HATアーキテクチャ
src/data/div2k.py             # Dataset & DataModule
src/losses/losses.py          # L1 + Perceptual Loss
src/lightning_module.py       # LightningModule
train.py                      # 学習エントリポイント
inference.py                  # 推論スクリプト
scripts/download_div2k.sh     # データダウンロード
scripts/train.sh              # 学習起動
```

## 参考文献

- [Activating More Pixels in Image Super-Resolution Transformer](https://arxiv.org/abs/2309.05239) (CVPR 2023)
- [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT)
