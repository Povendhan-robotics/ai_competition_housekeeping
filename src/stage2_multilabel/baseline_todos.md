# Stage 2 (Multi-label defects) — Things to Try

Beginner:
- Just run the baseline end-to-end.
- Try different backbones: resnet18, efficientnet_b0, mobilenetv3
- Try lr: 1e-3, 3e-4, 1e-4

Intermediate:
- Tune per-label thresholds on val (see eval.py --tune_thresholds)
- Add pos_weight for imbalance in configs/stage2_multilabel.yaml
- Reduce augmentation intensity if defects are subtle

Advanced:
- Multi-task learning with Stage 1 (shared backbone, two heads)
- Calibration of probabilities
- Robustness to lighting/blur (helpful in real QC)
