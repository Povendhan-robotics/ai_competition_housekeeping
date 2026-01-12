# Stage 1 (Binary QC) — Things to Try

Beginner:
- Change backbone: resnet18, efficientnet_b0, mobilenetv3
- Tune lr: 1e-3, 3e-4, 1e-4
- Increase epochs to 15–20 if dataset is large enough

Intermediate:
- Add class weights for imbalance
- Try label smoothing
- Try different augmentation intensity

Advanced:
- Confidence calibration (temperature scaling)
- Ensembling
- Robustness against lighting + blur
