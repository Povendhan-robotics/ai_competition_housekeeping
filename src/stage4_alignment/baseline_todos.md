# Stage 4 (Alignment pass/fail) — Ideas to Improve

Beginner:
- Tune symmetry_threshold in config
- Try different Canny thresholds

Intermediate:
- Crop ROI to bed region before computing symmetry
- Use Hough lines to estimate bed axis and compare pillow axis

Advanced:
- Segment bed + pillow (classical or ML) and compute geometric alignment
- Keypoint approach: detect corners/edges and compute angle/offset
- Train a small model for alignment_pass and compare with geometry baseline
