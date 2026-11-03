# Task 4 Report - Better Leaf Classifier

## 1. Comparison Table (This Year Dataset + Last Year Dataset)

Our group validation set has only 28 images, so 1 image is $1/28 \approx 3.6\%$ accuracy. Because of this noise, every variant was run with 3 seeds (0, 1, 2).
We measured accuracy on three validation sets:
1. **Group Val (28 imgs):** our group validation photos from this year.
2. **Camera-like Val (28 imgs):** our validation photos with random crop, rotation 15° and color jitter.
3. **Last Year Val (120 imgs):** historical validation set from previous years (other students, different phones).

| Variant | Train Data | Group Val (this year) | Camera-like Val | Last Year Val (120 imgs) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 1. Baseline `LeafNet` (15 ep, no BN, no aug) | 101 (group) | 54.8% ± 1.7% | 59.5% ± 3.4% | 36.7% ± 2.0% | Basic model from 3b, overfits to our group |
| 2. + BatchNorm (`LeafNetBN`, 15 ep) | 101 (group) | 58.3% ± 8.4% | 50.0% ± 2.9% | 45.0% ± 5.1% | Helps on last year data, but big variance |
| 3. + Geometric aug (crop, flips, rot) | 101 (group) | 54.8% ± 8.9% | 47.6% ± 8.9% | 39.2% ± 6.0% | 15 epochs is too short for heavy aug |
| 4. + ColorJitter (brightness, contrast, sat) | 101 (group) | 59.5% ± 1.7% | 60.7% ± 2.9% | 45.8% ± 5.3% | Better stability with different lighting |
| 5. + GAP, Dropout, AdamW, CosineLR, Best Ep | 101 (group) | 64.3% ± 3.2% | 65.5% ± 4.0% | 45.0% ± 5.0% | Less parameters (61k vs 208k), less overfit |
| 6. Pretrained `mobilenet_v3_small` + aug + best ep | 101 (group) | 67.9% ± 5.0% | 64.3% ± 7.6% | 74.2% ± 2.5% | Good on other phones, but big file (6.3 MB) |
| 7. `LeafNetGAP` + full recipe + **last year train data** | **801 (both)** | **72.6% ± 1.7%** | **71.4% ± 2.9%** | **84.2% ± 1.8%** | **Best result on both datasets, small (240 KB)** |

## 2. What helped and what is larger than noise

- **Adding last year training data (101 -> 801 images):** This brought the biggest jump. When training from scratch only on our 101 group images, accuracy on last year photos was stuck around 36-45% because network only knew our phones and backgrounds. With combined 801 training photos, accuracy jumped to **72.6%** on our group val (+18% above baseline, ~5 images) and **84.2%** on last year val (+47% above baseline), with very small noise (±1.8%).
- **Global Average Pooling (GAP) + Dropout (`LeafNetGAP`):** In original `LeafNet`, flattening $64 \times 6 \times 6$ into linear layer had 147k weights. Replacing it with `AdaptiveAvgPool2d((1, 1))` and Dropout reduced parameters from 208k to 61k and improved our group validation by ~5-9% (> 2 images).
- **ColorJitter:** Phone camera changes white balance and exposure all the time. Adding `ColorJitter` improved camera-like accuracy by ~10% and last year val by ~6% compared to only geometric agumentation.
- **Saving best epoch + CosineAnnealingLR + AdamW:** In Task 2 and 3 we took weights from last epoch (epoch 15), where model was already overfitted. Keeping `best_state` with cosine scheduler gave consistent improvement and reduced variance between seeds.

## 3. What did NOT help

- **Only geometric augmentation in 15 epochs (Variant 3):** Without more epochs and learning rate decay, adding random crops and flips on 101 images made variance huge (±8.9%) and accuracy dropped on both datasets because model did not converge in 15 epochs.
- **Fine-tuning whole MobileNet without regularization (Task 2):** In Task 2, MobileNet had 78.6% in epoch 1, but then overfitted to 57.1% in epoch 15 because learning rate $10^{-3}$ destroyed pretrained features on small dataset.

## 4. Which model I put on the phone

I deployed **Variant 7 (`LeafNetGAP` trained from scratch on combined dataset of 801 images from this year + last year)** into the web app.
It has highest accuracy on both datasets (72.6% on our group and 84.2% on last year photos) and the exported `leaf_model.onnx` is only **240 KB**.
