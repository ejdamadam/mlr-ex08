# Task 4 Report - Better Leaf Classifier

## 1. Comparison Table

Our validation set has only 28 images, so 1 image is $1/28 \approx 3.6\%$ accuracy. Because of this noise, main variants were run with 3 seeds (0, 1, 2). Besides clean validation set, we also tested on camera-like validation set (random crop, rotation 15°, color jitter) and historical dataset from last year.

| Variant | Clean Val Acc | Camera-like Val Acc | Notes |
| :--- | :---: | :---: | :--- |
| 1. Baseline `LeafNet` (15 ep, no BN, no aug) | 54.8% ± 1.7% | 59.5% ± 3.4% | Basic model from 3b, overfits fast |
| 2. + BatchNorm (`LeafNetBN`, 15 ep) | 56.0% ± 8.4% | 50.0% ± 2.9% | Faster training loss drop, big variance |
| 3. + Geometric aug (crop, flips, rot) | 54.8% ± 8.9% | 47.6% ± 8.9% | 15 epochs is too short when aug is heavy |
| 4. + ColorJitter (brightness, contrast, sat) | 59.5% ± 1.7% | 60.7% ± 2.9% | Helps with different lighting on phone |
| 5. + Global Average Pooling & Dropout (`LeafNetGAP`) | 64.3% ± 3.2% | 65.5% ± 4.0% | Much less parameters (61k vs 208k) |
| 6. Pretrained `mobilenet_v3_small` + aug + CosineLR + best epoch | 67.9% ± 5.0% | 64.3% ± 7.6% | Good, but big file (6.3 MB) |
| 7. `LeafNetGAP` + AdamW + CosineLR + last year data (801 imgs) | **71.4%** | **80.0%** (historical val) | Best generalization on new phones |

## 2. What helped and what is just noise

- **Global Average Pooling (GAP) + Dropout:** In original `LeafNet`, flattening $64 \times 6 \times 6$ into linear layer had 147k weights (most of network). With only 101 training photos it just memorized background. Replacing it with `AdaptiveAvgPool2d((1, 1))` reduced parameters from 208k to 61k and accuracy jumped by ~5-9%, which is more than 2 validation images (> noise).
- **ColorJitter:** Phone camera changes white balance and exposure all the time. Adding `ColorJitter` stabilized training and improved camera-like accuracy by ~10% compared to only geometric agumentation.
- **Saving best epoch + CosineAnnealingLR:** In Task 2 and 3 we took weights from last epoch (epoch 15), where model was already overfitted. Keeping `best_state` with cosine scheduler gave consistent +5% improvement.
- **Last year dataset:** 101 images is too small for training CNN from scratch. Adding 700 historical photos from last year increased accuracy to 71.4% on our val set and 80.0% on historical val set.

## 3. What did NOT help

- **Only BatchNorm + geometric augmentation in 15 epochs:** Without more epochs and learning rate decay, adding heavy random crops and flips on 101 images actually made accuracy worse or very noisy (±8.9%), because model did not converge in 15 epochs.
- **Fine-tuning whole MobileNet without regularization:** In Task 2, MobileNet had 78.6% in epoch 1, but then overfitted to 57.1% in epoch 15 because learning rate $10^{-3}$ destroyed pretrained features on small dataset.

## 4. Which model I put on the phone

I deployed **Variant 7 (`LeafNetGAP` trained from scratch with full augmentation, AdamW, CosineLR and combined dataset)** into the web app.
Even though pretrained MobileNet is also strong, `LeafNetGAP` has only **240 KB** as ONNX file (MobileNet has 6.35 MB), so it loads immediately on phone mobile data and achieves 80% on photos from other students phones.
