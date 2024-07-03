##!/bin/bash

echo "start training"
# python train.py ./configs/anet_i3d_aformer.yaml --output aformer200
# python train.py ./configs/anet_i3d1.yaml --output tridet1
echo "start testing..."
python eval.py ./configs/anet_i3d1.yaml ckpt/anet_i3d1_tridet1/model_best.pth.tar
# python eval.py ./configs/anet_i3d_aformer.yaml ckpt/anet_i3d_aformer_aformer200/model_best.pth.tar

# << results

# aformer 1 classes
# |tIoU = 0.50: mAP = 54.24 (%)
# |tIoU = 0.55: mAP = 50.86 (%)
# |tIoU = 0.60: mAP = 47.55 (%)
# |tIoU = 0.65: mAP = 44.48 (%)
# |tIoU = 0.70: mAP = 40.62 (%)
# |tIoU = 0.75: mAP = 36.24 (%)
# |tIoU = 0.80: mAP = 31.17 (%)
# |tIoU = 0.85: mAP = 24.76 (%)
# |tIoU = 0.90: mAP = 17.70 (%)
# |tIoU = 0.95: mAP = 8.68 (%)
# Avearge mAP: 35.63 (%)

# tridet 1 classes
# |tIoU = 0.50: mAP = 54.29 (%)
# |tIoU = 0.55: mAP = 51.25 (%)
# |tIoU = 0.60: mAP = 47.98 (%)
# |tIoU = 0.65: mAP = 44.83 (%)
# |tIoU = 0.70: mAP = 40.88 (%)
# |tIoU = 0.75: mAP = 36.69 (%)
# |tIoU = 0.80: mAP = 31.68 (%)
# |tIoU = 0.85: mAP = 25.88 (%)
# |tIoU = 0.90: mAP = 18.53 (%)
# |tIoU = 0.95: mAP = 8.25 (%)
# Avearge mAP: 36.03 (%)


# tridet200 classes
# |tIoU = 0.50: mAP = 48.06 (%)
# |tIoU = 0.55: mAP = 44.98 (%)
# |tIoU = 0.60: mAP = 42.16 (%)
# |tIoU = 0.65: mAP = 38.87 (%)
# |tIoU = 0.70: mAP = 35.23 (%)
# |tIoU = 0.75: mAP = 31.12 (%)
# |tIoU = 0.80: mAP = 26.29 (%)
# |tIoU = 0.85: mAP = 20.63 (%)
# |tIoU = 0.90: mAP = 14.10 (%)
# |tIoU = 0.95: mAP = 5.80 (%)
# Avearge mAP: 30.72 (%)


# aformer200
# |tIoU = 0.50: mAP = 46.48 (%)
# |tIoU = 0.55: mAP = 43.75 (%)
# |tIoU = 0.60: mAP = 40.87 (%)
# |tIoU = 0.65: mAP = 38.12 (%)
# |tIoU = 0.70: mAP = 34.75 (%)
# |tIoU = 0.75: mAP = 31.39 (%)
# |tIoU = 0.80: mAP = 27.06 (%)
# |tIoU = 0.85: mAP = 22.04 (%)
# |tIoU = 0.90: mAP = 15.15 (%)
# |tIoU = 0.95: mAP = 6.48 (%)
# Avearge mAP: 30.61 (%)