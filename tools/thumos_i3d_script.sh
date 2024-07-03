#!/bin/bash

echo "start training"
# python train.py ./configs/thumos_i3d.yaml --output origin
python train.py ./configs/thumos_i3d_pseudo.yaml --output weak
echo "start testing..."
# python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_tridet/epoch_039.pth.tar
# python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pseudo_weak/epoch_039.pth.tar
# python eval.py ./configs/thumos_i3d_aformer.yaml ckpt/thumos_i3d_aformer/epoch_034.pth.tar

# << results
# |tIoU = 0.10: mAP = 87.10 (%)
# |tIoU = 0.20: mAP = 86.24 (%)
# |tIoU = 0.30: mAP = 83.63 (%)
# |tIoU = 0.40: mAP = 80.33 (%)
# |tIoU = 0.50: mAP = 73.04 (%)
# |tIoU = 0.60: mAP = 62.16 (%)
# |tIoU = 0.70: mAP = 46.58 (%)
# Avearge mAP: 74.15 (%) (69.15 (%))