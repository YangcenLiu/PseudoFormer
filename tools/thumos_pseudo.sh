#!/bin/bash

echo "start training"
# python train.py ./configs/thumos_i3d.yaml --output origin

python train.py ./configs/thumos_i3d_pseudo.yaml --output pseudo


# echo "start testing..."
# python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_pseudo_pseudo1/epoch_039.pth.tar
# python eval.py ./configs/thumos_i3d.yaml ckpt/thumos_i3d_tridet/epoch_039.pth.tar
# python eval.py ./configs/thumos_i3d_aformer.yaml ckpt/thumos_i3d_aformer/epoch_034.pth.tar