echo "start training"
# python train.py ./configs/anet_i3d_aformer.yaml --output aformer200
python train.py ./configs/anet_i3d1_pseudo.yaml --output anetpseudo200
echo "start testing..."
python eval.py ./configs/anet_i3d1_pseudo.yaml ckpt/anet_i3d1_pseudo_anetpseudo200/model_best.pth.tar
# python eval.py ./configs/anet_i3d_aformer.yaml ckpt/anet_i3d_aformer_aformer200/model_best.pth.tar
