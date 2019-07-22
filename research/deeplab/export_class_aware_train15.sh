#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab

python export_cam_model.py \
  --logtostderr \
  --checkpoint_path="datasets/pascal_voc_seg/exp/class_aware_train15_on_trainaug_set/train_finetune2/model.ckpt-30000" \
  --export_path="datasets/pascal_voc_seg/exp/class_aware_train15_on_trainaug_set/train_finetune2/frozen_inference_graph.pb" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
  --aspp_with_batch_norm=true\
  --aspp_with_separable_conv=false\
  --decoder_use_separable_conv=false\
  --inference_scales=1.0