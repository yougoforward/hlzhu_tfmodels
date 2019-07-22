#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
python export_dpcam_model.py \
  --logtostderr \
  --checkpoint_path="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13/model.ckpt-30000" \
  --export_path="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13/frozen_inference_graph.pb" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0