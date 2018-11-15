#!/usr/bin/env bash

python vis_class_aware2.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --aspp_with_batch_norm=true\
  --aspp_with_separable_conv=false\
  --decoder_use_separable_conv=false\
  --checkpoint_dir="datasets/pascal_voc_seg/exp/class_aware_train13_on_trainaug_set/train_finetune"\
  --vis_logdir="datasets/pascal_voc_seg/exp/class_aware_train13_on_trainaug_set/vis_finetune"\
  --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
  --max_number_of_iterations=1