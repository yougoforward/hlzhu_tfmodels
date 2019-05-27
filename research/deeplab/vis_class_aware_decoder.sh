#!/usr/bin/env bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
python vis_class_aware2.py \
  --logtostderr \
  --vis_split="test" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --decoder_output_stride=4\
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --aspp_with_batch_norm=true\
  --aspp_with_separable_conv=false\
  --decoder_use_separable_conv=false\
  --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/trainval_finetune13"\
  --vis_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/vis_trainval_finetune13"\
  --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
  --max_number_of_iterations=1\
  --eval_scales=0.5\
  --eval_scales=0.75\
  --eval_scales=1.0\
  --eval_scales=1.25\
  --eval_scales=1.5\
  --eval_scales=1.75\
  --add_flipped_images=true

python vis_class_aware2.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --decoder_output_stride=4\
  --vis_crop_size=513 \
  --vis_crop_size=513 \
  --aspp_with_batch_norm=true\
  --aspp_with_separable_conv=false\
  --decoder_use_separable_conv=false\
  --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13"\
  --vis_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/vis_train_finetune13"\
  --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
  --max_number_of_iterations=1\
  --eval_scales=0.5\
  --eval_scales=0.75\
  --eval_scales=1.0\
  --eval_scales=1.25\
  --eval_scales=1.5\
  --eval_scales=1.75\
  --add_flipped_images=true