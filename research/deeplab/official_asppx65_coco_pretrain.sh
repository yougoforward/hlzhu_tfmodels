#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
python dual_pyramid_eval_fpn.py\
    --logtostderr\
    --eval_split="val"\
    --model_variant="xception_65"\
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --min_resize_value=513 \
    --max_resize_value=513 \
    --aspp_with_batch_norm=true \
    --aspp_with_separable_conv=false \
    --decoder_use_separable_conv=false \
    --dataset="pascal_voc_seg" \
    --checkpoint_dir="datasets/pascal_voc_seg/init_models/xception_65_coco_pretrained" \
    --eval_logdir="datasets/coco/exp/xception_65_coco_pretrained/eval" \
    --dataset_dir="datasets/coco/tfrecord" \
    --max_number_of_evaluations=1



