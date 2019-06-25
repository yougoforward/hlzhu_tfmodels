#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab

python3 dual_pyramid_train_fpn.py \
    --logtostderr \
    --num_clones=4 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --min_resize_value=513 \
    --max_resize_value=513 \
    --resize_factor=16 \
    --decoder_output_stride=4 \
    --training_number_of_steps=150000 \
    --fine_tune_batch_norm=true \
    --base_learning_rate=0.02 \
    --weight_decay=0.00004 \
    --aspp_with_batch_norm=true \
    --aspp_with_separable_conv=false \
    --decoder_use_separable_conv=false \
    --dataset="ade20k" \
    --tf_initial_checkpoint="datasets/pascal_voc_seg/init_models/xception/model.ckpt" \
    --train_logdir="datasets/ADE20K/exp/dpcan/train150" \
    --dataset_dir="datasets/ADE20K/tfrecord"


python3 dual_pyramid_eval_fpn.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --min_resize_value=513 \
    --max_resize_value=513 \
    --resize_factor=16 \
    --aspp_with_batch_norm=true \
    --aspp_with_separable_conv=false \
    --decoder_use_separable_conv=false \
    --dataset="ade20k" \
    --checkpoint_dir="datasets/ADE20K/exp/dpcan/train150"\
    --eval_logdir="datasets/ADE20K/exp/dpcan/eval150"\
    --dataset_dir="datasets/ADE20K/tfrecord"\
    --max_number_of_evaluations=1

python3 dual_pyramid_eval_fpn.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=12 \
    --atrous_rates=24 \
    --atrous_rates=36 \
    --output_stride=8 \
    --decoder_output_stride=4 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --min_resize_value=513 \
    --max_resize_value=513 \
    --resize_factor=8 \
    --aspp_with_batch_norm=true \
    --aspp_with_separable_conv=false \
    --decoder_use_separable_conv=false \
    --dataset="ade20k" \
    --checkpoint_dir="datasets/ADE20K/exp/dpcan/train150"\
    --eval_logdir="datasets/ADE20K/exp/dpcan/eval150_os8"\
    --dataset_dir="datasets/ADE20K/tfrecord"\
    --max_number_of_evaluations=1


python3 dual_pyramid_eval_fpn.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=12 \
    --atrous_rates=24 \
    --atrous_rates=36 \
    --output_stride=8 \
    --decoder_output_stride=4 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --min_resize_value=513 \
    --max_resize_value=513 \
    --resize_factor=8 \
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="ade20k" \
    --checkpoint_dir="datasets/ADE20K/exp/dpcan/train150"\
    --eval_logdir="datasets/ADE20K/exp/dpcan/eval150_os8_05175"\
    --dataset_dir="datasets/ADE20K/tfrecord"\
    --max_number_of_evaluations=1\
    --eval_scales=0.5 \
    --eval_scales=0.75 \
    --eval_scales=1.0 \
    --eval_scales=1.25 \
    --eval_scales=1.5 \
    --eval_scales=1.75 \
    --add_flipped_images=true


