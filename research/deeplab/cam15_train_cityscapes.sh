#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab

#python3 dual_pyramid_train_fpn.py \
#    --logtostderr \
#    --num_clones=2 \
#    --train_split="train"\
#    --model_variant="xception_65"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --train_crop_size=769\
#    --train_crop_size=769\
#    --train_batch_size=8\
#    --decoder_output_stride=4 \
#    --training_number_of_steps=90000\
#    --fine_tune_batch_norm=true\
#    --base_learning_rate=0.007\
#    --weight_decay=0.00004\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="cityscapes"\
#    --tf_initial_checkpoint="datasets/pascal_voc_seg/init_models/xception/model.ckpt"\
#    --train_logdir="datasets/cityscapes/exp/dpcan/train2"\
#    --dataset_dir="datasets/cityscapes/tfrecord"

#python3 dual_pyramid_eval_fpn.py \
#    --logtostderr \
#    --eval_split="val" \
#    --model_variant="xception_65" \
#    --atrous_rates=12 \
#    --atrous_rates=24 \
#    --atrous_rates=36 \
#    --output_stride=8 \
#    --decoder_output_stride=4 \
#    --eval_crop_size=1025 \
#    --eval_crop_size=2049 \
#    --weight_decay=0.00004\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="cityscapes" \
#    --checkpoint_dir="datasets/cityscapes/exp/dpcan/train2"\
#    --eval_logdir="datasets/cityscapes/exp/dpcan/eval2"\
#    --dataset_dir="datasets/cityscapes/tfrecord"\
#    --max_number_of_iterations=1\
#    --eval_scales=0.75\
#    --eval_scales=1.0\
#    --eval_scales=1.25\
#    --add_flipped_images=true


python3 dual_pyramid_train_fpn.py \
    --logtostderr \
    --num_clones=2 \
    --train_split="train"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --train_crop_size=769\
    --train_crop_size=769\
    --train_batch_size=8\
    --decoder_output_stride=4 \
    --training_number_of_steps=90000\
    --fine_tune_batch_norm=false\
    --base_learning_rate=0.001\
    --weight_decay=0.00004\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="cityscapes"\
    --tf_initial_checkpoint="datasets/cityscapes/exp/dpcan/train2/model.ckpt-90000"\
    --train_logdir="datasets/cityscapes/exp/dpcan/train_finetune2"\
    --dataset_dir="datasets/cityscapes/tfrecord"
##
python3 dual_pyramid_eval_fpn.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="cityscapes" \
    --checkpoint_dir="datasets/cityscapes/exp/dpcan/train_finetune2"\
    --eval_logdir="datasets/cityscapes/exp/dpcan/eval_finetune2"\
    --dataset_dir="datasets/cityscapes/tfrecord"\
    --max_number_of_iterations=1
#    --eval_scales=0.75\
#    --eval_scales=1.0\
#    --eval_scales=1.25\
#    --add_flipped_images=true
#
#python3 train_cityscapes.py \
#    --logtostderr \
#    --num_clones=2 \
#    --train_split="train_val"\
#    --model_variant="xception_65"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --train_crop_size=769\
#    --train_crop_size=769\
#    --train_batch_size=8\
#    --decoder_output_stride=4 \
#    --training_number_of_steps=90000\
#    --fine_tune_batch_norm=true\
#    --base_learning_rate=0.007\
#    --weight_decay=0.00004\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="cityscapes"\
#    --tf_initial_checkpoint="datasets/cityscapes/exp/dpcan/train/model.ckpt-90000"\
#    --train_logdir="datasets/cityscapes/exp/dpcan/train_finetune"\
#    --dataset_dir="datasets/cityscapes/tfrecord"
#
#python3 vis_cityscapes.py \
#    --logtostderr \
#    --eval_split="test" \
#    --model_variant="xception_65" \
#    --atrous_rates=6 \
#    --atrous_rates=12 \
#    --atrous_rates=18 \
#    --output_stride=16 \
#    --decoder_output_stride=4 \
#    --eval_crop_size=1025 \
#    --eval_crop_size=2049 \
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="cityscapes" \
#    --checkpoint_dir="datasets/cityscapes/exp/dpcan/train_finetune"\
#    --eval_logdir="datasets/cityscapes/exp/dpcan/eval_finetune"\
#    --dataset_dir="datasets/cityscapes/tfrecord"
