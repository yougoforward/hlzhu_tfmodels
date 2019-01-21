#!/usr/bin/env bash

cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
python dual_pyramid_eval_v3plusCam.py\
    --logtostderr\
    --eval_split="val"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --decoder_output_stride=4\
    --eval_crop_size=513\
    --eval_crop_size=513\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="pascal_voc_seg"\
    --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_v3plusCam_xception65_on_trainaug_set/train_finetune13"\
    --eval_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_v3plusCam_xception65_on_trainaug_set/eval_finetune13"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1\
    --eval_scales=0.5\
    --eval_scales=0.75\
    --eval_scales=1.0\
    --eval_scales=1.25\
    --eval_scales=1.5\
    --eval_scales=1.75
#    --add_flipped_images=true