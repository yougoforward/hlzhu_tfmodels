#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
python train.py\
    --logtostderr \
    --num_clones=8 \
    --train_split="train"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --train_crop_size=513\
    --train_crop_size=513\
    --train_batch_size=64\
    --decoder_output_stride=4\
    --training_number_of_steps=50000\
    --fine_tune_batch_norm=true\
    --base_learning_rate=0.04\
    --weight_decay=0.00004\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=true\
    --decoder_use_separable_conv=true\
    --dataset="cocovoc"\
    --tf_initial_checkpoint="datasets/pascal_voc_seg/init_models/xception/model.ckpt"\
    --train_logdir="datasets/cocovoc/exp/x65_train_set/train"\
    --dataset_dir="datasets/cocovoc/tfrecord"


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
    --checkpoint_dir="datasets/cocovoc/exp/x65_train_set/train" \
    --eval_logdir="datasets/cocovoc/exp/x65_train_set/val" \
    --dataset_dir="datasets/pascal_voc_seg/tfrecord" \
    --max_number_of_evaluations=1

python dual_pyramid_train_fpn.py\
    --logtostderr \
    --num_clones=8 \
    --train_split="train_aug"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --train_crop_size=513\
    --train_crop_size=513\
    --train_batch_size=32\
    --decoder_output_stride=4\
    --training_number_of_steps=30000\
    --fine_tune_batch_norm=true\
    --base_learning_rate=0.002\
    --weight_decay=0.00004\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="pascal_voc_seg"\
    --tf_initial_checkpoint="datasets/cocovoc/exp/x65_train_set/train/model.ckpt-50000"\
    --train_logdir="datasets/cocovoc/exp/x65_train_aug_set/train"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

python dual_pyramid_eval_fpn.py\
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
    --checkpoint_dir="datasets/cocovoc/exp/x65_train_aug_set/train"\
    --eval_logdir="datasets/cocovoc/exp/x65_train_aug_set/val"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1

##
#python dual_pyramid_train_fpn.py\
#    --logtostderr\
#    --num_clones=8 \
#    --train_split="train_aug"\
#    --model_variant="xception_65"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --decoder_output_stride=4\
#    --train_crop_size=513\
#    --train_crop_size=513\
#    --train_batch_size=64\
#    --training_number_of_steps=7500\
#    --fine_tune_batch_norm=false\
#    --base_learning_rate=0.0001\
#    --weight_decay=0.00004\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --tf_initial_checkpoint="datasets/coco/exp/dpx65_voc_trainaug_set/train64/model.ckpt-15000"\
#    --train_logdir="datasets/coco/exp/dpx65_voc_trainaug_set/finetune64"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"
#
#python dual_pyramid_eval_fpn.py\
#    --logtostderr\
#    --eval_split="val"\
#    --model_variant="xception_65"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --decoder_output_stride=4\
#    --eval_crop_size=513\
#    --eval_crop_size=513\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --checkpoint_dir="datasets/coco/exp/dpx65_voc_trainaug_set/finetune64"\
#    --eval_logdir="datasets/coco/exp/dpx65_voc_trainaug_set/val_finetune64"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
#    --max_number_of_evaluations=1
#    --eval_scales=0.5\
#    --eval_scales=0.75\
#    --eval_scales=1.0\
#    --eval_scales=1.25\
#    --eval_scales=1.5\
#    --eval_scales=1.75 \
#    --add_flipped_images=true

#python dual_pyramid_eval_fpn.py\
#    --logtostderr\
#    --eval_split="val"\
#    --model_variant="xception_65"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --decoder_output_stride=4\
#    --eval_crop_size=513\
#    --eval_crop_size=513\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13"\
#    --eval_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/eval_finetune13"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
#    --max_number_of_evaluations=1\
#    --eval_scales=0.5\
#    --eval_scales=0.75\
#    --eval_scales=1.0\
#    --eval_scales=1.25\
#    --eval_scales=1.5\
#    --eval_scales=1.75
##    --add_flipped_images=true
#
#python dual_pyramid_eval_fpn.py\
#    --logtostderr\
#    --eval_split="val"\
#    --model_variant="xception_65"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --decoder_output_stride=4\
#    --eval_crop_size=513\
#    --eval_crop_size=513\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13"\
#    --eval_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/eval_finetune13"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
#    --max_number_of_evaluations=1\
#    --eval_scales=0.5\
#    --eval_scales=0.75\
#    --eval_scales=1.0\
#    --eval_scales=1.25\
#    --eval_scales=1.5\
#    --eval_scales=1.75\
#    --add_flipped_images=true
#
#python dual_pyramid_eval_fpn.py\
#    --logtostderr\
#    --eval_split="val"\
#    --model_variant="xception_65"\
#    --atrous_rates=12\
#    --atrous_rates=24\
#    --atrous_rates=36\
#    --output_stride=8\
#    --decoder_output_stride=4\
#    --eval_crop_size=513\
#    --eval_crop_size=513\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13"\
#    --eval_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/eval_finetune13"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
#    --max_number_of_evaluations=1
##    --eval_scales=0.5\
##    --eval_scales=0.75\
##    --eval_scales=1.0\
##    --eval_scales=1.25\
##    --eval_scales=1.5\
##    --eval_scales=1.75\
##    --add_flipped_images=true
#
#python dual_pyramid_eval_fpn.py\
#    --logtostderr\
#    --eval_split="val"\
#    --model_variant="xception_65"\
#    --atrous_rates=12\
#    --atrous_rates=24\
#    --atrous_rates=36\
#    --output_stride=8\
#    --decoder_output_stride=4\
#    --eval_crop_size=513\
#    --eval_crop_size=513\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13"\
#    --eval_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/eval_finetune13"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
#    --max_number_of_evaluations=1\
#    --eval_scales=0.5\
#    --eval_scales=0.75\
#    --eval_scales=1.0\
#    --eval_scales=1.25\
#    --eval_scales=1.5\
#    --eval_scales=1.75
##    --add_flipped_images=true
#
#python dual_pyramid_train_fpn.py\
#    --logtostderr\
#    --num_clones=8 \
#    --train_split="trainval"\
#    --model_variant="xception_65"\
#    --atrous_rates=12\
#    --atrous_rates=24\
#    --atrous_rates=36\
#    --output_stride=8\
#    --decoder_output_stride=4\
#    --train_crop_size=513\
#    --train_crop_size=513\
#    --train_batch_size=32\
#    --training_number_of_steps=7500\
#    --fine_tune_batch_norm=false\
#    --base_learning_rate=0.0001\
#    --weight_decay=0.00004\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --tf_initial_checkpoint="datasets/coco/exp/dpx65_voc_trainaug_set/finetune64/model.ckpt-7500"\
#    --train_logdir="datasets/coco/exp/dpx65_voc_trainval_set/train32"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"