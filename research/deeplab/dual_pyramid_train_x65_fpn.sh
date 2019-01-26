#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
python dual_pyramid_train_fpn.py\
    --logtostderr \
    --num_clones=4 \
    --train_split="train_aug"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --train_crop_size=513\
    --train_crop_size=513\
    --train_batch_size=16\
    --decoder_output_stride=4\
    --training_number_of_steps=60000\
    --fine_tune_batch_norm=true\
    --base_learning_rate=0.007\
    --weight_decay=0.00004\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="pascal_voc_seg"\
    --tf_initial_checkpoint="datasets/pascal_voc_seg/init_models/xception/model.ckpt"\
    --train_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train115"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"
#
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
    --checkpoint_dir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train15"\
    --eval_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/eval15"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1
#
#python dual_pyramid_train_fpn.py\
#    --logtostderr\
#    --num_clones=2 \
#    --train_split="train_aug"\
#    --model_variant="xception_65"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --decoder_output_stride=4\
#    --train_crop_size=513\
#    --train_crop_size=513\
#    --train_batch_size=16\
#    --training_number_of_steps=30000\
#    --fine_tune_batch_norm=false\
#    --base_learning_rate=0.001\
#    --weight_decay=0.00004\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --tf_initial_checkpoint="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train12/model.ckpt-60000"\
#    --train_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

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

#python dual_pyramid_train_fpn.py\
#    --logtostderr\
#    --num_clones=2 \
#    --train_split="trainval"\
#    --model_variant="xception_65"\
#    --atrous_rates=12\
#    --atrous_rates=24\
#    --atrous_rates=36\
#    --output_stride=8\
#    --decoder_output_stride=4\
#    --train_crop_size=513\
#    --train_crop_size=513\
#    --train_batch_size=8\
#    --training_number_of_steps=30000\
#    --fine_tune_batch_norm=false\
#    --base_learning_rate=0.001\
#    --weight_decay=0.00004\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --tf_initial_checkpoint="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/train_finetune13/model.ckpt-30000"\
#    --train_logdir="datasets/pascal_voc_seg/exp/dual_pyramid_train_x65_fpn_on_trainaug_set/trainval_finetune13"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"