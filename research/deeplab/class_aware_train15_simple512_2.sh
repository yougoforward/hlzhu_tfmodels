#!/bin/bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
python class_aware_train15_simple512_2.py\
    --logtostderr \
    --num_clones=2 \
    --train_split="train_aug"\
    --model_variant="resnet_v1_50_beta"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --train_crop_size=513\
    --train_crop_size=513\
    --train_batch_size=16\
    --multi_grid=1\
    --multi_grid=2\
    --multi_grid=4\
    --training_number_of_steps=30000\
    --fine_tune_batch_norm=true\
    --base_learning_rate=0.007\
    --weight_decay=0.0001\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="pascal_voc_seg"\
    --tf_initial_checkpoint="datasets/pascal_voc_seg/init_models/resnet_v1_50/model.ckpt"\
    --train_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_res50_multigrad_on_trainaug_set/train2"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

python class_aware_eval15_simple512_2.py\
    --logtostderr\
    --eval_split="val"\
    --model_variant="resnet_v1_50_beta"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --multi_grid=1\
    --multi_grid=2\
    --multi_grid=4\
    --eval_crop_size=513\
    --eval_crop_size=513\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="pascal_voc_seg"\
    --checkpoint_dir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_res50_multigrad_on_trainaug_set/train2"\
    --eval_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_res50_multigrad_on_trainaug_set/eval2"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1

python class_aware_train15_simple512_2.py\
    --logtostderr\
    --num_clones=2 \
    --train_split="train_aug"\
    --model_variant="resnet_v1_50_beta"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --multi_grid=1\
    --multi_grid=2\
    --multi_grid=4\
    --train_crop_size=513\
    --train_crop_size=513\
    --train_batch_size=16\
    --training_number_of_steps=30000\
    --fine_tune_batch_norm=false\
    --base_learning_rate=0.001\
    --weight_decay=0.0001\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="pascal_voc_seg"\
    --tf_initial_checkpoint="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_res50_multigrad_on_trainaug_set/train2/model.ckpt-30000"\
    --train_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_res50_multigrad_on_trainaug_set/train_finetune2"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

python class_aware_eval15_simple512_2.py
    --logtostderr\
    --eval_split="val"\
    --model_variant="resnet_v1_50_beta"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --multi_grid=1\
    --multi_grid=2\
    --multi_grid=4\
    --eval_crop_size=513\
    --eval_crop_size=513\
    --aspp_with_batch_norm=true\
    --aspp_with_separable_conv=false\
    --decoder_use_separable_conv=false\
    --dataset="pascal_voc_seg"\
    --checkpoint_dir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_res50_multigrad_on_trainaug_set/train_finetune2"\
    --eval_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_res50_multigrad_on_trainaug_set/eval_finetune2"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1