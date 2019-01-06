#!/bin/bash
python class_aware_train15_nosigmoidsoftmax1loss.py\
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
    --train_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_nosigmoidsoftmax1loss_res50_multigrad_on_trainaug_set/train"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

python class_aware_eval15.py\
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
    --checkpoint_dir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_nosigmoidsoftmax1loss_res50_multigrad_on_trainaug_set/train"\
    --eval_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_nosigmoidsoftmax1loss_res50_multigrad_on_trainaug_set/eval"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1
#
python class_aware_train15_nosigmoidsoftmax1loss.py\
    --logtostderr\
    --num_clones=2\
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
    --tf_initial_checkpoint="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_nosigmoidsoftmax1loss_res50_multigrad_on_trainaug_set/train/model.ckpt-30000"\
    --train_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_nosigmoidsoftmax1loss_res50_multigrad_on_trainaug_set/train_finetune_001_os8"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

python class_aware_eval15.py\
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
    --checkpoint_dir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_nosigmoidsoftmax1loss_res50_multigrad_on_trainaug_set/train_finetune_001_os8"\
    --eval_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_add2_256256256_nosigmoidsoftmax1loss_res50_multigrad_on_trainaug_set/eval_finetune_001_os8"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1
#    --eval_scales=0.5\
#    --eval_scales=0.75\
#    --eval_scales=1.0\
#    --eval_scales=1.25\
#    --eval_scales=1.5\
#    --eval_scales=1.75\
#    --add_flipped_images=true

#
#python class_aware_train15.py\
#    --logtostderr\
#    --num_clones=2\
#    --train_split="trainval"\
#    --model_variant="resnet_v1_50_beta"\
#    --atrous_rates=6\
#    --atrous_rates=12\
#    --atrous_rates=18\
#    --output_stride=16\
#    --multi_grid=1\
#    --multi_grid=2\
#    --multi_grid=4\
#    --train_crop_size=513\
#    --train_crop_size=513\
#    --train_batch_size=16\
#    --training_number_of_steps=30000\
#    --fine_tune_batch_norm=false\
#    --base_learning_rate=0.001\
#    --weight_decay=0.0001\
#    --aspp_with_batch_norm=true\
#    --aspp_with_separable_conv=false\
#    --decoder_use_separable_conv=false\
#    --dataset="pascal_voc_seg"\
#    --tf_initial_checkpoint="datasets/pascal_voc_seg/exp/class_aware_train15_res50_multigrad_on_trainaug_set/train_finetune/model.ckpt-30000"\
#    --train_logdir="datasets/pascal_voc_seg/exp/class_aware_train15_res50_multigrad_on_trainaug_set/test_finetune_001"\
#    --dataset_dir="datasets/pascal_voc_seg/tfrecord"