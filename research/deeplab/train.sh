#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1
python train.py\
    --logtostderr \
    --num_clones=2 \
    --train_split="train_aug"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --train_crop_size=513\
    --train_crop_size=513\
    --train_batch_size=16\
    --training_number_of_steps=30000\
    --fine_tune_batch_norm=true\
    --base_learning_rate=0.007\
    --dataset="pascal_voc_seg"\
    --tf_initial_checkpoint="datasets/pascal_voc_seg/init_models/xception/model.ckpt"\
    --train_logdir="datasets/pascal_voc_seg/exp/train_on_trainaug_set/train"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

python train.py\
    --logtostderr\
    --num_clones=2\
    --train_split="train"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=8\
    --train_crop_size=513\
    --train_crop_size=513\
    --train_batch_size=8\
    --training_number_of_steps=30000\
    --fine_tune_batch_norm=false\
    --base_learning_rate=0.001\
    --dataset="pascal_voc_seg"\
    --tf_initial_checkpoint="datasets/pascal_voc_seg/exp/train_on_trainaug_set/train/model.ckpt-30000"\
    --train_logdir="datasets/pascal_voc_seg/exp/train_on_trainaug_set/train_finetune"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"

python eval.py\
    --logtostderr\
    --eval_split="val"\
    --model_variant="xception_65"\
    --atrous_rates=6\
    --atrous_rates=12\
    --atrous_rates=18\
    --output_stride=16\
    --eval_crop_size=513\
    --eval_crop_size=513\
    --dataset="pascal_voc_seg"\
    --checkpoint_dir="datasets/pascal_voc_seg/exp/train_on_trainaug_set/train_finetune"\
    --eval_logdir="datasets/pascal_voc_seg/exp/train_on_trainaug_set/eval_finetune"\
    --dataset_dir="datasets/pascal_voc_seg/tfrecord"\
    --max_number_of_evaluations=1449