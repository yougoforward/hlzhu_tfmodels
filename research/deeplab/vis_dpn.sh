#!/usr/bin/env bash
cd ..
CURRENT_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/slim
export PYTHONPATH=$PYTHONPATH:$CURRENT_DIR:$CURRENT_DIR/deeplab
cd ./deeplab
#python3 vis_class_aware2.py \
#  --logtostderr \
#  --vis_split="test" \
#  --colormap_type="cityscapes" \
#  --model_variant="xception_65" \
#  --atrous_rates=12 \
#  --atrous_rates=24 \
#  --atrous_rates=36 \
#  --output_stride=8 \
#  --decoder_output_stride=4\
#  --vis_crop_size=1025 \
#  --vis_crop_size=2049 \
#  --aspp_with_batch_norm=true\
#  --aspp_with_separable_conv=false\
#  --decoder_use_separable_conv=false\
#  --dataset="cityscapes" \
#  --checkpoint_dir="datasets/cityscapes/exp/dpcan/trainval"\
#  --vis_logdir="datasets/cityscapes/exp/dpcan/vis_test"\
#  --dataset_dir="datasets/cityscapes/tfrecord"\
#  --max_number_of_evaluations=1\
#  --eval_scales=0.5\
#  --eval_scales=0.75\
#  --eval_scales=1.0\
#  --eval_scales=1.25\
#  --eval_scales=1.5\
#  --eval_scales=1.75\
#  --add_flipped_images=true


#python3 vis_class_aware2.py \
#  --logtostderr \
#  --vis_split="test" \
#  --colormap_type="cityscapes" \
#  --model_variant="xception_65" \
#  --atrous_rates=12 \
#  --atrous_rates=24 \
#  --atrous_rates=36 \
#  --output_stride=8 \
#  --decoder_output_stride=4\
#  --vis_crop_size=1025 \
#  --vis_crop_size=2049 \
#  --aspp_with_batch_norm=true\
#  --aspp_with_separable_conv=false\
#  --decoder_use_separable_conv=false\
#  --dataset="cityscapes" \
#  --checkpoint_dir="datasets/cityscapes/exp/dpcan/train3"\
#  --vis_logdir="datasets/cityscapes/exp/dpcan/vis_test_train"\
#  --dataset_dir="datasets/cityscapes/tfrecord"\
#  --max_number_of_evaluations=1\
#  --eval_scales=0.5\
#  --eval_scales=0.75\
#  --eval_scales=1.0\
#  --eval_scales=1.25\
#  --eval_scales=1.5\
#  --eval_scales=1.75\
#  --add_flipped_images=true

#python3 vis_class_aware2.py \
#  --logtostderr \
#  --vis_split="test" \
#  --colormap_type="cityscapes" \
#  --model_variant="xception_65" \
#  --atrous_rates=12 \
#  --atrous_rates=24 \
#  --atrous_rates=36 \
#  --output_stride=8 \
#  --decoder_output_stride=4\
#  --vis_crop_size=1025 \
#  --vis_crop_size=2049 \
#  --aspp_with_batch_norm=true\
#  --aspp_with_separable_conv=false\
#  --decoder_use_separable_conv=false\
#  --dataset="cityscapes" \
#  --checkpoint_dir="datasets/cityscapes/exp/dpcan/train_val_os16"\
#  --vis_logdir="datasets/cityscapes/exp/dpcan/vis_test_trainval_os16"\
#  --dataset_dir="datasets/cityscapes/tfrecord"\
#  --max_number_of_evaluations=1\
#  --eval_scales=0.5\
#  --eval_scales=0.75\
#  --eval_scales=1.0\
#  --eval_scales=1.25\
#  --eval_scales=1.5\
#  --eval_scales=1.75\
#  --add_flipped_images=true

python3 vis_class_aware2.py \
  --logtostderr \
  --vis_split="test" \
  --colormap_type="cityscapes" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --decoder_output_stride=4\
  --vis_crop_size=1025 \
  --vis_crop_size=2049 \
  --aspp_with_batch_norm=true\
  --aspp_with_separable_conv=false\
  --decoder_use_separable_conv=false\
  --dataset="cityscapes" \
  --checkpoint_dir="datasets/cityscapes/exp/dpcan/trainval2"\
  --vis_logdir="datasets/cityscapes/exp/dpcan/vis_test_trainval2"\
  --dataset_dir="datasets/cityscapes/tfrecord"\
  --max_number_of_evaluations=1\
  --eval_scales=0.5\
  --eval_scales=0.75\
  --eval_scales=1.0\
  --eval_scales=1.25\
  --eval_scales=1.5\
  --eval_scales=1.75\
  --add_flipped_images=true

#python3 vis_class_aware2.py \
#  --logtostderr \
#  --vis_split="val" \
#  --colormap_type="cityscapes" \
#  --model_variant="xception_65" \
#  --atrous_rates=12 \
#  --atrous_rates=24 \
#  --atrous_rates=36 \
#  --output_stride=8 \
#  --decoder_output_stride=4\
#  --vis_crop_size=1025 \
#  --vis_crop_size=2049 \
#  --aspp_with_batch_norm=true\
#  --aspp_with_separable_conv=false\
#  --decoder_use_separable_conv=false\
#  --dataset="cityscapes" \
#  --checkpoint_dir="datasets/cityscapes/exp/dpcan/train3"\
#  --vis_logdir="datasets/cityscapes/exp/dpcan/vis_val"\
#  --dataset_dir="datasets/cityscapes/tfrecord"\
#  --max_number_of_evaluations=1\
#  --eval_scales=0.5\
#  --eval_scales=0.75\
#  --eval_scales=1.0\
#  --eval_scales=1.25\
#  --eval_scales=1.5\
#  --eval_scales=1.75\
#  --add_flipped_images=true