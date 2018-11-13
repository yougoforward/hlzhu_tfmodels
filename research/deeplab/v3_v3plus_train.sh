#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=6,7 sh v3train.sh 2>&1 | tee v3train.log

CUDA_VISIBLE_DEVICES=6,7 sh v3train_res50.sh 2>&1 | tee v3train_res50.log

CUDA_VISIBLE_DEVICES=6,7 sh v3train_res101.sh 2>&1 | tee v3train_res101.log

CUDA_VISIBLE_DEVICES=6,7 sh v3plustrain.sh 2>&1 | tee v3plustrain.log
