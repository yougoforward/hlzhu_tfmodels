#!/usr/bin/env bash
sh dual_pyramid_train.sh 2>&1 | tee dual_pyramid_train_skip3x3_concat256.log
sh dual_pyramid_train_x65.sh 2>&1 | tee dual_pyramid_train_x65_skip3x3__concat256.log