#!/usr/bin/env bash
sh dual_pyramid_train.sh 2>&1 | tee dual_pyramid_train_skip3x3_add256.log
sh dual_pyramid_train_x65.sh 2>&1 | tee dual_pyramid_train_skip3x3_x65_add256.log