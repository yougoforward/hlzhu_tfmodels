#!/usr/bin/env bash
#sh merge2aspp_train.sh 2>&1 | tee merge2aspp_train.log
sh dual_pyramid_train.sh 2>&1 | tee dual_pyramid_train8.log

sh dual_pyramid_train_with_boundary.sh 2>&1 | tee dual_pyramid_train_with_boundary8.log

