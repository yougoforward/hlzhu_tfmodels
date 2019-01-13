#!/usr/bin/env bash
sh dual_pyramid_train_fpn.sh 2>&1 | tee dual_pyramid_train_fpn_concat256.log
sh dual_pyramid_train_x65_fpn.sh 2>&1 | tee dual_pyramid_train_x65_fpn_concat256.log