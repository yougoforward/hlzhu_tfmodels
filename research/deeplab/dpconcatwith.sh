#!/usr/bin/env bash
sh dual_pyramid_train_with_boundary.sh 2>&1 | tee dual_pyramid_train_with_boundary_concat256.log
sh dual_pyramid_train_x65_with_boundary.sh 2>&1 | tee dual_pyramid_train_x65_with_boundary_concat256.log