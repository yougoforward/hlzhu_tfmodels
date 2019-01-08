#!/usr/bin/env bash
sh dual_pyramid_train_v3plusCam.sh 2>&1| tee dual_pyramid_train_skip3x3_v3plusCam_add256.log
sh dual_pyramid_train_v3plusCam_x65.sh 2>&1| tee dual_pyramid_train_skip3x3_v3plusCam_x65_add256.log