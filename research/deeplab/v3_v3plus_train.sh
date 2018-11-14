#!/usr/bin/env bash

sh v3train.sh 2>&1 | tee v3train.log

sh v3train_res50.sh 2>&1 | tee v3train_res50.log

sh v3train_res101.sh 2>&1 | tee v3train_res101.log

sh v3plustrain.sh 2>&1 | tee v3plustrain.log
