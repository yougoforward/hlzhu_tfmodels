#!/usr/bin/env bash

sh merge2aspp_train.sh 2>&1 | tee merge2aspp_train.log

sh merge2aspp256_train.sh 2>&1 | tee merge2aspp256_train.log