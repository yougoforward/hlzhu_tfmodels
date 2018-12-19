#!/usr/bin/env bash

sh pmerge2aspp_train.sh 2>&1 | tee pmerge2aspp_train.log

sh pmerge2aspp256_train.sh 2>&1 | tee pmerge2aspp256_train.log