#!/usr/bin/env bash
sh v3_pascal_train512.sh 2>&1 | tee v3_pascal_train512.log
sh v3plus_pascal_train512.sh 2>&1 | tee v3plus_pascal_train512.log
