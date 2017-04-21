#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -e

export PYTHONUNBUFFERED="True"

DATA_PATH=$1
WEIGHT_PATH=$2

time python ./tools/extract_features.py --device gpu --device_id 0 \
  --weights ${WEIGHT_PATH} \
  --imdb voc_2007_train \
  --data_path ${DATA_PATH} \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network VGGnet_test \
  --max_number_images 6820 

# It was too hard to figure out how to properly parse max_images as a bash command arg,
# so I just hardcoded it to KITTI's dataset size. Sorry about that.
