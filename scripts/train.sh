#!/bin/sh
export DETECTRON2_DATASETS='gs_net/data/datasets'
# export RSIB_CKPT='dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
export RSIB_CKPT='gs_net/third_party/experiments/sup_unfreeze_20260318_225917/backbone_for_gsnet/epoch_12.pth'

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

# FloodNet
python3 train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 MODEL.SEM_SEG_HEAD.IGNORE_VALUE 0 \
 MODEL.SEM_SEG_HEAD.NUM_CLASSES 40 \
 MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON "datasets/landdiscover.json" \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/floodnet.json" \
 TEST.EVAL_PERIOD 0 \
 DATASETS.TRAIN \(\"LandDiscover_50K\"\,\) \
 DATASETS.TEST \(\"FloodNet\"\,\) \
 $opts