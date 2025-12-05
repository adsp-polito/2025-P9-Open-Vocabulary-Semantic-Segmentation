# export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS='[root path of datasets]'
export RSIB_CKPT='[path to RSIB CKPT]'

RESULTS='[expected path of training results]'

sh scripts/train.sh configs/vitb_384.yaml 2 $RESULTS \
SOLVER.IMS_PER_BATCH 4 \
SOLVER.MAX_ITER 30002 \
DATALOADER.NUM_WORKERS 8 \

sh scripts/eval.sh configs/vitb_384.yaml 2 $RESULTS/Eval \
MODEL.WEIGHTS $RESULTS/model_0029999.pth

