# export CUDA_VISIBLE_DEVICES=1
export DETECTRON2_DATASETS='gs_net/data/datasets'
export RSIB_CKPT='dinov3/vitl16-sat493m/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'

RESULTS='output/floodnet_training'

sh scripts/train.sh configs/rn101_224.yaml 1 $RESULTS \
SOLVER.IMS_PER_BATCH 4 \
SOLVER.MAX_ITER 30002 \
DATALOADER.NUM_WORKERS 8 \

sh scripts/eval.sh configs/rn101_224.yaml 1 $RESULTS \
MODEL.WEIGHTS $RESULTS/model_final.pth
