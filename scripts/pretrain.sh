
TRAIN_CONFIGS='configs/pretrain.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.gpu)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.pretrain --configs_yml=${TRAIN_CONFIGS} 