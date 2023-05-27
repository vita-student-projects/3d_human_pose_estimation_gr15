TRAIN_CONFIGS='configs/v1_resnet_3dpw_ft.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.gpu)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.train --configs_yml=${TRAIN_CONFIGS}