#!/bin/bash

#training convnet
NUM_GPUS=$(($1))
NUM_CPUS=$((22*$NUM_GPUS))

BATCH_SIZE=$((8*$NUM_GPUS))

#############
arrVar="5"
# arrVar="MIG-8e7cd1d8-3e6e-59de-9448-20cd0a47ddc2,MIG-935d9f7f-8114-5be3-85e9-1002223dc30e"
arrVar2="device="$arrVar
arrVar3="\""$arrVar2"\""
###



# # ## Versione per usare le MIG gpu virtuali
# docker run \
#     --interactive --tty \
#     --rm \
#     --user $(id -u):$(id -g) \
#     --cpus $NUM_CPUS \
#     --runtime nvidia \
#     -e NVIDIA_VISIBLE_DEVICE=$arrVar3 \
#     --memory="200g" \
#     --volume $PWD:$PWD \
#     --workdir $PWD \
#     gianlucacarloni/cau_med:v1 $NUM_GPUS $arrVar \
#                                 --dataset_dir './' \
#                                 --backbone resnet50 \
#                                 --batch-size 2 \
#                                 --print-freq 700 \
#                                 --seed 42 \
#                                 --epochs 40 \
#                                 --lr 3e-5 \
#                                 --weight-decay 1e-3 \
#                                 --optim AdamW \
#                                 --num_class 15 \
#                                 --img_size 480 \
#                                 --dropoutrate_randomaddition 0.65 \

## VERSIONE FUNZIONANTE con DDP multi GPU, AL 18 OTTOBRE 2023 
docker run \
    --interactive --tty \
    --rm \
    --user $(id -u):$(id -g) \
    --cpus $NUM_CPUS \
    --gpus $arrVar3 \
    --memory="350g" \
    --volume $PWD:$PWD \
    --workdir $PWD \
    gianlucacarloni/cau_med:v1 $NUM_GPUS $arrVar \
                                --dataset_dir './' \
                                --backbone resnet50 \
                                --optim Adam_twd \
                                --num_class 15 \
                                --img_size 384 \
                                --batch-size $BATCH_SIZE \
                                --subset_take1everyN 3 \
                                --print-freq 200 \
                                --seed 42 \
                                --epochs 40 \
                                --weight-decay 1e-6 \
                                --lr 1e-5 \
                                --dropoutrate_randomaddition 0.5 \



    # gianlucacarloni/cau_med:v1  \ FUNZIONANTE AL 12 OTTOBRE CON LA VERSIONE SINGOLA GPU, DI GITHUB ONLINE.
    #                             --dataset_dir './' \
    #                             --backbone resnet50 \
    #                             --batch-size 48 \
    #                             --print-freq 100 \
    #                             --world-size 1 \
    #                             --rank 0 \
    #                             --epochs 40 \
    #                             --lr 1e-4 \
    #                             --optim AdamW \
    #                             --num_class 15 \
    #                             --img_size 128 \
    #                             --weight-decay 1e-2 \
    #                             --cutout \
    #                             --n_holes 1 \
    #                             --cut_fact 0.5 \
# #$@
# --user $(id -u):$(id -g) \
    #                             --output "./out" \

# gianlucacarloni/cau_med:v1
# gianlucacarloni/cau_med:v1 $NUM_GPUS $arrVar \
# gianlucacarloni/cau_med:v1  $NUM_GPUS $arrVar  \
