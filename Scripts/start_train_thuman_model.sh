#!/bin/bash
# parameters
tensorboard_port=6006
dist_port=8800
tensorboard_folder='./log/'
echo "The tensorboard_port:" ${tensorboard_port}
echo "The dist_port:" ${dist_port}

# command
# delete the previous tensorboard files
if [ -d "${tensorboard_folder}" ]; then
    rm -r ${tensorboard_folder}
fi

echo "Begin to train the model!"
CUDA_VISIBLE_DEVICES=0,1 python -u Source/main.py \
                        --mode train \
                        --batchSize 8 \
                        --gpu 2 \
                        --trainListPath ./Datasets/thuman_training_list_3_15.csv \
                        --imgWidth 512 \
                        --imgHeight 512 \
                        --dataloaderNum 0 \
                        --maxEpochs 150 \
                        --imgNum 1950 \
                        --sampleNum 1 \
                        --log ${tensorboard_folder} \
                        --lr 0.001 \
                        --dist false \
                        --modelName BodyReconstruction \
                        --modelDir ./Checkpoint_4_15_4/ \
                        --port ${dist_port} \
                        --dataset thuman2.0 > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

#echo "Begin to watch TrainRun.log file!"
#tail -f TrainRun.log