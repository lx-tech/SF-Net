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
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u Source/main.py \
                        --mode train \
                        --batchSize 16 \
                        --gpu 2 \
                        --trainListPath ./Datasets/thuman_training_list.csv \
                        --imgWidth 512 \
                        --imgHeight 512 \
                        --dataloaderNum 0 \
                        --maxEpochs 200 \
                        --imgNum 2842 \
                        --sampleNum 1 \
                        --log ${tensorboard_folder} \
                        --lr 0.001 \
                        --dist false \
                        --modelName BodyReconstruction \
                        --port ${dist_port} \
                        --dataset thuman2.0 > TrainRun.log 2>&1 &
echo "You can use the command (>> tail -f TrainRun.log) to watch the training process!"

echo "Start the tensorboard at port:" ${tensorboard_port}
nohup tensorboard --logdir ${tensorboard_folder} --port ${tensorboard_port} \
                        --bind_all --load_fast=false > Tensorboard.log 2>&1 &
echo "All processes have started!"

#echo "Begin to watch TrainRun.log file!"
#tail -f TrainRun.log