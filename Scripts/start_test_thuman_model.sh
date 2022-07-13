#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -u Source/main.py \
                        --mode test \
                        --batchSize 1 \
                        --gpu 1 \
                        --trainListPath ./Datasets/thuman_testing_list.csv \
                        --imgWidth 512 \
                        --imgHeight 512 \
                        --dataloaderNum 0 \
                        --maxEpochs 45 \
                        --imgNum 60 \
                        --sampleNum 1 \
                        --lr 0.001 \
                        --log ./TestLog/ \
                        --dist False \
                        --modelName BodyReconstruction \
                        --outputDir ./DebugResult/ \
                        --resultImgDir ./OutputResult/\
                        --modelDir ./Checkpoint/ \
                        --dataset thuman2.0 \
                        --save_mesh True
echo "Finish!"