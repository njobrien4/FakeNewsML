#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1
#SBATCH --job-name=augment
#SBATCH --mem=100GB
#SBATCH --gres=gpu:2
#SBATCH -t 6:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

singularity exec -B /om:/om --nv /om/user/njobrien/containers/localtensorflow.img python "/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/train.py" --filter_sizes="3" --num_filters=128 --positive_data_file="/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/no_media_rb_noem.txt" --negative_data_file="/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/no_media_fb_noem.txt"

