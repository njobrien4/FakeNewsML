#!/bin/bash
#SBATCH -n 1
#SBATCH --array=1
#SBATCH --job-name=augment
#SBATCH --mem=80GB
#SBATCH --gres=gpu:3
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

singularity exec -B /om:/om --nv /om/user/njobrien/containers/localtensorflow.img python "/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/train.py" --positive_data_file="/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/all_real_bodies_noem_no_trump.txt" --negative_data_file="/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/non_trump_fake_bodies.txt" --filter_sizes="3" --num_filters=128

