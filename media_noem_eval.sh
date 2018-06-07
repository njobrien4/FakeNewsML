#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1
#SBATCH --job-name=augment
#SBATCH --mem=16GB
#SBATCH --gres=gpu:0
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm

singularity exec -B /om:/om --nv /om/user/njobrien/containers/localtensorflow.img python "/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/eval.py" --checkpoint_dir="/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/log/runs/1523582598/checkpoints/" --eval_train --positive_data_file="/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/media_rb_noem.txt" --negative_data_file="/om/user/njobrien/FakeNews/cnn-text-classification-tf (body)/data/news-data/media_fb_noem.txt" --trigram_dir="media_noem/"
