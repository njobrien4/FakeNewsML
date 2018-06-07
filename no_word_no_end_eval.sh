#!/bin/bash
#SBATCH -n 2
#SBATCH --array=1
#SBATCH --job-name=augment
#SBATCH --mem=16GB
#SBATCH --gres=gpu:0
#SBATCH -t 2:00:00
#SBATCH --workdir=./log/
#SBATCH --qos=cbmm


singularity exec -B /om:/om --nv /om/user/njobrien/containers/localtensorflow.img python "/om/user/njobrien/FakeNews/public/eval.py" --checkpoint_dir="/om/user/njobrien/FakeNews/public/log/runs/1528206991/checkpoints/" --eval_train --positive_data_file="/om/user/njobrien/FakeNews/public/data/news-data/no_words/erb_test_end.txt" --negative_data_file="/om/user/njobrien/FakeNews/public/data/news-data/no_words/efb_test.txt"  --trigram_dir="no_word_enchant_no_end/"
