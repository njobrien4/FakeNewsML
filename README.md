#To run train/evaluate:
  (this should all be done in openmind container with tensorflow)
  a) download and unzip GoogleNews-vectors-negative300.bin.gz ( I got from: https://github.com/mmihaltz/word2vec-GoogleNews-vectors)
  b) change directories in sh file (i.e.: launch_no_word_end.sh) to match the full path of train.py, positive data, and negative data files
  c) add modules and make log directory: 
module add openmind/cuda/8.0
module add openmind/cudnn/8.0-5.1
module load openmind/singularity/2.5-dev
mkdir log
  d) run "sbatch launch_no_word_end.sh" (Or whichever model you want to train)
  e) view the log to see where the model was saved (should be a number such as "1528206991")
  f) In the sh eval file (i.e. vi no_word_no_end_eval.sh) change directory to runs/YOUR_NUMBER/checkpoints/ (i.e. runs/1528206991/checkpoints/)
  g) In the sh eval file (i.e. vi no_word_no_end_eval.sh) you can change trigram_dir to whatever to you want
  h) make a directory inside of log that is named whatever you change the value of trigram_dir to be
  i) run "sbatch no_word_no_end_eval.sh" or whatever your eval sh file is
  
#To find all the websites (fake and real) that were misclassified (you can get section/type of fake news from website list):
  a) put false_pos.txt and false_neg.txt in same directory as websites.py
  b) run "python websites.py" and "python websites_fake.py"
 
 
#(A) To get the fake and real trigrams and then (B) remove those words included in both and then separate by part of speech
 
 to do (A)
  a) change "cur_dir = " in real_fake_pos.py to be whatever the trigram directory was that you used (i.e. "log/no_word_no_end/")
  b) run " python real_fake_pos.py"
  
 to do (B)
  a) run "python get_unique_true_false.py" (as log as most_fake.txt, etc are in current directory)
  
 
#Weights histogram is made by calling:
python
import interpret
interpret.make_weight_histogram(weights)

#Other info from 
interpret.get_info(3, "all_wi_ai.npy", "all_top_n_neurons.txt", "best_trigrams_pickle.txt") 
