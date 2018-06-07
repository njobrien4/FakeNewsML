cur_dir = "no_word_enchant_no_end/"

import interpret
most_fake, most_real, least_fake, least_real =[], [], [], []

import pickle

with open(cur_dir+"best_trigrams_pickle.txt",'rb') as f2:
	best_trigrams = pickle.load(f2)

with open(cur_dir+"all_top_n_neurons.txt", 'rb') as f:
	all_top_neurons = pickle.load(f)

with open(cur_dir+"best_trigrams_pickle.txt",'rb') as f2:
	best_trigrams = pickle.load(f2)

import numpy as np

all_wi_ai = np.load(cur_dir+"all_wi_ai.npy")

for i in range(5504):
	mf, mr, lf, lr = interpret.get_info2(i, all_wi_ai, all_top_neurons, best_trigrams)
	most_real.extend(mr)
	most_fake.extend(mf)
	least_real.extend(lr)
	least_fake.extend(lf)

with open(cur_dir+'most_real.txt', 'w') as mrt:
	mrt.write(' '.join(most_real))
with open(cur_dir+'least_real.txt', 'w') as lrt:
	lrt.write(' '.join(least_real))
with open(cur_dir+'most_fake.txt', 'w') as mft:
	mft.write(' '.join(most_fake))
with open(cur_dir+'least_fake.txt','w') as lft:
	lft.write(' '.join(least_fake))

#print(most_real)
