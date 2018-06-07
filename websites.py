all_urls = open('all_guardian_urls.txt').readlines()+open('all_nyt_urls.txt').readlines()
all_bodies = open('all_guardian_bodies.txt').readlines()+open('all_nyt_bodies.txt').readlines()

with open('false_neg.txt') as file:
	real = file.readlines()


from fuzzywuzzy import fuzz
real_websites=[]

for i, realy in enumerate(real):
	best_score=0
	for ind, body in enumerate(all_bodies):
		new_score = fuzz.token_set_ratio(body, realy)
		if new_score>best_score:
			best_score = new_score
			best_item = body
			best_url = all_urls[ind]
		real_websites.append(best_url)

print(real_websites)

