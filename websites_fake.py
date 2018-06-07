from fuzzywuzzy import fuzz
import csv
with open('false_pos.txt') as file:
	fake = file.readlines()


all_fake_bodies = []
with open('fake_news_clean_full_body_new.csv') as file:
	reader = csv.reader(file,dialect='excel')
	for row in reader:
		all_fake_bodies.append(row)
fake_websites=[]
for i, fakey in enumerate(fake):
	best_score = 0
	for ind, body in enumerate(all_fake_bodies):
		new_score = fuzz.token_set_ratio(body[2], fakey)
		if new_score>best_score:
			best_score = new_score
			best_item = body[2]
			best_url = body[1]
	fake_websites.append(best_url)	

print(fake_websites)
