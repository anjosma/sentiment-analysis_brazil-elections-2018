import pandas as pd
import json
import numpy as np
np.random.seed(42)

def sent_16():
	top = 800000
	df_pos = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='utf-8', header=None, usecols=[0,5]).head(top)
	df_neg = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='utf-8', header=None, usecols=[0,5]).tail(top)
	database = df_pos[5].tolist() + df_neg[5].tolist()
	labels = df_pos[0].tolist() + df_neg[0].tolist()
	for n, i in enumerate(labels):
		if i == 4:
			labels[n] = 1
	return database, labels

def sent_16_translate():
	df = pd.read_csv('translate.csv', header=None, usecols=[0,4])
	df_neg = df[df[4] == '0']
	df_pos = df[df[4] == '1']

	database = df_pos[0].tolist() + df_neg[0].tolist()
	labels = df_pos[4].tolist() + df_neg[4].tolist()
	labels = [int(i) for i in labels]
	return database, labels

def political_data():
	df = pd.read_csv('labeled_samples2.csv', index_col=[0])
	database = df['text'].tolist()
	labels = df['label'].tolist()
	return database, labels
	
def pos_neg():
	database = []
	labels = []
	files = ['negativos_stream_download.json', 'positivos_stream_download.json']
	for file in files:
		with open(file, 'r') as f:
			for line in f:
				try:
					tweet = json.loads(line)
					try: 
						try:
							x = ('from RT extended_full_text')
							text = tweet["retweeted_status"]["extended_tweet"]["full_text"]
						except:
							x = ('from RT extended_text')
							text = tweet["retweeted_status"]["text"]
					except:
						try:
							x = ('from extended_text')
							text = tweet["extended_tweet"]["full_text"]
						except:
							x = ('from text')
							text = tweet["text"]
					database.append(str(text))
					if 'negativo' in file:
						labels.append(0)
					else:
						labels.append(1)
				except:
					pass
	return database, labels