import json
import pandas as pd
from random import shuffle
from main import transform_text
from keras.models import load_model

alckmin = ['alckmin', 'geraldo alckmin', 'psdb', 'partido da social democracia brasileira']
amoedo = ['amoedo', 'amoêdo', 'joão amoedo', 'joao amoêdo', 'joão amoêdo', 'partido novo']
marina = ['marina', 'marina silva', 'partido verde']
ciro = ['ciro', 'ciro gomes', 'cirão', 'cirão da massa', 'pdt', 'partido democrático trabalhista']
boulos = ['boulos', 'guilherme boulos', 'psol', 'partido socialismo e liberdade']
meirelles = ['henrique meireles', 'henrique meirelles', 'meirelles', 'meireles', 'mdb', 'movimento democrático brasileiro']
bolsonaro = ['jair bolsonaro', 'bolsonaro', 'bolsomito', 'psl', 'partido social liberal','bozo', 'bozonaro', 'bonossauro']
davila = ["manuela d'ávila", "d'ávila", 'davila', 'dávila', 'manuela davila', 'manuela' ,'manuela dávila', 'pcdob', 'partido comunista do brasil']
dias = ['alvaro dias', 'álvaro dias', 'partido podemos']
daciolo = ['cabo daciolo', 'daciolo', 'partido patriota']
lula = ['lula', 'luiz inacio', 'luiz inácio']
haddad = ['haddad','hadad', 'fernando haddad', 'fernando hadad', 'pt']

json_files = ['./debates/monitor2018_08_09.json', './debates/monitor2018_08_17.json',
'./debates/monitor2018_09_26.json', './debates/monitor2018_09_30.json',
'./debates/monitor2018_10_04.json', './debates/monitor2018_10_28.json']
	
debates = ['band', 'redetv', 'sbt', 'record', 'globo', 'apuracoes']

model = load_model('./models/political_data/political_datamultiple1_model.h5')

candidatos = [alckmin, amoedo, marina, ciro, boulos, meirelles, bolsonaro, dias,
	daciolo, lula, haddad]

def parse():
	for debate_file, debate in zip(json_files, debates):
		texts = []
		createds = []
		is_retweet = []
		is_verifieds = []
		source_list = []
		place_list = []
		location_list = []
		sentiment_list = []
		users_list = []

		with open(debate_file, 'r') as f:
			i = 0
			for line in f:
				tweet = json.loads(line)

				try:
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
					text = text.replace('\n', ' ')
					texts.append(text)
				except:
					pass

				try:
					try:
						created = tweet["created_at"]
					except:
						created = None

					try:	
						retweeted = tweet["retweeted"]
					except:
						retweeted = None

					try:
						verified = tweet["user"]["verified"]
					except:
						verified = None

					try:
						source = tweet["source"]
					except:
						source = None

					try:
						try:
							place = tweet["place"]["full_name"]
						except:
							place = tweet["place"]
					except:
						place = None

					try:
						location = tweet["user"]["location"]
					except:
						location = None

					try:
						user = tweet["user"]["screen_name"]
					except:
						user = None

					location_list.append(location)
					place_list.append(place)
					source_list.append(source)
					is_retweet.append(retweeted)
					createds.append(created)
					is_verifieds.append(verified)
					users_list.append(user)
				except:
					pass

				print(i, debate, 'loading')
				
				i = i + 1

		texts = [text.lower() for text in texts]

		new_texts = []
		candidato_in = []
		new_location = []
		new_place = []
		new_source = []
		new_rt = []
		new_createds = []
		new_verified = []
		new_users = []

		i = 0
		for text in texts:
			for candidato in candidatos:
				others = []
				for cand in candidatos:
					for nam in cand:
						others.append(nam)
				others = [name for name in others if name not in candidato]
				for name in candidato:
					if name in text:
						if not any(nome in text for nome in others):
							if text not in new_texts:
								new_texts.append(text)
								candidato_in.append(candidato[0])
								new_location.append(location_list[i])
								new_place.append(place_list[i])
								new_source.append(source_list[i])
								new_rt.append(is_retweet[i])
								new_createds.append(createds[i])
								new_verified.append(is_verifieds[i])
								new_users.append(users_list[i])
			print(i, debate, 'parsing')
			i = i + 1

		sentiment_list = []
		print(debate, 'predicting')
		transformed_text = transform_text(new_texts)
		y = model.predict([transformed_text, transformed_text, transformed_text])
		y = [1 if i > 0.55 else 0 for i in y]

		df = pd.DataFrame({'text': new_texts, 'candidato': candidato_in, 'sentiment': y,
		'location': new_location, 'place': new_place,
		'source': new_source, 'rt': new_rt,
		'created': new_createds, 'verified': new_verified, 'user': new_users})
		df.to_csv('candidatos/sentimento_candidato_'+debate+'.csv')

parse()