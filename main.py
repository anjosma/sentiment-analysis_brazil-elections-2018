from dataset import *
from keras.models import load_model
from data import clean_text, max_length, encode_text, create_tokenizer, remove_duplicates, under_sampling
from models import *
import numpy as np
import os
from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score,recall_score
np.random.seed(42)

def transform_text(texts):
	texts, _ = clean_text(texts, labels=None)
	tokenizer = create_tokenizer(texts, load=True)
	tweets = encode_text(tokenizer, texts, 26)
	return tweets

def train():
	
	for dataset, dataset_name in zip(datasets, datasets_names):
		j = 0
		texts, labels = dataset()
		texts, labels = clean_text(texts, labels)
		print('TAMANHO DO DATASET BRUTO:', len(labels))
		texts, labels = remove_duplicates(texts, labels)
		print('TAMANHO DO DATASET REMOVENDO REPETIÇÕES:', len(labels))
		texts, labels = under_sampling(texts, labels)
		print('TAMANHO DO DATASET FINAL:', len(texts))
		tokenizer = create_tokenizer(texts)
		length = max_length(texts)
		vocab_size = len(tokenizer.word_index) + 1
		tweets = encode_text(tokenizer, texts, length)

		for model_, model_name in zip(models, models_names):
			print('\nTAMANHO:', length)
			print('TAMANHO DO VOCABULARIO:', vocab_size)
			
			k_fold = 0
			sss = StratifiedShuffleSplit(n_splits=3, random_state=42, test_size=0.2)
			labels = np.array([int(i) for i in labels])
			
			for train_index, test_index in sss.split(tweets, labels):
				x_train, x_test = tweets[train_index], tweets[test_index]
				y_train, y_test = labels[train_index], labels[test_index]

				path = './models/'+dataset_name+'/'
				if not os.path.exists(path): os.makedirs(path)

				model = model_(length, vocab_size)
				check = ModelCheckpoint(path+dataset_name+model_name+str(k_fold)+'_model.h5', monitor='val_loss',
					save_best_only=True)
				stop = EarlyStopping(monitor='val_loss', patience=5)
				plot_model(model, to_file=path+model_name+'model.png', show_shapes=True)
				print(model.summary())
				model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

				class_weight_list = compute_class_weight('balanced', np.unique(y_train), y_train)
				class_weight = dict(zip(np.unique(y_train), class_weight_list))
				print(class_weight)
				callbacks = [check, stop]

				try:
					h = model.fit([x_train,x_train,x_train],
						y_train, epochs=epochs, batch_size=batch_size,
						validation_data=([x_test, x_test, x_test],
						y_test), callbacks=callbacks,
						class_weight=class_weight, verbose=1)
				except:
					h = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
						validation_data=(x_test, y_test), callbacks=callbacks, class_weight=class_weight,
						verbose=1)

				del model
				model = load_model(path+dataset_name+model_name+str(k_fold)+'_model.h5')
				
				try:
					y_pred = model.predict(x_test)
				except:
					y_pred = model.predict([x_test, x_test, x_test])
				
				for threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
					y_pred_ = [1 if y > threshold else 0 for y in y_pred]
					print('thresh', threshold)
					print(accuracy_score(y_test, y_pred_))
					
					
					log_path = path + 'log.csv'
					to_save = {
						'arc': model_name, 
						'fold': k_fold,
						'acc': accuracy_score(y_test, y_pred_),
						'prec': precision_score(y_test, y_pred_),
						'rec': recall_score(y_test, y_pred_),
						'f1': f1_score(y_test, y_pred_),
						'dataset': dataset_name,
						'thresh': threshold
					}

					df = pd.DataFrame([to_save])

					if k_fold == 0 and j == 0:
						with open(log_path, 'w') as f: df.to_csv(f, header=True)
					else:
						with open(log_path, 'a') as f: df.to_csv(f, header=False)
					j = j + 1
					df_ = pd.read_csv(log_path, index_col=[0])
					print(model_name)
					print(df_[df_['arc'] == model_name].acc.mean())
					print(df_[df_['arc'] == model_name].acc.std())

				k_fold = k_fold + 1

if __name__ == '__main__':
	epochs = 1000
	batch_size = 8

	datasets = [political_data]
	datasets_names = ['political_data']

	models = [cnn_blstm,multiple]
	models_names = ['cnn_blstm', 'multiple']

	train()
