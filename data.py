import string
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import re
import unidecode
import pickle
np.random.seed(42)

def save_tokenizer(tokenizer):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

def load_stopwords():
    with open('stopwords.txt', 'r') as f:
        words = [i.replace('\n', '') for i in f]
    return words

def remove_duplicates(x, y):
    df = pd.DataFrame({"x":x, "y":y})
    df = df.drop_duplicates(subset=['x'], keep=False)
    texts = df['x'].tolist()
    labels = df['y'].tolist()
    print(len([labels for i in labels if i == 1]))
    return texts, labels

def under_sampling(x, y):
	x = pd.DataFrame(x)
	x.insert(loc=0, column='target', value=y)
	count_class_0, count_class_1 = x.target.value_counts()
	df_class_0 = x[x['target'] == 0]
	df_class_1 = x[x['target'] == 1]
	df_class_0_under = df_class_0.sample(count_class_1)
	x_under = pd.concat([df_class_0_under, df_class_1], axis=0)
	y = x_under.target.tolist()
	x = x_under.drop(['target'], 1)
	x = x[0].tolist()
	return x, y

def clean_text(texts, labels):
    new_text = []
    new_labels = []
    i = 0
    for text in texts:
        text = text.lower()
        text = unidecode.unidecode(text)
        text = ''.join(re.sub("(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text))
        text = ''.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text))
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        tokens = text.split()
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [word for word in tokens if word.isalpha()]
        stop_words = stopwords.words('portuguese')+load_stopwords()
        tokens = [w for w in tokens if not w in stop_words]
        tokens = [word for word in tokens if len(word) > 1]
        tokens = ' '.join(tokens)
        new_text.append(tokens)
        if not labels == None:
            new_labels.append(labels[i])
        i = i + 1
    return new_text, new_labels

def create_tokenizer(lines, load=None):
    if load: 
        tokenizer = load_tokenizer()
    else:
        tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(lines)
    if not load: save_tokenizer(tokenizer)
    return tokenizer

def max_length(lines):
	return max([len(s.split()) for s in lines])

def encode_text(tokenizer, lines, length):
	encoded = tokenizer.texts_to_sequences(lines)
	padded = pad_sequences(encoded, 
                            padding='post',
                            maxlen=26
                            )
	return padded

