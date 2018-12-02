from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Sequential
import numpy as np
np.random.seed(42)

emb_size = 8
filters = 3
lstm_number = 10
def multiple(length, vocab_size):
	inputs1 = Input(shape=(length,))
	embedding1 = Embedding(vocab_size, emb_size)(inputs1)
	dropout1_0 = Dropout(0.5)(embedding1)
	conv1 = Conv1D(filters=filters, kernel_size=2, activation='relu', padding='same')(dropout1_0)
	pool1 = MaxPooling1D(pool_size=2)(conv1)
	dropout1_1 = Dropout(0.5)(pool1)
	b_lstm1 = Bidirectional(LSTM(lstm_number, return_sequences=True, recurrent_dropout=0.5, dropout=0.5))(dropout1_1)
	lstm1 = Bidirectional(LSTM(lstm_number, recurrent_dropout=0.5, dropout=0.5))(b_lstm1)

	inputs2 = Input(shape=(length,))
	embedding2 = Embedding(vocab_size, emb_size)(inputs2)
	dropout2_0 = Dropout(0.5)(embedding2)
	conv2 = Conv1D(filters=filters, kernel_size=3, activation='relu', padding='same')(dropout2_0)
	pool2 = MaxPooling1D(pool_size=2)(conv2)
	dropout2_1 = Dropout(0.5)(pool2)
	b_lstm2 = Bidirectional(LSTM(lstm_number, return_sequences=True, recurrent_dropout=0.5, dropout=0.5))(dropout2_1)
	lstm2 = Bidirectional(LSTM(lstm_number, recurrent_dropout=0.5, dropout=0.5))(b_lstm2)

	inputs3 = Input(shape=(length,))
	embedding3 = Embedding(vocab_size, emb_size)(inputs3)
	dropout3_0 = Dropout(0.5)(embedding3)
	conv3 = Conv1D(filters=filters, kernel_size=4, activation='relu', padding='same')(dropout3_0)
	pool3 = MaxPooling1D(pool_size=2)(conv3)
	dropout3_1 = Dropout(0.5)(pool3)
	b_lstm3 = Bidirectional(LSTM(lstm_number, return_sequences=True, recurrent_dropout=0.5, dropout=0.5))(dropout3_1)
	lstm3 = Bidirectional(LSTM(lstm_number, recurrent_dropout=0.5, dropout=0.5))(b_lstm3)

	merged = concatenate([lstm1, lstm2, lstm3])

	dropout = Dropout(0.5)(merged)
	outputs = Dense(1, activation='sigmoid')(dropout)
	model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

	return model

def cnn_blstm(length, vocab_size):

	model = Sequential()
	model.add(Embedding(vocab_size, emb_size, input_length=length))
	model.add(Dropout(0.5))
	model.add(Conv1D(filters, 3, padding='same', activation='relu'))
	model.add(MaxPooling1D())
	model.add(Dropout(0.5))
	model.add(Bidirectional(LSTM(lstm_number, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
	model.add(Bidirectional(LSTM(lstm_number, dropout=0.5, recurrent_dropout=0.5)))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	return model