'''
TextGenerator - character by character. 
Notes:- 
	The two text samples (small and large) provided right at the bottom, correspond to the TRIALs having 818 characters and 1535 characters respectively. 
	All other TRIALs are run on the first article in the MediumArticles.csv dataset.
	To change total_chars you can change the article selected from the dataset or use any other text data and feed it to the variable 'raw_text'.
TRIAL ONE:  	total chars = 818  &&  epoch_num = 20  &&  seq_len = 40 -
.						best LOSS = 2.9146
            
TRIAL TWO:  	total chars = 1535  &&  epoch_num = 50  &&  seq_len = 40 -
.					  	best LOSS = 2.4977
            
TRIAL THREE: 	total chars = 818  &&  epoch_num = 50  &&  seq_len = 40 -
.					    best LOSS = 2.7519
              
TRIAL FOUR: 	total chars = 1535  &&  epoch_num = 20  &&  seq_len = 40 -
.					    best LOSS = 2.8652
              
TRIAL FIVE: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 40 -
.					    best LOSS = 2.6772
              
TRIAL SIX: 		total chars = 11932  &&  epoch_num = 50  &&  seq_len = 40 - 
.   					best LOSS = 0.9772!!!
              
TRIAL SEVEN: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 16 - 
. 		  				best LOSS = 2.6503
TRIAL EIGHT: 	total chars = 11932  &&  epoch_num = 20  &&  seq_len = 100 -
.					    best LOSS = 2.6431
'''

import sys
import numpy as np
import pandas as pd
import string
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def main():
	global model
	modes = ['exp', 'train', 'generate']												#mode is either 'exp', 'train' or 'generate'
	mode = modes[0]
	
	#Variable parameters
	#num = the number of the article from the dataset ~ Size of the learning data | epoch_num = number of times the algorithm trains over the learning data | seq_len = window size of characters.
	num = 1
	epoch_num = 20
	seq_len = 100

	
	#MAIN CODE
	#load file data
	path = "MediumArticles.csv"
	df = pd.read_csv(path)
	if mode == 'exp':
		print("\nThe Journey of the Data!\n\n1. Data loaded from dataset-\n", df.head(3))
	
	#Extract the article from the df, remove paragraphing and punctuation
	data = get_text(num, df, mode)
	raw_text = prepare(data)
	if mode == 'exp':
		print("\n4. Final Cleaned Data-\n", raw_text[:100])

	#create a mapping from char to int and reverse
	chars = sorted(list(set(raw_text)))														#Set creates a set of unique chars and sorted sorts the list in ascending order
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	#Summarize the data
	n_chars = len(raw_text)																	#No. of characters in the text
	n_vocab = len(chars)																	#No. of unique characters in the text
	print("\nTotal chars = ", n_chars)
	print("Total vocab = ", n_vocab)
	#prepare input and output pair sequences
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_len):
		seq_in = raw_text[i:i + seq_len]
		seq_out = raw_text[i + seq_len]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append([char_to_int[char] for char in seq_out])
	n_patterns = len(dataX)
	print("Total patterns = ", n_patterns)
	
	if mode == 'exp':
		print("Last input sequence is - ", seq_in, end="\n\n")
		print("dataX (input mapped to ints). The integer mapped values of the 100 (seq_len) characters in the first 22 input sequences:\n", dataX[:22], end="\n\n")
		print("dataY (output mapped to ints). Integer mapped values of the expected output to each of the 22 inpupt sequences:\n", dataY[:22], end="\n\n")

	#Modify the data
	#Reshape
	X = np.reshape(dataX, (n_patterns, seq_len, 1))
	#Normalize
	X = X / float(n_vocab)
	#One hot encode the output
	y = np_utils.to_categorical(dataY)

	if mode == 'exp':
		print("After dividing dataX[[]] by n_vocab(44):\n", X[:22], end="\n\n")
		print("X shape is: ", X.shape, end="\n\n")

	#Create Model - 
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.1))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	print(model.summary())

	if mode is 'train': 
		#Define checkpoint to create a weight-file after every epoch
		filepath = "weights-improvement-9-{epoch:02d}-{loss:.4f}.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		#fit the model with callbacks to the checkpoint
		model.fit(X, y, epochs=epoch_num, batch_size=64, callbacks=callbacks_list)
		print("\n\t\t~Fin~\n")
	elif mode is 'generate':
		best_file = "weights-improvement-9-20-2.6431.hdf5"
		model.load_weights(best_file)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		#Pick random sequence from the input-sequences as a seed value to act as input for the prediction
		start = np.random.randint(0, len(dataX)-1)
		pattern = dataX[start]
		print("Seed: ", end="")
		#Input-sequences are stored as ints - convert them to chars
		print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
		#generate the characters, arbitrarily chosen 900 characters to generate, can be any number
		for i in range(900):
			#preprocess steps as done before training as well - reshape and divide by len(vocab)
			x = np.reshape(pattern,(1, len(pattern), 1))
			x = x / float(n_vocab)
			prediction = model.predict(x, verbose=0)
			index = np.argmax(prediction)													#Finds the value with 1 in one hot encoding
			result = int_to_char[index]
			seq_in = [int_to_char[value] for value in pattern]
			sys.stdout.write(result)														#print(result, end="")
			pattern.append(index)															#Add generated character to the input sequence
			pattern = pattern[1:len(pattern)]												#Maintain constant length of input sequence
		print("\n\t\t~Fin~\n")
	elif mode is 'exp':
		print("\n\t\t~Fin~\n")
	else:
		print("[ModeError] Select mode from: [generate / train / exp], all case-sensitive.")
		print("\n\t\t~Fin~\n")

def prepare(txt):
	#Change to lowercase and remove punctuations.
	txt = txt.lower()
	punc = string.punctuation.translate({ord(c): None for c in "!.?"})						#Remove terminating symbols and create a string of all other punctuations
	txt = txt.translate({ord(c): None for c in punc})										#Remove these punctuations from the text
	return txt 																				#(ord() returns the UTF code for the character)

def get_text(num, df, mode):
	#num is the number of the article to extract from the dataset. Example done on the first article.
	for i in range(num - 1, num):
		data = df.text[i]
		if mode == 'exp':
			print("\n2. Raw data to train on-\n", data[:100], "...")
		data = " ".join([line.strip() for line in data.split("\n")])						#replace newline characters with a space ("\n"  ->  " ")
		if mode == 'exp':
			print("\n3. Raw data after removing newline characters-\n", data[:100], "...")
	return data

if __name__ == "__main__":
	main()

'''
