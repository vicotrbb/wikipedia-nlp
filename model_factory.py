import math
import numpy as np
import string
import json
import os
import sys
import logging
import boto3

from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

import nltk
nltk.download('punkt')
nltk.download('stopwords')

chars = 0
maxlen = 60

def setupLogger():
	logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def downloadDataset():
    s3 = boto3.client('s3')
    bucket = os.getenv('S3_DATASET_BUCKET')
    file = 'wikipedia-content-dataset.json'
    s3.download_file(bucket, file, file)
    logging.info(f'dataset downloaded')


def prepareData(dataFile):
    f = open(dataFile,)
    data = json.load(f)

    content = list(data[x] for x in data.keys())
    text = ''

    for c in content:
        for i in c:
            text += i

    logging.info(f'Corpus length:', len(text))

    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    text = ''
    for c in words:
        text += c
        text += ' '
    text = text.strip()
    logging.info(f'Finished to load file')
    return text


def prepareTrainingData(text):
	step = 3
	sentences = []
	next_chars = []

	for i in range(0, len(text) - maxlen, step):
	    sentences.append(text[i: i + maxlen])
	    next_chars.append(text[i + maxlen])
	logging.info(f'Numero de sequencias:', len(sentences))

	chars = sorted(list(set(text)))
	logging.info(f'Caracteres unicos:', len(chars))
	char_indices = dict((char, chars.index(char)) for char in chars)

	logging.info(f'Vetorizando o texto')
	x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
	    for t, char in enumerate(sentence):
	        x[i, t, char_indices[char]] = 1
	    y[i, char_indices[next_chars[i]]] = 1
	logging.info(f'Finished to prepare data')
	return x, y


def prepareTrainModel(x, y):
	model = Sequential([
        LSTM(len(chars), return_sequences=True, input_shape=(maxlen, len(chars))),
        LSTM(len(chars), return_sequences=True),
        LSTM(len(chars)),
        Dense(len(chars), activation='relu'),
        Dropout(0.2),
        Dense(len(chars), activation='softmax')
	])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	for epoch in range(1, 100):
		logging.info(f'Epoch {epoch}')
		model.fit(x, y, batch_size=128, epochs=1)

	logging.info(f'Finished to train model')
	return model


def saveModel(model):
    s3 = boto3.client('s3')
    file = 'wikipedia-nlp.hdf5'
    gen = os.getenv('GENERATION')
    bucket = os.getenv('S3_BUCKET')

    model.save(file)

    s3.upload_file(file, bucket, gen+'/'+file)
    return 0


setupLogger()
downloadDataset()
text = prepareData('wikipedia-content-dataset.json')
x, y = prepareTrainingData(text)
model = prepareTrainModel(x, y)
saveModel(model)
logging.info(f'Model training finished and file saved to s3.')