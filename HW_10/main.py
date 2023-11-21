import numpy as np

from keras.layers import Dense, LSTM, Input, Embedding
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

with open('train data/train_data_eng', 'r', encoding='utf-8') as f:
  texts_eng = f.readlines()
  texts_eng[0] = texts_eng[0].replace('\ufeff', '')

with open('train data/train_data_ukr', 'r', encoding='utf-8') as f:
  texts_ukr = f.readlines()
  texts_ukr[0] = texts_ukr[0].replace('\ufeff', '')

print("text eng :: ", texts_eng)
print("text ukr :: ", texts_ukr)

texts = texts_eng + texts_ukr
count_eng = len(texts_eng)
count_ukr = len(texts_ukr)
total_lines = count_eng + count_ukr
print(count_eng, count_ukr, total_lines)

maxWordsCount = 1000
tokenizer = Tokenizer(num_words = maxWordsCount, lower = True, split = ' ', char_level = False)
tokenizer.fit_on_texts(texts)

dist = list(tokenizer.word_counts.items())

max_text_len = 10
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen = max_text_len)

X = data_pad
Y = np.array([[1, 0]] * count_eng + [[0, 1]] * count_ukr)

indeces = np.random.choice(X.shape[0], size = X.shape[0], replace = False)
X = X[indeces]
Y = Y[indeces]

model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))
model.add(LSTM(128, return_sequences = True))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = Adam(learning_rate = 0.001))
history = model.fit(X, Y, batch_size = 32, epochs = 50)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

print("Enter text to test neural network:")
t = input()
data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen = max_text_len)

res = model.predict(data_pad)
if (np.argmax(res) == 0):
  print("Text is most likely in English.")
else:
  print("Text is most likely in Ukrainian.")