
import pandas as pd
data = pd.read_csv('Senti.csv')
# Keeping only the neccessary columns
data = data[['text','sentiment']]

data = data[data.sentiment != "Neutral"]
data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

for idx,row in data.iterrows():
    row[0] = row[0].replace('rt',' ')
    
max_fatures = 2000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

Y = pd.get_dummies(data['sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print('Shape of training samples:',X_train.shape,Y_train.shape)
print('Shape of testing samples:',X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(max_fatures, 128 ,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(128))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

batch_size = 32
model.fit(X_train, Y_train, epochs = 5, batch_size=batch_size, verbose = 2)

score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("Score: %.2f" % (score))
print("Accuracy: %.2f" % (acc))

text = 'We are going to Delhi'
tester = np.array([text])
tester = pd.DataFrame(tester)
tester.columns = ['text']

tester['text'] = tester['text'].apply(lambda x: x.lower())
tester['text'] = tester['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

max_fatures = 2000
test = tokenizer.texts_to_sequences(tester['text'].values)
test = pad_sequences(test)

if X.shape[1]>test.shape[1]:
    test = np.pad(test[0], (X.shape[1]-test.shape[1],0), 'constant')
    
test = np.array([test])

prediction = model.predict(test)
print('Prediction value:',prediction[0])

model = Sequential()
model.add(Embedding(max_features, 8))
model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


#Influence of Embedding
model = Sequential()
model.add(Embedding(max_features, 4))
model.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)


#Influence of Dropout
#Dropout with probability 

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(8, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

#rnn with 2 layers

model = Sequential()
model.add(Embedding(max_features, 8))
model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))
model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0, return_sequences=True))
model.add(LSTM(8, dropout=0.0, recurrent_dropout=0.0))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
