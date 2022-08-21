import librosa as lb
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard,EarlyStopping
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import os
import pickle

r'''.\venv\Scripts\activate'''
r'''tensorboard --logdir=.'''

dataset_path = "TESS Toronto emotional speech set data"
voices = []
labels = []
for dir in os.listdir(dataset_path):
    for voice in os.listdir(os.path.join(dataset_path, dir)):
        sound_path = os.path.join(dataset_path, dir, voice)
        voices.append(sound_path)
        labels.append(' '.join(dir.split('_')[1:]).strip())

df = pd.DataFrame()
df['speech'] = voices
df['label'] = labels

'''DATA EXPLORATION'''


def plot_wav(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion)
    lb.display.waveshow(data, sr=sr)
    plt.show()


def plot_sectrogram(data, emotion):
    x = lb.stft(data)
    xdb = lb.amplitude_to_db(abs(x))
    plt.figure(figsize=(10, 4))
    plt.title(emotion)
    lb.display.specshow(xdb, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()


# emotion = 'Fear'
# path = np.asarray(df['speech'][df['label'] == emotion])[0]
#
# data, sampling_rate = lb.load(path)
# plot_wav(data, sampling_rate, emotion)
# plot_sectrogram(data, emotion)

'''FEATURE EXTRACTION'''


def extract_feature(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


# x = df['speech'].apply(lambda x: extract_feature(x))
# x = np.array([i for i in x])
# x = np.expand_dims(x, -1)
# pickle.dump(x, open('features.pkl', 'wb'))
with open('features.pkl', 'rb') as f:
    x = pickle.load(f, encoding='utf-8')

encoder = OneHotEncoder()
y = encoder.fit_transform(df[['label']])
y = y.toarray()

'''CREATE LSTM MODEL'''
model = Sequential()
model.add(LSTM(123, return_sequences=False, input_shape=(40, 1)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics='accuracy')

log_path = "Logs"
logs_callback = TensorBoard(log_path)
earlystop=EarlyStopping(monitor='val_loss',patience=150,restore_best_weights=True)
model.fit(x, y, epochs=500, validation_split=0.1, shuffle=True, callbacks=[logs_callback,earlystop])
model.save('model.h5')
