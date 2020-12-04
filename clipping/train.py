from audio2mfcc import AudioDataset
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

def read_data(load_exist):
    audio_dir = '../data/audio/'
    audio_files = ['berrettini_nadal', 'cilic_nadal', 'federer_dimitrov']
    label_dir = '../data/label/'

    if(load_exist == False):
        all_audio = []
        all_dis_flag = []
        for audio_file in audio_files:
            dataset = AudioDataset(audio_dir, label_dir, audio_file)
            for i in range(len(dataset)):
                all_audio.append(dataset[i]['audio'])
                all_dis_flag.append(dataset[i]['dis_flag'])
        all_audio = np.asarray(all_audio)
        all_dis_flag = np.asarray(all_dis_flag)
        print("Complete reading data")
        return all_audio, all_dis_flag

def split_data(all_audio, all_dis_flag):
    train_x, val_x = all_audio[:100], all_audio[100:]
    train_y, val_y = all_dis_flag[:100], all_dis_flag[100:]
    return train_x, train_y, val_x, val_y

def create_model():
    model = Sequential()
    model.add(Dense(50, input_dim=13, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])
    print("Complete creating model")
    return model

def train(model, all_audio, all_dis_flag):
    train_x, train_y, val_x, val_y = split_data(all_audio, all_dis_flag)
    print(sum(val_y))
    print(len(val_y))
    assert False
    epochs = 500
    callbacks = [
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            mode="auto",
        )
    ]


    model.fit(train_x, train_y, epochs=epochs, callbacks=callbacks, 
        validation_data=(val_x, val_y), shuffle=True)

all_audio, all_dis_flag = read_data(False)
model = create_model()
train(model, all_audio, all_dis_flag)
