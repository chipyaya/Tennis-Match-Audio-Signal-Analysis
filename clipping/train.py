from audio2mfcc import AudioDataset
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Masking, Conv2D, MaxPooling2D, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

MFCC_SIZE = 13
MAX_LEN = 130

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mfcc-original')
    parser.add_argument('--model_name', type=str, default='CNN')
    args = parser.parse_args()
    return args

def read_data(load_exist, mode):
    audio_dir = '../data/complete_audio/'
    audio_files = ['berrettini_nadal', 'cilic_nadal', 'federer_dimitrov']
    label_dir = '../data/label/'

    if(load_exist == False):
        all_audio = []
        all_dis_flag = []
        l_map = {}
        for audio_file in audio_files:
            dataset = AudioDataset(audio_dir, label_dir, audio_file, mode)
            for i in range(len(dataset)):
                if(MAX_LEN == 130):
                    zeros = np.zeros((dataset[i]['audio'].shape[0], MAX_LEN-dataset[i]['audio'].shape[1]))
                    all_audio.append(np.concatenate((dataset[i]['audio'], zeros), axis=1))
                    all_dis_flag.append(dataset[i]['dis_flag'])
                elif(MAX_LEN == 173):
                    all_audio.append(dataset[i]['audio'])
                    all_dis_flag.append(dataset[i]['dis_flag'])

        all_audio = np.asarray(all_audio)
        all_dis_flag = np.asarray(all_dis_flag)
        print("Complete reading data")
        return all_audio, all_dis_flag

def create_nn_model():
    model = Sequential()
    model.add(Dense(32, input_dim=MFCC_SIZE, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    print("Complete creating nn model")
    return model

def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(MFCC_SIZE, MAX_LEN, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten()) 
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4), metrics=['accuracy'])
    return model

def create_rnn_model():
    training_length = 10
    model = Sequential()
    model.add(
    Embedding(input_dim=MFCC_SIZE,
              input_length = training_length,
              output_dim=100,
              trainable=True,
              mask_zero=True))
    model.add(Masking(mask_value=0.0))
    model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def train_cnn(model, all_audio, all_dis_flag, model_name):
    train_x, val_x, train_y, val_y = train_test_split(all_audio, all_dis_flag, test_size=0.2, shuffle= True, random_state=1)
    train_x = np.expand_dims(train_x, axis=3)
    val_x = np.expand_dims(val_x, axis=3)
    epochs = 500
    callbacks = [
        keras.callbacks.ModelCheckpoint("checkpoint/cnn_{epoch}.h5", monitor='val_accuracy', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    ]

    model.fit(train_x, train_y, batch_size=4, epochs=epochs, callbacks=callbacks, 
        validation_data=(val_x, val_y), shuffle=True)
    print("Guess all 0 on val set accuracy: {:.4f}".format(1-(sum(val_y)/val_y.shape[0])))
    print("cnn on val set best accuracy:")
    model.evaluate(val_x, val_y)
    
def train_nn(model, all_audio, all_dis_flag, model_name):
    train_x, val_x, train_y, val_y = train_test_split(all_audio, all_dis_flag, test_size=0.2, shuffle= True, random_state=1)
    epochs = 500
    callbacks = [
        keras.callbacks.ModelCheckpoint("checkpoint/nn_{epoch}.h5", monitor='val_accuracy', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    ]

    model.fit(train_x, train_y, batch_size=4, epochs=500, callbacks=[], 
        validation_data=(val_x, val_y), shuffle=True)
    model.fit(train_x, train_y, batch_size=4, epochs=epochs, callbacks=callbacks, 
        validation_data=(val_x, val_y), shuffle=True)
    print("Guess all 0 on val set accuracy: {:.4f}".format(1-(sum(val_y)/val_y.shape[0])))
    print("nn on val set best accuracy:")
    model.evaluate(val_x, val_y)

args = parse()
if(args.mode == "mfcc-original"):
    MAX_LEN = 130
elif(args.mode == "mfcc-delta"):
    MFCC_SIZE = 26
    MAX_LEN = 173
elif(args.mode != "mfcc-avg"):
    MAX_LEN = 173
all_audio, all_dis_flag = read_data(False, args.mode)
if(args.model_name == "CNN"):
    model = create_cnn_model()
    train_cnn(model, all_audio, all_dis_flag, args.model_name)
elif(args.model_name == "NN"):
    model = create_nn_model()
    train_nn(model, all_audio, all_dis_flag, args.model_name)
