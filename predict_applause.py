import os
import pickle
import librosa
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import load_model


def extract_features(f):
    try:
        y, sr = librosa.load(f)
        mfcc = librosa.feature.mfcc(y,n_mfcc=13)
        delta = librosa.feature.delta(mfcc)
        return y, sr, np.vstack([mfcc,delta])
    except:
        print(f"{f} failed")

def get_feats_with_wondow(S,window_size):
    features = []
    for i in range(window_size,S.shape[1]-window_size):
        feature = S[:,i-window_size:i+window_size]
        features.append(feature.reshape((-1)))
    return features        

def get_applause_instances(probs, frame_rate, threshold = 0.5, min_length = 10):
    instances = []
    current_list = []
    for i in range(len(probs)):
        if np.min(probs[i:i+1]) > threshold:
            current_list.append(i)
        else:
            if len(current_list) > 0:
                instances.append(current_list)
                current_list = []

    instances = [frame_span_to_time_span(collapse_to_start_and_end_frame(i), frame_rate) for i in instances if len(i) > min_length]
    return instances

def frame_to_time(frame_index, frame_rate):
    return(frame/frame_rate)

def seconds_to_frames(s, frame_rate):
    return(int(s*frame_rate))

def collapse_to_start_and_end_frame(instance_list):
    return (instance_list[0], instance_list[-1])

def frame_span_to_time_span(frame_span, frame_rate):
    return (frame_span[0] / frame_rate, frame_span[1] / frame_rate)

def seconds_to_samples(s,sr):
    return s*sr

def cut_applause_segments(instance_list,y,sr):
    new_audio = []
    for start, end in instance_list:
        sample_start = int(seconds_to_samples(start,sr))
        sample_end = int(seconds_to_samples(end,sr))
        clip = y[sample_start:sample_end]
        new_audio = np.concatenate([new_audio,clip])
    return new_audio

def normalize_X(X,means,std_devs):
    for i in range(X.shape[1]):
        X[:,i] -= means[i]
        X[:,i] /= std_devs[i]
    return X


if __name__ == '__main__':
    audio_root = 'data/audio/'
    files = [audio_root + filename for filename in os.listdir(audio_root) \
        if os.path.isfile(audio_root + filename)]

    model = load_model('models/applause-model.h5')
    # with open('cached/means.pkl','rb') as f:
    #     means = pickle.load(f)
    # with open('cached/std_devs.pkl', 'rb') as f:
    #     std_devs = pickle.load(f)
    

    for f in files:
        print(f'file{f}')
        y, sr, feats = extract_features(f)
        all_features = np.array(get_feats_with_wondow(feats, 5))
        # all_features = normalize_X(all_features,means,std_devs)
        preds = model.predict_proba(all_features, batch_size=256)
        smooth_preds = pd.Series(np.transpose(preds)[0]).rolling(5).mean()[4:]
        frame_rate = preds_per_second = len(preds) / (float(len(y))/sr)
        instances = get_applause_instances(smooth_preds, frame_rate)
        print(instances)
        # segments = cut_applause_segments(instances, y, sr)