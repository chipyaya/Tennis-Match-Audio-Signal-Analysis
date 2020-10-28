import os
import pickle
import librosa
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import load_model
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import argparse


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

def get_applause_instances(probs, frame_rate, threshold=0.5, min_length=10):
    instances = []
    current_list = []
    for i in range(len(probs)):
        if np.min(probs[i:i+1]) > threshold:
            current_list.append(i)
        else:
            if len(current_list) > 0:
                instances.append(current_list)
                current_list = []

    instances = [frame_span_to_time_span(
        collapse_to_start_and_end_frame(i), frame_rate) for i in instances \
        if len(i) > min_length]
    return instances

def frame_to_time(frame_index, frame_rate):
    return(frame/frame_rate)

def seconds_to_frames(s, frame_rate):
    return(int(s*frame_rate))

def collapse_to_start_and_end_frame(instance_list):
    return (instance_list[0], instance_list[-1])

def frame_span_to_time_span(frame_span, frame_rate):
    # return (frame_span[0] / frame_rate, frame_span[1] / frame_rate)
    return (round(frame_span[0] / frame_rate), round(frame_span[1] / frame_rate))

def seconds_to_samples(s, sr):
    return s * sr

def cut_applause_segments(instance_list, y, sr):
    new_audio = []
    for start, end in instance_list:
        sample_start = int(seconds_to_samples(start, sr))
        sample_end = int(seconds_to_samples(end, sr))
        clip = y[sample_start:sample_end]
        new_audio = np.concatenate([new_audio, clip])
    return new_audio

def normalize_X(X,means,std_devs):
    for i in range(X.shape[1]):
        X[:,i] -= means[i]
        X[:,i] /= std_devs[i]
    return X

def draw_rms(times, rms, filename):
    plt.yscale('log')
    plt.ylim(1e-3, 1e-1)
    plt.plot(times, rms)
    plt.savefig(filename)
    plt.clf()

def detect_start(times):
    start_times = []
    for i in range(len(times)):
        if i == 0:
            start_times.append(round(times[i]))
        else:
            if times[i] - times[i - 1] > 3:
                start_times.append(round(times[i]))
    return start_times

def calc_rms(y, f):
    fig_name = 'rms-{}.png'.format(f.split('/')[-1].split('.')[0])
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms)
    draw_rms(times, rms, fig_name)
    print(max(times))
    print(sec_to_min(max(times)))
    THR = 0.005
    rms[rms > THR] = THR
    minimum_idx = argrelextrema(rms, np.less)[0]
    print('len(minimum_idx):', len(minimum_idx))
    print('times[minimum_idx]:', times[minimum_idx])
    start_times = detect_start(times[minimum_idx])
    print('start times:', start_times)
    fig_name = 'rms-clip-{}.png'.format(f.split('/')[-1].split('.')[0])
    draw_rms(times, rms, fig_name)
    return start_times

def sec_to_min(sec):
    return '{}m{}s'.format(int(sec // 60), int(sec % 60))

def seconds_to_minutes(intervals):
    intervals_minutes = []
    for interval in intervals:
        intervals_minutes.append((sec_to_min(interval[0]), sec_to_min(interval[1])))
    return intervals_minutes

def gen_final_segments(applause_intervals, start_times):
    segments = []
    if len(start_times) == 0:
        return segments
    j = 0
    start, end = start_times[0], 0
    temp_list = []
    for i in range(1, len(start_times)):
        while j < len(applause_intervals) and \
                applause_intervals[j][1] <= start_times[i]:
            temp_list.append(applause_intervals[j])
            j += 1
        if len(temp_list) > 0:
            end = temp_list[-1][0]
            segments.append((start, end))
            start = start_times[i]
            temp_list = []
    segments.append((start, applause_intervals[-1][0]))
    return segments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file')
    parser.add_argument('--audio_folder')
    args = parser.parse_args()
    files = [args.audio_file]
    # audio_root = args.audio_folder
    # files = [audio_root + filename for filename in os.listdir(audio_root) \
        # if os.path.isfile(audio_root + filename)]

    model = load_model('models/applause-model.h5')
    # with open('cached/means.pkl','rb') as f:
    #     means = pickle.load(f)
    # with open('cached/std_devs.pkl', 'rb') as f:
    #     std_devs = pickle.load(f)

    for f in files:
        print(f'file: {f}')
        y, sr, feats = extract_features(f)
        start_times = calc_rms(y, f)
        all_features = np.array(get_feats_with_wondow(feats, 5))
        # all_features = normalize_X(all_features,means,std_devs)
        preds = model.predict_proba(all_features, batch_size=256)
        smooth_preds = pd.Series(np.transpose(preds)[0]).rolling(5).mean()[4:]
        frame_rate = preds_per_second = len(preds) / (float(len(y))/sr)
        applause_intervals = get_applause_instances(smooth_preds, frame_rate)
        print('applause_intervals:', applause_intervals)
        applause_intervals_minutes = seconds_to_minutes(applause_intervals)
        print('applause_intervals_minutes:', applause_intervals_minutes)
        segments = gen_final_segments(applause_intervals, start_times)
        print('final segments:', segments)
        segments_minutes = seconds_to_minutes(segments)
        print('segments_minutes:', segments_minutes)
        # segments = cut_applause_segments(instances, y, sr)