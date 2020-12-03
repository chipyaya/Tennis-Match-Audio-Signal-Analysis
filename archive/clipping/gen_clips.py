import os
import librosa
import numpy as np
import pandas as pd
from keras.models import load_model
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import argparse
import pickle


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

def combine_consecutive_intvls(applause_intvls):
    if len(applause_intvls) == 0:
        return applause_intvls
    new_intvls = [applause_intvls[0]]
    for i in range(1, len(applause_intvls)):
        if applause_intvls[i][0] - new_intvls[-1][1] <= 2:
            new_intvls[-1] = (new_intvls[-1][0], applause_intvls[i][1])
        else:
            new_intvls.append(applause_intvls[i])
    return new_intvls

def frame_to_time(frame_index, frame_rate):
    return(frame/frame_rate)

def seconds_to_frames(s, frame_rate):
    return(int(s*frame_rate))

def collapse_to_start_and_end_frame(instance_list):
    return (instance_list[0], instance_list[-1])

def frame_span_to_time_span(frame_span, frame_rate):
    # return (frame_span[0] / frame_rate, frame_span[1] / frame_rate)
    return (round(frame_span[0] / frame_rate),
        round(frame_span[1] / frame_rate))

def seconds_to_samples(s, sr):
    return s * sr

def draw_rms(times, rms, filename):
    plt.yscale('log')
    plt.ylim(1e-3, 1e-1)
    plt.plot(times, rms)
    plt.savefig(filename)
    plt.clf()

def find_local_min_times(rms, times, threshold):
    rms[rms > threshold] = threshold
    minimum_idx = argrelextrema(rms, np.less)[0]
    return times[minimum_idx]

def select_start_times(times):
    start_times = []
    for i in range(len(times)):
        if i == 0:
            start_times.append(round(times[i]))
        else:
            if times[i] - times[i - 1] > 3:
                start_times.append(round(times[i]))
    return start_times

def detect_start_times(y, f):
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms)
    # print('video length: {} sec = {}'.format(
    #     max(times), format_time(max(times))))

    # fig_name = 'rms-{}.png'.format(f.split('/')[-1].split('.')[0])
    # draw_rms(times, rms, fig_name)

    # find local min under a threshold
    start_time_candidates = find_local_min_times(rms, times, 0.005)
    start_times = select_start_times(start_time_candidates)
    # fig_name = 'rms-clip-{}.png'.format(f.split('/')[-1].split('.')[0])
    # draw_rms(times, rms, fig_name)
    return start_times

def format_time(sec):
    return '{:02d}:{:02d}'.format(int(sec // 60), int(sec % 60))

def format_time_intvls(intvls):
    intvls_minutes = []
    for intvl in intvls:
        intvls_minutes.append((format_time(intvl[0]), format_time(intvl[1])))
    return intvls_minutes

def format_time_list(times):
    times_minutes = []
    for time in times:
        times_minutes.append(format_time(time))
    return times_minutes

def gen_final_clips(applause_intvls, start_times):
    clips = []
    if len(start_times) == 0:
        return clips
    j = 0
    start, end = start_times[0], 0
    temp_list = []
    for i in range(1, len(start_times)):
        while j < len(applause_intvls) and \
                applause_intvls[j][1] <= start_times[i]:
            temp_list.append(applause_intvls[j])
            j += 1
        if len(temp_list) > 0:
            # end = temp_list[-1][0]
            end = temp_list[0][0]
            clips.append((start, end))
            start = start_times[i]
            temp_list = []
    clips.append((start, applause_intvls[-1][0]))
    return clips

def dump_intvls(intvls, f):
    filename = 'clips-{}.p'.format(f.split('/')[-1].split('.')[0])
    pickle.dump(intvls, open('results/{}'.format(filename), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file')
    # parser.add_argument('--audio_folder')
    args = parser.parse_args()
    files = [args.audio_file]
    # audio_root = args.audio_folder
    # files = [audio_root + filename for filename in os.listdir(audio_root) \
        # if os.path.isfile(audio_root + filename)]

    model = load_model('models/applause-model.h5')

    for f in files:
        print(f'file: {f}')
        y, sr, feats = extract_features(f)
        all_features = np.array(get_feats_with_wondow(feats, 5))
        preds = model.predict_proba(all_features, batch_size=256)
        smooth_preds = pd.Series(np.transpose(preds)[0]).rolling(5).mean()[4:]
        frame_rate = len(preds) / (float(len(y)) / sr)  # preds_per_second
        applause_intvls = get_applause_instances(smooth_preds, frame_rate)
        # print('\napplause intvls:', applause_intvls)
        applause_intvls_formatted = format_time_intvls(applause_intvls)
        print('\napplause intvls:', applause_intvls_formatted)
        applause_intvls = combine_consecutive_intvls(applause_intvls)
        # print('\ncombined applause intvls:', applause_intvls)
        applause_intvls_formatted = format_time_intvls(applause_intvls)
        print('\ncombined applause intvls:', applause_intvls_formatted)
        start_times = detect_start_times(y, f)
        # print('\nstart times:', start_times)
        start_times_formatted = format_time_list(start_times)
        print('\nstart times:', start_times_formatted)
        clips = gen_final_clips(applause_intvls, start_times)
        print('\nfinal clips:', clips)
        clips_formatted = format_time_intvls(clips)
        print('\nfinal clips:', clips_formatted)
        dump_intvls(clips, f)