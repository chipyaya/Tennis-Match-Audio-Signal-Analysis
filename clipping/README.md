# WN-FPJ-UTCS
# Automatic clips generation using audio features
Our goal is to generate clips that do not contain audience applauses and commentator's speech.

To achieve this, we detect the start of each clip via signal volumes and assign the end of a clip to the start time of audience applauses.

The intervals of applauses are detected by a pretrained model retrieved from https://github.com/jrgillick/Applause.


## Generate clips
```
python3 gen_clips.py --audio_file ../data/audio/us-open-2019-highlights.wav
# clips dumped to results/clips-us-open-2019-highlights.p
```

## Load clips using pickle
```
import pickle
clips = pickle.load(open('results/clips-us-open-2019-highlights.p', 'rb'))
```

## Structure of clips
- clips = [(start time 0, end time 0), (start time 1, end time 1), ..]
- unit: second
- e.g. clips = [(0, 7), (14, 22), (25, 48), (50, 61), (63, 98), (107, 117), (122, 125), (132, 161), (164, 168), (174, 180)]