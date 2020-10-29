import os
import pickle
import time
import sys
import argparse

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-i', '--input', help='input video')
parser.add_argument('-r', '--frame', help='frame count')
parser.add_argument('-o', '--output', help='output folder')
parser.add_argument('-c', '--clipsResult', help='clips result')
args = parser.parse_args()

clips = pickle.load(open(args.clipsResult, 'rb'))
os.system("mkdir {}".format(args.output))

for clip in clips:
    print(clip[0], clip[1])
    startTime = time.strftime('%M:%S', time.gmtime(clip[0]))
    endTime = time.strftime('%M:%S', time.gmtime(clip[1]))
    duration = time.strftime('%M:%S', time.gmtime(clip[1]-clip[0]))
    
    folder = "{}-{}-fps{}".format(startTime, endTime, args.frame)
    os.system("mkdir {}/{}".format(args.output, folder))

    command = "ffmpeg -ss {} -i {} -r {} -t {} {}/{}/%03d.png".format(startTime, args.input, args.frame, duration, args.output, folder)
    #print(command)
    os.system(command)
