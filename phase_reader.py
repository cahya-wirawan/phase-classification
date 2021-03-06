import random
import csv
import numpy as np
import pandas as pd
from keras.utils import np_utils


def phase_read(filename, sta, max_length_phase: {'P':100, 'S':100, 'T':100, 'N':100 }):
    phase_index = {'P':0, 'S':1, 'T':2, 'N':3}
    features = [[], [], [], []]
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            if sta != "ALL" and row[1] != sta:
                i += 1
                continue
            x = row[8:24]
            try:
                x = [float(y) for y in x]
            except:
                continue
            features[phase_index[row[4]]].append(x)
            i += 1

    phase_list = list(phase_index.keys())
    phase_list.sort()
    all_entries = 0
    for phase in phase_list:
        phase_length = len(features[phase_index[phase]])
        all_entries += min(max_length_phase[phase], phase_length)
        print("{}: {} entries".format(phase, min(max_length_phase[phase], phase_length)))
    print("Summary: {} entries".format(all_entries))

    features_compact_x = []
    features_compact_y = []
    for index in phase_index:
        phase_length = len(features[phase_index[index]])
        if phase_length == 0:
            continue
        indices = np.arange(phase_length)
        random.shuffle(indices)
        f = np.array(features[phase_index[index]])
        # i = indices[:max_length_per_phase]
        i = indices[:max_length_phase[index]]
        features_compact_x.extend(f[i])
        features_compact_y.extend([phase_index[index]]*min(max_length_phase[index], phase_length))

    features_compact_x = np.array(features_compact_x)
    # One hot labels
    features_compact_y = np_utils.to_categorical(features_compact_y, len(phase_index))

    return features_compact_x, features_compact_y


def phase_load(filename, sta, max_length_phase: {'P':100, 'S':100, 'T':100, 'N':100 }):
    phase_index = {'P':0, 'S':1, 'T':2, 'N':3}
    features = [[], [], [], []]

    df = pd.read_csv(filepath_or_buffer=filename)
    