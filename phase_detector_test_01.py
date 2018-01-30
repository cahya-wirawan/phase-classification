import numpy as np
from keras.models import model_from_yaml
from keras.utils import np_utils
import random
import csv

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
max_length = 100
FILENAME="data/phase/ml_feature_bck2.csv"
STA = "LPAZ"
weight_file_path = "phase_weights_best.hdf5"
model_file_path = "phase_model.yaml"

def phase_read(filename, sta, max_length_phase: {'P':100, 'S':100, 'T':100, 'N':100 }):
    phase_index = {'P':0, 'S':1, 'T':2, 'N':3}
    features = [[], [], [], []]
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        i = 0
        for row in reader:
            if i == 0 or row[1] != sta:
                i += 1
                continue
            x = row[8:24]
            try:
                x = [float(y) for y in x]
            except:
                continue
            features[phase_index[row[4]]].append(x)
            i += 1

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

    return features_compact_x, features_compact_y


# load dataset
X, Y = phase_read(FILENAME, STA, {'P': 1000, 'S': 1000, 'T': 1000, 'N': 1000 })
X = np.array(X)
# One hot labels
Y_one_hot = np_utils.to_categorical(Y)

# load YAML and create model
yaml_file = open(model_file_path, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights(weight_file_path)
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y_one_hot, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

