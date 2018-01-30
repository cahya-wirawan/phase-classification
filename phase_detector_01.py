import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import random
import csv

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
max_length = 100
FILENAME="data/phase/ml_feature_bck2.csv"
STA = "LPAZ"

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
X, Y = phase_read(FILENAME, STA, {'P':1000, 'S':1000, 'T':1000, 'N':1000 })
X = np.array(X)
# One hot labels
dummy_y = np_utils.to_categorical(Y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)

kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
