import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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
X, Y = phase_read(FILENAME, STA, {'P':3000, 'S':1290, 'T':4000, 'N':5000 })
X = np.array(X)
# One hot labels
dummy_y = np_utils.to_categorical(Y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


file_path = "weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=100, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

#grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=kfold, fit_params=dict(callbacks=callbacks_list))

results = cross_val_score(estimator, X, dummy_y, cv=kfold, fit_params={'callbacks':[checkpoint]})
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
