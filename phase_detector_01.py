import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from phase_reader import phase_read

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
FILENAME="data/phase/ml_feature_bck2.csv"
STA = "LPAZ"

weight_file_path = "phase_weights_best.hdf5"
model_file_path = "phase_model.yaml"

# define baseline model
def baseline_model():
    global model_file_path
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
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_file_path, "w") as yaml_file:
        yaml_file.write(model_yaml)
    return model


# load dataset
X, Y = phase_read(FILENAME, STA, {'P':3000, 'S':1290, 'T':4000, 'N':5000 })

checkpoint = ModelCheckpoint(weight_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=100, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kfold, fit_params={'callbacks':[checkpoint]})
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
