import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from phase_reader import phase_read

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
FILENAME="data/phase/ml_feature_bck2_train.csv"
# possible value for STA: LPAZ, URZ and ALL
STA = "ALL"

weight_file_path = "results/phase_weights_best_{}.hdf5".format(STA.lower())
model_file_path = "results/phase_model_{}.yaml".format(STA.lower())

# define baseline model
def baseline_model():
    global model_file_path
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_file_path, "w") as yaml_file:
        yaml_file.write(model_yaml)
    return model


# load dataset
X, Y = phase_read(FILENAME, STA, {'P':6000, 'S':3000, 'T':8000, 'N':10000 })

tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(weight_file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
estimator = KerasClassifier(build_fn=baseline_model, epochs=2000, batch_size=500, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, Y, cv=kfold, fit_params={'callbacks':[checkpoint, tensorboard]})
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
