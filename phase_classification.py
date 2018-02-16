import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import model_from_yaml
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from phase_reader import phase_read
from phase_utils import print_cm
from phase_dataset import PhaseDataset

# define baseline model
def baseline_model(layers, dropout=0.1):
    global model_file_path

    # create model
    model = Sequential()
    model.add(Dense(layers[0], input_dim=16, activation='relu'))
    model.add(Dropout(dropout))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(model_file_path, "w") as yaml_file:
        yaml_file.write(model_yaml)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--action", choices=["train", "test"], default="train",
                        help="set the action, either training or test the dataset")
    parser.add_argument("--dataset", default="data/phase/ml_features.csv",
                        help="set the path to the dataset")
    parser.add_argument("--train_dataset", default="data/phase/ml_feature_bck2_train.csv",
                        help="set the path to the training dataset")
    parser.add_argument("--test_dataset", default="data/phase/ml_feature_bck2_test.csv",
                        help="set the path to the test dataset")
    parser.add_argument("-e", "--epochs", type=int, default=2000,
                        help="set the epochs number)")
    parser.add_argument("-l", "--layers", default="128 128 64 48 48 32 32 48 32 16",
                        help="set the hidden layers)")
    parser.add_argument("-d", "--dropout", type=float, default=0.1,
                        help="set the dropout)")
    parser.add_argument("-s", "--station", default="ALL",
                        help="set the station name, it supports currently only LPAZ, URZ and ALL")
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help="set the verbosity)")
    parser.add_argument("-P", type=int, default=6000,
                        help="set the number of entries of P to be read from the dataset)")
    parser.add_argument("-S", type=int, default=3000,
                        help="set the number of entries of S to be read from the dataset)")
    parser.add_argument("-T", type=int, default=8000,
                        help="set the number of entries of T to be read from the dataset)")
    parser.add_argument("-N", type=int, default=10000,
                        help="set the number of entries of N to be read from the dataset)")
    args = parser.parse_args()

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    epochs = args.epochs
    dataset = args.dataset
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    station = args.station
    weight_file_path = "results/phase_weights_best_{}.hdf5".format(station.lower())
    model_file_path = "results/phase_model_{}.yaml".format(station.lower())
    try:
        layers = [int(units) for units in args.layers.split(" ")]
    except ValueError:
        print("The layers should be a list of integer, delimited by a whitespace")
        exit(1)
    dropout = args.dropout

    pd = PhaseDataset(filename=dataset)
    train_x, train_y, test_x, test_y = pd.get_dataset(stations=["URZ"],
                                                 phase_list={'P': ['regP'], 'S': ['regS'],'T': ['tele'], 'N': ['N']})

    if args.action == "train":
        # load dataset
        # X, Y = phase_read(train_dataset, station, {'P': args.P, 'S': args.S, 'T': args.T, 'N': args.N})

        tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(weight_file_path, monitor='acc', verbose=args.verbose, save_best_only=True, mode='max')
        estimator = KerasClassifier(build_fn=baseline_model, layers=layers, dropout=dropout,
                                    epochs=epochs, batch_size=500, verbose=args.verbose)
        kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

        results = cross_val_score(estimator, train_x, train_y, cv=kfold, fit_params={'callbacks':[checkpoint, tensorboard]})
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    else:
        # load dataset
        # X, Y = phase_read(test_dataset, station, {'P': args.P, 'S': args.S, 'T': args.T, 'N': args.N})

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
        score = loaded_model.evaluate(test_x, test_y, verbose=0)
        prediction = loaded_model.predict(test_x, verbose=0)
        cm = confusion_matrix(Y.argmax(axis=1), prediction.argmax(axis=1))
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        print("Confusion matrix:")
        phases = []
        for phase in ['P', 'S', 'T', 'N']:
            if vars(args)[phase] != 0:
                phases.append(phase)
        print_cm(cm, labels=phases)