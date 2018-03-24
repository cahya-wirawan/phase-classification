import argparse
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from sklearn.externals import joblib
from phase_utils import print_cm
from phase_features_loader import PhaseFeaturesLoader
from phase_model_gcforest import model_gcforest
from sklearn.metrics import classification_report
from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--action", choices=["train", "test"], default="train",
                        help="set the action, either training or test the dataset")
    parser.add_argument("--train_dataset", default="data/phase/ml_features_train.csv",
                        help="set the path to the training dataset")
    parser.add_argument("--test_dataset", default="data/phase/ml_features_test.csv",
                        help="set the path to the test dataset")
    parser.add_argument("-m", "--model", default=None,
                        help="set the path to the pre-trained or output model")
    parser.add_argument("--cv", type=bool, default=False,
                        help="enable / disable a full cross validation with n_splits=10")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="set the batch size)")
    parser.add_argument("-e", "--epochs", type=int, default=2000,
                        help="set the epochs number)")
    parser.add_argument("-l", "--layers", default="128 128 64 48 48 32 32 48 32 16",
                        help="set the hidden layers)")
    parser.add_argument("-d", "--dropout", type=float, default=0.1,
                        help="set the dropout)")
    parser.add_argument("-s", "--stations", default="URZ",
                        help="set the station name, it supports currently only LPAZ and URZ")
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help="set the verbosity)")
    parser.add_argument("-p", "--phase_length", default="URZ 6840 6840 6840 20520",
                        help="set the number of entries of phases per stations to be read from the dataset.\n" +
                             "The default is for the training, for the test use 'URZ 2280 2280 2280 6840, " +
                             "LPAZ 160 160 160 480'")

    args = parser.parse_args()

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    epochs = args.epochs
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset
    phase_length = {}
    try:
        for p in args.phase_length.split(","):
            s = p.strip().split(" ")
            phase_length.update({s[0]:{"regP": int(s[1]), "regS": int(s[2]), "tele": int(s[3]), "N": int(s[4])}})
    except ValueError:
        print("It should be a list of a station name followed by four numbers.")
        exit(1)
    stations_lower = [station.lower() for station in sorted(phase_length.keys())]
    layers = []
    try:
        layers = [int(units) for units in args.layers.split(" ")]
    except ValueError:
        print("The layers should be a list of integer, delimited by a whitespace")
        exit(1)

    dropout = args.dropout
    batch_size = args.batch_size
    validation_split = 0.1
    if args.model is None:
        model_file_path = "results/phase_svm.dat"
    else:
        model_file_path = args.model

    model = model_gcforest

    if args.action == "train":
        # load train dataset
        pd = PhaseFeaturesLoader(filename=train_dataset, validation_split=validation_split,
                                 phase_length=phase_length, batch_size=batch_size)
        tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(model_file_path, monitor='acc', verbose=args.verbose,
                                     save_best_only=True, mode='max')
        if args.cv:
            train_x, train_y = pd.get_dataset()
            kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
            estimator = KerasClassifier(build_fn=model, layers=layers, dropout=dropout,
                                        epochs=epochs, batch_size=500, verbose=args.verbose)
            results = cross_val_score(estimator, train_x, train_y, cv=kfold,
                                      fit_params={'callbacks':[checkpoint, tensorboard]})

            print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        else:
            model = model("phase_gcforest.json")
            print(model)
            train_x, train_y = pd.get_dataset(expand_dim=False, y_onehot=False)
            #sme = SMOTEENN(random_state=42)
            #train_x_res, train_y_res = sme.fit_sample(train_x, train_y)
            #class_weight = {0:1, 1:1, 2:1, 3:1}
            # train_x = np.reshape(train_x, (train_x.shape[0], train_x[2]))
            # train_y = np.reshape(train_y, (train_y.shape[0], train_y[2]))
            model.fit_transform(train_x, train_y)

            # save model to file
            joblib.dump(model, model_file_path)

            # load test dataset
            pd = PhaseFeaturesLoader(filename=test_dataset, phase_length=phase_length, batch_size=batch_size)
            test_x, test_y = pd.get_dataset(expand_dim=False, y_onehot=False)

            y_pred = model.predict(test_x)
            predictions = [np.round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(test_y, predictions)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
    else:
        # load test dataset
        pd = PhaseFeaturesLoader(filename=test_dataset, phase_length=phase_length, batch_size=batch_size)
        test_x, test_y = pd.get_dataset(expand_dim=False, y_onehot=False)

        # load model & weight
        loaded_model = joblib.load(model_file_path)
        print("Loaded model: {}".format(loaded_model))

        y_pred = loaded_model.predict(test_x)

        predictions = [np.round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(test_y, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print_report(loaded_model)
        # calculate the probability
        # y_pred = loaded_model.predict_proba(test_x)
        class_name = ["regP", "regS", "tele", "N"]
        print(classification_report_imbalanced(test_y, predictions, target_names=class_name))