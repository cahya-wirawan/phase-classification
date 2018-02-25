import argparse
import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import concatenate
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from phase_utils import print_cm
from phase_wavelet_loader import PhaseWaveletLoader


# define baseline model
def baseline_model(dropout=0.25):
    input_bhe = Input(shape=(40, 400, 1), name='input_bhe')
    conv_bhe = Conv2D(64, (3, 3), activation='relu', data_format="channels_last")(input_bhe)
    conv_bhe = Conv2D(64, (3, 3), activation='relu', data_format="channels_last")(conv_bhe)
    mp_bhe = MaxPooling2D(pool_size=(2,2))(conv_bhe)
    conv_bhe = Conv2D(32, (3, 3), activation='relu', data_format="channels_last")(mp_bhe)
    conv_bhe = Conv2D(32, (3, 3), activation='relu', data_format="channels_last")(conv_bhe)
    mp_bhe = MaxPooling2D(pool_size=(2,2))(conv_bhe)
    model_bhe = Dropout(dropout)(mp_bhe)

    input_bhz = Input(shape=(40, 400, 1), name='input_bhz')
    conv_bhz = Conv2D(64, (3, 3), activation='relu', data_format="channels_last")(input_bhz)
    conv_bhz = Conv2D(64, (3, 3), activation='relu', data_format="channels_last")(conv_bhz)
    mp_bhz = MaxPooling2D(pool_size=(2,2))(conv_bhz)
    conv_bhz = Conv2D(32, (3, 3), activation='relu', data_format="channels_last")(mp_bhz)
    conv_bhz = Conv2D(32, (3, 3), activation='relu', data_format="channels_last")(conv_bhz)
    mp_bhz = MaxPooling2D(pool_size=(2,2))(conv_bhz)
    model_bhz = Dropout(dropout)(mp_bhz)

    input_bhn = Input(shape=(40, 400, 1), name='input_bhn')
    conv_bhn = Conv2D(64, (3, 3), activation='relu', data_format="channels_last")(input_bhn)
    conv_bhn = Conv2D(64, (3, 3), activation='relu', data_format="channels_last")(conv_bhn)
    mp_bhn = MaxPooling2D(pool_size=(2,2))(conv_bhn)
    conv_bhn = Conv2D(32, (3, 3), activation='relu', data_format="channels_last")(mp_bhn)
    conv_bhn = Conv2D(32, (3, 3), activation='relu', data_format="channels_last")(conv_bhn)
    mp_bhn = MaxPooling2D(pool_size=(2,2))(conv_bhn)
    model_bhn = Dropout(dropout)(mp_bhn)

    model_concatenated = concatenate([model_bhe, model_bhz, model_bhn])
    model_flatten = Flatten()(model_concatenated)

    model_fc = Dense(128, activation='relu')(model_flatten)
    model_fc = Dense(64, activation='relu')(model_fc)
    model_fc = Dropout(dropout)(model_fc)
    model_fc = Dense(64, activation='relu')(model_fc)
    model_fc = Dropout(dropout)(model_fc)
    output = Dense(2, activation='softmax', name='output')(model_fc)

    model = Model(inputs=[input_bhe, input_bhz, input_bhn], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--action", choices=["train", "test"], default="train",
                        help="set the action, either training or test the dataset")
    parser.add_argument("--train_dataset", default="data/phase/wavelets_train.hdf5",
                        help="set the path to the training dataset")
    parser.add_argument("--test_dataset", default="data/phase/wavelets_test.hdf5",
                        help="set the path to the test dataset")
    parser.add_argument("-m", "--model", default=None,
                        help="set the path to the pre-trained model/weights")
    parser.add_argument("--cv", type=bool, default=False,
                        help="enable / disable a full cross validation with n_splits=10")
    parser.add_argument("-e", "--epochs", type=int, default=2000,
                        help="set the epochs number)")
    parser.add_argument("-d", "--dropout", type=float, default=0.1,
                        help="set the dropout)")
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help="set the verbosity)")
    parser.add_argument("-p", "--phase_length", default="URZ 1000 1000 1000 0",
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
            phase_length.update({s[0]: {"regP": int(s[1]), "regS": int(s[2]), "tele": int(s[3]), "N": int(s[4])}})
    except ValueError:
        print("It should be a list of a station name followed by four numbers.")
        exit(1)
    stations_lower = [station.lower() for station in sorted(phase_length.keys())]
    layers = []

    dropout = args.dropout
    if args.model is None:
        model_file_path = "results/phase_weights_waveform_s_{}_d_{}.hdf5".\
            format("_".join(stations_lower), dropout)
    else:
        model_file_path = args.model

    if args.action == "train":
        # load train dataset
        pwl = PhaseWaveletLoader(filename=train_dataset)
        train_x_bhe, train_x_bhz, train_x_bhn, train_y = pwl.get_dataset(phase_length=phase_length)

        tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(model_file_path, monitor='acc', verbose=args.verbose,
                                     save_best_only=True, mode='max')
        if args.cv:
            kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
            estimator = KerasClassifier(build_fn=baseline_model, dropout=dropout,
                                        epochs=epochs, batch_size=500, verbose=args.verbose)
            results = cross_val_score(estimator, [train_x_bhe, train_x_bhz, train_x_bhn], train_y, cv=kfold,
                                      fit_params={'callbacks':[checkpoint, tensorboard]})

            print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        else:
            model = baseline_model(dropout=dropout)
            print(model.summary())
            history = model.fit(x=[train_x_bhe, train_x_bhz, train_x_bhn], y=train_y,
                                batch_size=10, epochs=epochs, verbose=args.verbose,
                                validation_split=0.1, callbacks=[checkpoint, tensorboard])
            print("Max of acc: {}, val_acc: {}".
                  format(max(history.history["acc"]), max(history.history["val_acc"])))
            print("Min of loss: {}, val_loss: {}".
                  format(min(history.history["loss"]), min(history.history["val_loss"])))
    else:
        # load test dataset
        pd = PhaseWaveletLoader(filename=test_dataset)
        test_x_bhe, test_x_bhz, test_x_bhn, test_y = pd.get_dataset(phase_length=phase_length)

        # load model & weight
        loaded_model = load_model(model_file_path)
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        score = loaded_model.evaluate([test_x_bhe, test_x_bhz, test_x_bhn], test_y, verbose=0)
        prediction = loaded_model.predict([test_x_bhe, test_x_bhz, test_x_bhn], verbose=0)
        cm = confusion_matrix(test_y.argmax(axis=1), prediction.argmax(axis=1))
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
        print("Confusion matrix:")
        phases = ['regP', 'regS', 'tele', 'N']
        print_cm(cm, labels=phases)