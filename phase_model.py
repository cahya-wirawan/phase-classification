import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib
import xgboost as xgb
import gcforest.gcforest
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
import autosklearn.classification
from collections import Counter

from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object,), {})

class Classifier(ABC):
    __instances__ = dict()

    def __init__(self):
        Classifier.__instances__[self.__class__.__name__] = self

    def class_name(self):
        return self.__class__.__name__

    @abstractmethod
    def create_model(self, param):
        pass

    @abstractmethod
    def fit(self, x_train, y_train, verbose=0, sampling_type=None):
        pass

    @abstractmethod
    def predict(self, x_test, y_test=None, sampling_type=None):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @staticmethod
    def resample(x, y, sampling_type=None):
        x_out, y_out = x, y
        if sampling_type == "smoteenn":
            sme = SMOTEENN(random_state=1)
            x_out, y_out = sme.fit_sample(x, y)
        else:
            if sampling_type == "enn":
                enn = EditedNearestNeighbours(random_state=1)
                x_out, y_out = enn.fit_sample(x, y)

        print("Before resampling:", sorted(Counter(y).items()))
        print("After resampling:", sorted(Counter(y_out).items()))
        return x_out, y_out

    @staticmethod
    def sparsify(y, n_classes=4):
        'Returns labels in binary NumPy array'
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                         for i in range(len(y))])


class NN(Classifier):
    def __init__(self, epochs=2000, n_features=16, layers=None, dropout=0.2, seed=1, cv=False,
                 batch_size=1024, model_file_path = "results/phase_nn.hdf5"):
        super().__init__()
        self.model = None
        self.n_features = n_features
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_file_path = model_file_path
        self.seed = seed
        self.cv = cv
        if layers is None:
            self.layers = [32, 32]

        if self.cv:
            self.kfold = KFold(n_splits=10, shuffle=True, random_state=self.seed)
            self.estimator = KerasClassifier(build_fn=self.create_model, epochs=self.epochs, batch_size=self.batch_size,
                                        param={"layers": self.layers, "dropout": self.dropout, "n_features": self.n_features})
        else:
            self.model = self.create_model({"layers": self.layers, "dropout": self.dropout, "n_features": self.n_features})

    def create_model(self, param=None):
        # create model
        model = Sequential()
        model.add(Dense(param["layers"][0], input_dim=param["n_features"], activation='relu'))
        model.add(Dropout(param["dropout"]))
        for units in param["layers"][1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(param["dropout"]))
        model.add(Dense(4, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def set_layers(self, layers):
        self.layers = layers

    def fit(self, x_train, y_train, verbose=0, sampling_type=None):
        x_train, y_train = Classifier.resample(x_train, y_train, sampling_type)
        #x_train = np.expand_dims(x_train, axis=1)
        y_train = Classifier.sparsify(y_train)
        #y_train = np.expand_dims(y_train, axis=1)
        tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(self.model_file_path, monitor='acc', verbose=verbose,
                                     save_best_only=True, mode='max')
        if self.cv:
            results = cross_val_score(self.estimator, x_train, y_train, cv=self.kfold,
                                      fit_params={'callbacks':[checkpoint, tensorboard]})
            print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        else:
            history = self.model.fit(x=x_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs,
                                     verbose=verbose, validation_split=0.1, callbacks=[checkpoint, tensorboard])
            print("Max of acc: {}, val_acc: {}".
                  format(max(history.history["acc"]), max(history.history["val_acc"])))
            print("Min of loss: {}, val_loss: {}".
                  format(min(history.history["loss"]), min(history.history["val_loss"])))

    def predict(self, x_test, y_test=None, sampling_type=None):
        x_test, y_test = Classifier.resample(x_test, y_test, sampling_type)
        # x_test = np.expand_dims(x_test, axis=1)
        if y_test is not None:
            y_test = Classifier.sparsify(y_test)
            # y_test = np.expand_dims(y_test, axis=1)
            score = self.model.evaluate(x_test, y_test, verbose=0)
            print("Accuracy: {}".format(score[1]*100))
        probability = self.model.predict(x_test, verbose=0)
        return probability

    def load(self, model_file_path):
        self.model = load_model(model_file_path)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def save(self, model_file_path):
        # save model to file
        self.model.save(model_file_path)

class SVM(Classifier):
    def __init__(self):
        super().__init__()
        self.model = self.create_model({})


    def create_model(self, param):
        params_grid = [
            #{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
            #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
            {'C': [1000], 'gamma': [0.001], 'kernel': ['rbf'], 'probability': [True]}
        ]

        model = GridSearchCV(svm.SVC(), params_grid, cv=5, scoring='accuracy', n_jobs=-1)
        return model

    def fit(self, x_train, y_train, verbose=0, sampling_type=None):
        x_train, y_train = Classifier.resample(x_train, y_train, sampling_type)
        print(self.model)
        self.model.fit(x_train, y_train)

    def predict(self, x_test, y_test=None, sampling_type=None):
        x_test, y_test = Classifier.resample(x_test, y_test, sampling_type)
        probability = self.model.predict_proba(x_test)
        if y_test is not None:
            y_pred = self.model.predict(x_test)
            prediction = [np.round(value) for value in y_pred]
            accuracy = accuracy_score(y_test, prediction)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
        return probability

    def load(self, model_file_path):
        self.model = joblib.load(model_file_path)

    def save(self, model_file_path):
        # save model to file
        joblib.dump(self.model, model_file_path)


class XGBoost(Classifier):
    def __init__(self):
        super().__init__()
        self.model = self.create_model({})

    def create_model(self, param):
        seed = 10
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        # set xgboost params
        params_grid = {
            'max_depth': [5, 6, 7, 8],
            'n_estimators': [i for i in range(88, 92, 1)],
            'learning_rate': np.linspace(0.1, 1, 20),
            #'max_depth': [6],
            #'n_estimators': [i for i in range(90, 91, 1)],
            #'learning_rate': np.linspace(0.1, 1, 2),
        }

        params_fixed = {
            'objective': 'multi:softprob',
            'silent': 1,
            'n_jobs': -1,
            'verbose_eval': True
        }

        num_round = 30  # the number of training iterations

        model = GridSearchCV(
            estimator=xgb.XGBClassifier(**params_fixed, seed=seed),
            param_grid=params_grid,
            cv=cv,
            scoring='accuracy'
        )
        return model

    def fit(self, x_train, y_train, verbose=0, sampling_type=None):
        x_train, y_train = Classifier.resample(x_train, y_train, sampling_type)
        print(self.model)
        self.model.fit(x_train, y_train)

    def predict(self, x_test, y_test=None, sampling_type=None):
        x_test, y_test = Classifier.resample(x_test, y_test, sampling_type)
        probability = self.model.predict_proba(x_test)
        print(y_test.shape)
        # y_list = np.zeros(4, dtype=int)
        if y_test is not None:
            """
            for i in range(10):
                print(y_test[len(y_test)-i-1], probability[len(y_test)-i-1])
                print(x_test[len(y_test)-i-1])
            for i in range(len(y_test)):
                y_list[y_test[i]] += 1
            print(y_list)
            """
            y_pred = self.model.predict(x_test)
            prediction = [np.round(value) for value in y_pred]
            # evaluate predictions
            accuracy = accuracy_score(y_test, prediction)
            print("Accuracy: {}".format(accuracy * 100.0))
        return probability

    def load(self, model_file_path):
        self.model = joblib.load(model_file_path)

    def save(self, model_file_path):
        # save model to file
        joblib.dump(self.model, model_file_path)

class GCForest(Classifier):
    def __init__(self):
        super().__init__()
        self.model = self.create_model({})

    def create_model(self, param):
        config = {
            "cascade": {
                "random_state": 0,
                "max_layers": 100,
                "early_stopping_rounds": 3,
                "n_classes": 4,
                "estimators": [
                    {"n_folds":5,"type":"RandomForestClassifier","n_estimators":10,"max_depth":None,"n_jobs":-1},
                    {"n_folds":5,"type":"XGBClassifier","n_estimators":10,"max_depth":5,
                     "objective":"multi:softprob", "silent":True, "nthread":-1,
                     "learning_rate":0.1},
                    {"n_folds":5,"type":"ExtraTreesClassifier","n_estimators":10,"max_depth":None,"n_jobs":-1},
                    {"n_folds":5,"type":"LogisticRegression"}
                ]
            }
        }

        model = gcforest.gcforest.GCForest(config)
        return model

    def fit(self, x_train, y_train, verbose=0, sampling_type=None):
        x_train, y_train = Classifier.resample(x_train, y_train, sampling_type)
        print(self.model)
        self.model.fit_transform(x_train, y_train)

    def predict(self, x_test, y_test=None, sampling_type=None):
        x_test, y_test = Classifier.resample(x_test, y_test, sampling_type)
        probability = self.model.predict_proba(x_test)
        if y_test is not None:
            y_pred = self.model.predict(x_test)
            prediction = [np.round(value) for value in y_pred]
            accuracy = accuracy_score(y_test, prediction)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
        return probability

    def load(self, model_file_path):
        self.model = joblib.load(model_file_path)

    def save(self, model_file_path):
        # save model to file
        joblib.dump(self.model, model_file_path)


class AutoML(Classifier):
    def __init__(self):
        super().__init__()
        self.model = self.create_model({})

    def create_model(self, param):
        model = autosklearn.classification.AutoSklearnClassifier()
        return model

    def fit(self, x_train, y_train, verbose=0, sampling_type=None):
        x_train, y_train = Classifier.resample(x_train, y_train, sampling_type)
        print(self.model)
        self.model.fit(x_train, y_train)

    def predict(self, x_test, y_test=None, sampling_type=None):
        x_test, y_test = Classifier.resample(x_test, y_test, sampling_type)
        probability = self.model.predict_proba(x_test)
        if y_test is not None:
            y_pred = self.model.predict(x_test)
            prediction = [np.round(value) for value in y_pred]
            accuracy = accuracy_score(y_test, prediction)
            print("Accuracy: %.2f%%" % (accuracy * 100.0))
        return probability

    def load(self, model_file_path):
        self.model = joblib.load(model_file_path)

    def save(self, model_file_path):
        # save model to file
        joblib.dump(self.model, model_file_path)
