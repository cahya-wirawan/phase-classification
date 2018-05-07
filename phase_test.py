import sys
import time
import phase_model as pm
from phase_features_loader import PhaseFeaturesLoader

dataset_train = "data/phase/ml_features_train.csv"
dataset_test = "data/phase/ml_features_test.csv"
dataset_train_big = "data/phase/ml_features.csv"
dataset_train_tiny = "data/phase/ml_features_tiny.csv"
dataset_train_middle = "data/phase/ml_features_train_middle.csv"
dataset_test_middle = "data/phase/ml_features_test_middle.csv"
dataset_train_relabeled = "data/phase/ml_features_train_relabeled.csv"
dataset_test_relabeled = "data/ph<ase/ml_features_test_relabeled.csv"
STA = "URZ"
phases = ["regP", "regS", "tele", "N"]

#x_indices = ['INANG1', 'INANG3', 'HMXMN', 'HVRATP', 'HVRAT', 'HTOV1', 'HTOV2', 'HTOV3', 'HTOV4', 'HTOV5',
#             'PER', 'RECT', 'PLANS', 'NAB', 'TAB', 'TIME', 'ARID']
x_indices = ['INANG1', 'INANG3', 'HMXMN', 'HVRATP', 'HVRAT', 'HTOV1', 'HTOV2', 'HTOV3', 'HTOV4', 'HTOV5',
             'PER', 'RECT', 'PLANS', 'NAB', 'TAB', 'SLOW']
phase_length_all = {"URZ": {"regP": 10000, "regS": 10000, "tele": 50000, "N": 500000}}
validation_split = 0.1
batch_size = 64
# load train dataset
# phase_length = {"URZ": {"regP": 300, "regS": 300, "tele": 300, "N": 300*3}}
phase_length = {"URZ": {"regP": 6840, "regS": 6840, "tele": 6840, "N": 6840*3}}
pd_train = PhaseFeaturesLoader(filename=dataset_train, validation_split=validation_split,
                               phase_length=phase_length, batch_size=batch_size, x_indices=x_indices)

x_train, y_train = pd_train.get_dataset(expand_dim=False, y_onehot=False)

# load test dataset
# phase_length = {"URZ": {"regP": 150, "regS": 150, "tele": 150, "N": 150*3}}
phase_length = {"URZ": {"regP": 2280, "regS": 2280, "tele": 2280, "N": 2280*3}}
pd_test = PhaseFeaturesLoader(filename=dataset_test, phase_length=phase_length, batch_size=batch_size, x_indices=x_indices)
x_test, y_test = pd_test.get_dataset(expand_dim=False, y_onehot=False)
# print(pd_test.get_phase_index(100089180))

#classifiers = ["NN", "SVM", "XGBoost", "GCForest", "AutoML"]
classifiers = ["NN"]
classifier_index = {classifier: i for i, classifier in enumerate(classifiers)}
functions = globals().copy()
classifier_class = {c: getattr(sys.modules["phase_model"], c) for c in classifiers}
print(classifier_class)
print(classifier_index)

def run_model_train_predict_all():
    for name in classifiers:
        print(classifier_class[name])
        clf = classifier_class[name]()
        sampling_type="nosampling"
        time_start = time.time()
        clf.fit(x_train, y_train, verbose=1, sampling_type=sampling_type)
        time_end = time.time()
        print("Training time: {} seconds".format(time_end - time_start))
        clf.save("results/phase_train_{}_{}.mdl".format(clf.class_name().lower(), sampling_type))
        clf.predict(x_test, y_test, sampling_type=sampling_type)

run_model_train_predict_all()
