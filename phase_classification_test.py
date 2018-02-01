import numpy as np
from keras.models import model_from_yaml
from keras.utils import np_utils
from phase_reader import phase_read

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
FILENAME="data/phase/ml_feature_bck2.csv"
STA = "LPAZ"
weight_file_path = "phase_weights_best_{}.hdf5".format(STA.lower())
model_file_path = "phase_model_{}.yaml".format(STA.lower())

# load dataset
X, Y = phase_read(FILENAME, STA, {'P': 1000, 'S': 1000, 'T': 1000, 'N': 1000 })

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
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

