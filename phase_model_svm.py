from sklearn import svm
from sklearn.model_selection import GridSearchCV

# define baseline model
def model_svm(layers, dropout=0.1, layer_number=None):
    params_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    # num_round = 30  # the number of training iterations
    model = GridSearchCV(svm.SVC(), params_grid, cv=5, scoring='accuracy', n_jobs=10)
    return model
