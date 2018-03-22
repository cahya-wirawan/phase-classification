import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score


# define baseline model
def model_xgboost(layers, dropout=0.1, layer_number=None):
    seed = 10
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # set xgboost params
    params_grid = {
    }

    params_fixed = {
        'max_depth': 4,
        'n_estimators': 70,
        'learning_rate': 0.333,
        'objective': 'multi:softprob',
        'silent': 1,
        'n_jobs': 10,
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