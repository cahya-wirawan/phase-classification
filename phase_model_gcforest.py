from gcforest.gcforest import GCForest
from gcforest.utils.config_utils import load_json
# from sklearn.model_selection import GridSearchCV


# define baseline model
def model_gcforest(config_file):
    config = load_json(config_file)
    model = gc = GCForest(config)
    return model


if __name__ == "__main__":
    model = model_gcforest("phase_gcforest.json")
