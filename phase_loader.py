import pandas as pd
import numpy as np
from keras.utils import np_utils

class PhaseLoader(object):
    """
    PhaseLoader
    """
    phases = ['regP', 'regS', 'tele', 'N']
    phase_index = {'regP':0, 'regS':1, 'tele':2, 'N':3}
    x_indices = ['PER', 'RECT', 'PLANS', 'INANG1', 'INANG3', 'HMXMN', 'HVRATP', 'HVRAT', 'NAB', 'TAB',
                 'HTOV1', 'HTOV2', 'HTOV3', 'HTOV4', 'HTOV5', 'SLOW']
    y_indices = ['CLASS_PHASE']

    def __init__(self, filename, random_state=1):
        """
        :param filename:
        :param random_state:
        """
        self.df = pd.read_csv(filepath_or_buffer=filename)
        self.dataset = {}
        self.random_state = random_state

    def get_dataset(self, phase_length, manual=False):
        """
        :param phase_length:
            { "URZ":{'regP':10, 'regS':10, 'tele':10, 'N':30},
            "LPAZ":{'regP':10, 'regS':10, 'tele':10, 'N':30}}
        :param manual:
        :return:
        """

        if not manual:
            dataset = (self.df['SOURCE'] != 'M')
        else:
            dataset = True
        dataset_phases = {}
        dataset_phases_all = None
        for p in PhaseLoader.phases:
            for s in phase_length:
                dp = self.df[(self.df['CLASS_PHASE'] == p) & (self.df['STA'] == s) & dataset]
                dp_length = len(dp)
                dp = dp.sample(min(dp_length, phase_length[s][p]),random_state=self.random_state)
                if p not in dataset_phases:
                    dataset_phases[p] = dp
                else:
                    dataset_phases[p] = pd.concat([dataset_phases[p], dp])
            print("length {}:{}".format(p, len(dataset_phases[p])))
            dataset_phases_all = pd.concat([dataset_phases_all, dataset_phases[p]])
        dataset_x = dataset_phases_all[PhaseLoader.x_indices].values
        dataset_y = dataset_phases_all[PhaseLoader.y_indices].values.tolist()
        dataset_y = [PhaseLoader.phase_index[y[0]] for y in dataset_y]
        dataset_y = np_utils.to_categorical(dataset_y, len(PhaseLoader.phases))

        # randomize the order of the datasets
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_x)
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_y)

        return dataset_x, dataset_y


if __name__ == "__main__":
    phase_dataset = PhaseLoader(filename="data/phase/ml_features_tiny.csv")
    dataset_x, dataset_y = phase_dataset.\
        get_dataset(phase_length={ "URZ":{'regP': 5, 'regS': 5, 'tele': 5, 'N': 8},
                                   "LPAZ":{'regP': 4, 'regS': 4, 'tele': 4, 'N': 6}},
                    manual=False)
    assert len(dataset_x) == 41
    assert len(dataset_y) == 41
    print(len(dataset_x), len(dataset_y))