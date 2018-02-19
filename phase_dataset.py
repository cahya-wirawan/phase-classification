import pandas as pd
import numpy as np
from keras.utils import np_utils

class PhaseDataset(object):
    """
    PhaseDataset
    """
    phases = ['regP', 'regS', 'tele', 'N']
    x_indices = ['PER', 'RECT', 'PLANS', 'INANG1', 'INANG3', 'HMXMN', 'HVRATP', 'HVRAT', 'NAB', 'TAB',
                 'HTOV1', 'HTOV2', 'HTOV3', 'HTOV4', 'HTOV5', 'SLOW']
    y_indices = ['CLASS_PHASE']
    phase_index = {'regP':0, 'regS':1, 'tele':2, 'N':3}

    def __init__(self, filename, random_state=1):
        self.df = pd.read_csv(filepath_or_buffer=filename)
        self.dataset_train = {}
        self.dataset_test = {}
        self.random_state = random_state

    def get_dataset(self, stations, phase_list, split_ratio=0.75, manual=False):
        """
        :param stations: list of station, example: ['URZ', 'LPAZ']
        :param phase_list: list of phases, example:
            {'P':['regP'], 'S':['regS'], 'T':['tele'], 'N':['N']}
            or
            {'PST':['regP', 'regS', 'tele'], 'N':['N']}
        :param manual:
        :return:
        """

        dataset = None
        for s in stations:
            dataset_new = (self.df['STA'] == s)
            if dataset is None:
                dataset = dataset_new
            else:
                dataset = dataset | dataset_new
        if not manual:
            dataset = (self.df['SOURCE'] != 'M') & dataset
        dataset_phases = {}
        dataset_count = {}
        for p in PhaseDataset.phases:
            dataset_phases[p] = self.df[(self.df['CLASS_PHASE'] == p) & dataset]
            dataset_count[p] = len(dataset_phases[p])
        print(dataset_count['regP'], dataset_count['regS'], dataset_count['tele'], dataset_count['N'])
        sample_PST_count = min(dataset_count['regP'], dataset_count['regS'], dataset_count['tele'])
        sample_N_count = 3*sample_PST_count

        for p in PhaseDataset.phases:
            if p == 'N':
                dataset_phases[p] = dataset_phases[p].sample(sample_N_count, random_state=self.random_state)
            else:
                dataset_phases[p] = dataset_phases[p].sample(sample_PST_count, random_state=self.random_state)

        ds = {}
        train_x = None
        train_y = None
        test_x = None
        test_y = None
        for pl in sorted(phase_list):
            ds[pl] = None
            for p in phase_list[pl]:
                if ds[pl] is None:
                    ds[pl] = dataset_phases[p]
                else:
                    ds[pl] = pd.concat([ds[pl], dataset_phases[p]])
            ds[pl] = ds[pl].sample(frac=1, random_state=self.random_state)
            print("ds {}:{}".format(pl, ds[pl].shape))
            train_length = int(split_ratio*len(ds[pl]))
            self.dataset_train[pl] = ds[pl][:train_length]
            self.dataset_test[pl] = ds[pl][train_length:]
            if train_x is None:
                train_x = self.dataset_train[pl][PhaseDataset.x_indices].values
            else:
                train_x = np.concatenate((train_x, self.dataset_train[pl][PhaseDataset.x_indices].values))
            if train_y is None:
                train_y = [PhaseDataset.phase_index[y[0]]
                           for y in self.dataset_train[pl][PhaseDataset.y_indices].values.tolist()]
            else:
                train_y = np.concatenate((train_y,
                                          [PhaseDataset.phase_index[y[0]]
                                           for y in self.dataset_train[pl][PhaseDataset.y_indices].values.tolist()]))
            if test_x is None:
                test_x = self.dataset_test[pl][PhaseDataset.x_indices].values.tolist()
            else:
                test_x = np.concatenate((test_x, self.dataset_test[pl][PhaseDataset.x_indices].values.tolist()))
            if test_y is None:
                test_y = [PhaseDataset.phase_index[y[0]]
                          for y in self.dataset_test[pl][PhaseDataset.y_indices].values.tolist()]
            else:
                test_y = np.concatenate((test_y,
                                         [PhaseDataset.phase_index[y[0]]
                                          for y in self.dataset_test[pl][PhaseDataset.y_indices].values.tolist()]))

        train_y = np_utils.to_categorical(train_y, len(phase_list))
        test_y = np_utils.to_categorical(test_y, len(phase_list))

        return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    phase_dataset = PhaseDataset(filename="data/phase/ml_features_tiny.csv")
    train_x, train_y, test_x, test_y = phase_dataset.\
        get_dataset(stations=["LPAZ"],
                    phase_list={'P':['regP'], 'S':['regS'], 'T':['tele'], 'N':['N']},
                    split_ratio=0.75, manual=False)
    #                                                        phase_list={'PST':['regP', 'regS', 'tele'], 'N':['N']})
    print(len(train_x), len(train_y), len(test_x), len(test_y))
    print(train_x)