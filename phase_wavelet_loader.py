import pandas as pd
import numpy as np
from keras.utils import np_utils
import h5py
import random


class PhaseWaveletLoader(object):
    """
    PhaseWaveletLoader
    """
    phases = ['regP', 'regS', 'tele', 'N']
    phase_index = {phase: index for index, phase in enumerate(phases)}
    x_indices = ['PER', 'RECT', 'PLANS', 'INANG1', 'INANG3', 'HMXMN', 'HVRATP', 'HVRAT', 'NAB', 'TAB',
                 'HTOV1', 'HTOV2', 'HTOV3', 'HTOV4', 'HTOV5', 'SLOW']
    y_indices = ['CLASS_PHASE']

    def __init__(self, filename_waveform, filename_features, random_state=1):
        """
        :param filename:
        :param random_state:
        """
        self.wvfile = h5py.File(filename_waveform, "r")
        self.features = pd.read_csv(filepath_or_buffer=filename_features)
        random.seed(random_state)
        self.random_state = random_state

    def get_dataset(self, phase_length, manual=False):
        """
        :param phase_length:
            { "URZ":{'regP':10, 'regS':10, 'tele':10, 'N':30},
            "LPAZ":{'regP':10, 'regS':10, 'tele':10, 'N':30}}
        :param manual:
        :return:
        """
        dataset_x_bhe = None
        dataset_x_bhz = None
        dataset_x_bhn = None
        dataset_x_features = None
        dataset_y = None
        for s in phase_length:
            for p in PhaseWaveletLoader.phases:
                if phase_length[s][p] == 0:
                    continue
                try:
                    arids_group = self.wvfile["/station/{}/{}".format(s, p)]
                except KeyError:
                    continue
                arids_length = len(arids_group)
                arids = list(arids_group)
                random.shuffle(arids)
                try:
                    arids_current = arids[:min(arids_length, phase_length[s][p])]
                except KeyError:
                    continue
                bhe = [arids_group["{}".format(arid)][0] for arid in arids_current]
                bhz = [arids_group["{}".format(arid)][1] for arid in arids_current]
                bhn = [arids_group["{}".format(arid)][2] for arid in arids_current]

                features = [self.features[(self.features["ARID"] == int(arid))][PhaseWaveletLoader.x_indices].values
                      for arid in arids_current]

                if dataset_x_bhe is None:
                    dataset_x_bhe = bhe
                    dataset_x_bhz = bhz
                    dataset_x_bhn = bhn
                    dataset_x_features = features
                else:
                    dataset_x_bhe = np.concatenate([dataset_x_bhe, bhe])
                    dataset_x_bhz = np.concatenate([dataset_x_bhz, bhz])
                    dataset_x_bhn = np.concatenate([dataset_x_bhn, bhn])
                    dataset_x_features = np.concatenate([dataset_x_features, features])
                if dataset_y is None:
                    dataset_y = [PhaseWaveletLoader.phase_index[p]]*len(arids_current)
                else:
                    dataset_y.extend([PhaseWaveletLoader.phase_index[p]]*len(arids_current))
                print("{}/{}: {} wavelets loaded".format(s, p, len(arids_current)))
        dataset_y = np_utils.to_categorical(dataset_y, len(PhaseWaveletLoader.phases))
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_x_bhe)
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_x_bhz)
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_x_bhn)
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_x_features)
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_y)

        return np.expand_dims(dataset_x_bhe, axis=3), np.expand_dims(dataset_x_bhz, axis=3), \
               np.expand_dims(dataset_x_bhn, axis=3), np.expand_dims(dataset_x_features, axis=3), \
               np.array(dataset_y)


if __name__ == "__main__":
    phase_wavelet = PhaseWaveletLoader(filename_features="data/phase/ml_features.csv",
                                       filename_waveform="data/phase/wavelets_log.hdf5")
    dataset_x_bhe, dataset_x_bhz, dataset_x_bhn, \
    dataset_x_features, dataset_y = phase_wavelet.\
        get_dataset(phase_length={"URZ":{'regP': 5, 'regS': 5, 'tele': 5, 'N': 5},
                                   "LPAZ":{'regP': 4, 'regS': 4, 'tele': 5, 'N': 5}})
    # assert len(dataset_x) == 41
    # assert len(dataset_y) == 41
    print(len(dataset_x_bhe), len(dataset_y))