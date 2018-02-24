import pandas as pd
import numpy as np
from keras.utils import np_utils
import h5py
import random


class PhaseWaveletLoader(object):
    """
    PhaseWaveletLoader
    """
    phases = ['regP', 'regS']
    phase_index = {phase: index for index, phase in enumerate(phases)}

    def __init__(self, filename, random_state=1):
        """
        :param filename:
        :param random_state:
        """
        self.wvfile = h5py.File(filename, "r")
        random.seed(random_state)

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
        dataset_y = None
        for p in PhaseWaveletLoader.phases:
            for s in phase_length:
                arids_group = self.wvfile["/station/{}/{}".format(s, p)]
                arids_length = len(arids_group)
                arids = list(arids_group)
                random.shuffle(arids)
                arids_current = arids[:min(arids_length, phase_length[s][p])]
                bhe = [arids_group["{}".format(arid)][0] for arid in arids_current]
                bhz = [arids_group["{}".format(arid)][1] for arid in arids_current]
                bhn = [arids_group["{}".format(arid)][2] for arid in arids_current]
                if dataset_x_bhe is None:
                    dataset_x_bhe = bhe
                    dataset_x_bhz = bhz
                    dataset_x_bhn = bhn
                else:
                    dataset_x_bhe.extend(bhe)
                    dataset_x_bhz.extend(bhz)
                    dataset_x_bhn.extend(bhn)
                if dataset_y is None:
                    dataset_y = [PhaseWaveletLoader.phase_index[p]]*len(arids_current)
                else:
                    dataset_y.extend([PhaseWaveletLoader.phase_index[p]]*len(arids_current))

        dataset_y = np_utils.to_categorical(dataset_y, len(PhaseWaveletLoader.phases))

        return np.expand_dims(dataset_x_bhe, axis=3), np.expand_dims(dataset_x_bhz, axis=3), \
               np.expand_dims(dataset_x_bhn, axis=3), np.array(dataset_y)


if __name__ == "__main__":
    phase_wavelet = PhaseWaveletLoader(filename="data/phase/wavelets.hdf5")
    dataset_x_bhe, dataset_x_bhz, dataset_x_bhn, dataset_y = phase_wavelet.\
        get_dataset(phase_length={"URZ":{'regP': 5, 'regS': 5},
                                   "LPAZ":{'regP': 4, 'regS': 4}})
    # assert len(dataset_x) == 41
    # assert len(dataset_y) == 41
    print(len(dataset_x_bhe), len(dataset_y))