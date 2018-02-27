import h5py
from random import shuffle, seed


def phase_spliter(filename, filename_training, filename_test, seed_number=10):
    ds_test_counter = {'LPAZ-regP': 250, 'LPAZ-regS': 100, 'LPAZ-tele': 250, 'LPAZ-N': 500,
                       'URZ-regP': 250, 'URZ-regS': 250, 'URZ-tele': 250, 'URZ-N': 500}

    seed(seed_number)
    with h5py.File(filename, "r") as f:
        f_train = h5py.File(filename_training, "w")
        f_test = h5py.File(filename_test, "w")
        station_group = f["/station"]
        for station in station_group:
            phase_group = station_group[station]
            for phase in phase_group:
                arids_group = phase_group[phase]
                arids_length = len(arids_group)
                arids = list(arids_group)
                shuffle(arids)
                index = "{}-{}".format(station, phase)
                ds_test_counter[index] = min(arids_length, ds_test_counter[index])
                for arid in arids:
                    if index in ds_test_counter:
                        wavelets = arids_group[arid]
                        if ds_test_counter[index] == 0:
                            f_train.create_dataset("/station/{}/{}/{}".format(station, phase, arid), data=wavelets)
                        else:
                            f_test.create_dataset("/station/{}/{}/{}".format(station, phase, arid), data=wavelets)
                            ds_test_counter[index] -= 1

        f_train.close()
        f_test.close()


if __name__ == "__main__":
    filename = "data/phase/wavelets.hdf5"
    filename_training = "data/phase/wavelets_train.hdf5"
    filename_test = "data/phase/wavelets_test.hdf5"

    phase_spliter(filename, filename_training, filename_test)