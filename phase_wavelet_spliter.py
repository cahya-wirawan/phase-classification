from random import shuffle, seed


def phase_spliter(filename, filename_training, filename_test, seed_number=10):
    ds_test_counter = {'LPAZ-regP': 160, 'LPAZ-regS': 160, 'LPAZ-tele': 160, 'LPAZ-N': 480,
                'URZ-regP': 2280, 'URZ-regS': 2280, 'URZ-tele': 2280, 'URZ-N': 6840}

    seed(seed_number)
    with open(filename, "r") as file:
        f_train = open(filename_training, "w")
        f_test = open(filename_test, "w")
        line = file.readline()
        f_train.write(line)
        f_test.write(line)
        lines = file.readlines()
        shuffle(lines)
        for line in lines:
            row = line.split(',')
            index = "{}-{}".format(row[1].strip('"'), row[6].strip('"'))
            if index in ds_test_counter and row[8].strip('"') != "M":
                if ds_test_counter[index] == 0:
                    f_train.write(line)
                else:
                    f_test.write(line)
                    ds_test_counter[index] -= 1


if __name__ == "__main__":
    filename = "data/phase/ml_features.csv"
    filename_training = "data/phase/ml_features_train.csv"
    filename_test = "data/phase/ml_features_test.csv"

    phase_spliter(filename, filename_training, filename_test)