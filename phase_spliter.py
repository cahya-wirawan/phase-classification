from random import shuffle


def phase_spliter(filename, filename_training, filename_test):
    counter = {'LPAZ-P': 300, 'LPAZ-S': 120, 'LPAZ-T': 200, 'LPAZ-N': 500,
                'URZ-P': 300, 'URZ-S': 120, 'URZ-T': 200, 'URZ-N': 500}


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
            index = "{}-{}".format(row[1].strip('"'), row[4].strip('"'))
            if index in counter:
                if counter[index] == 0:
                    f_train.write(line)
                else:
                    f_test.write(line)
                    counter[index] -= 1

filename = "data/phase/ml_feature_bck2.csv"
filename_training = "data/phase/ml_feature_bck2_train.csv"
filename_test = "data/phase/ml_feature_bck2_test.csv"

phase_spliter(filename, filename_training, filename_test)