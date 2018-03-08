import pandas as pd
import numpy as np

class PhaseFeaturesLoader(object):
    """
    PhaseLoader
    """
    phases = ['regP', 'regS', 'tele', 'N']
    phase_index = {'regP':0, 'regS':1, 'tele':2, 'N':3}
    x_indices = ['INANG1', 'INANG3', 'HMXMN', 'HVRATP', 'HVRAT', 'HTOV1', 'HTOV2', 'HTOV3', 'HTOV4', 'HTOV5',
                 'PER', 'RECT', 'PLANS', 'NAB', 'TAB', 'SLOW']
    y_indices = ['CLASS_PHASE']

    def __init__(self, filename, random_state=1, dim_x=16, batch_size=32, shuffle=True, validation_split=0.1,
                 phase_length=None, manual=False):
        """
        :param filename:
        :param random_state:
        """
        self.dataset = {}
        self.dim_x = dim_x
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.df = pd.read_csv(filepath_or_buffer=filename)
        self.phase_length = {"URZ":{'regP': 100, 'regS': 100, 'tele': 100, 'N': 300}}
        if phase_length is not None:
            self.phase_length = phase_length
        self.manual = manual

        if not manual:
            dataset = (self.df['SOURCE'] != 'M')
        else:
            dataset = True
        dataset_phases = {}
        dataset_phases_all = None
        for p in PhaseFeaturesLoader.phases:
            for s in self.phase_length:
                dp = self.df[(self.df['CLASS_PHASE'] == p) & (self.df['STA'] == s) & dataset]
                dp_length = len(dp)
                dp = dp.sample(min(dp_length, phase_length[s][p]),random_state=self.random_state)
                if p not in dataset_phases:
                    dataset_phases[p] = dp
                else:
                    dataset_phases[p] = pd.concat([dataset_phases[p], dp])
            print("length {}:{}".format(p, len(dataset_phases[p])))
            dataset_phases_all = pd.concat([dataset_phases_all, dataset_phases[p]])
        self.ids = dataset_phases_all["ARID"].values

        np.random.seed(self.random_state)
        np.random.shuffle(self.ids)
        training_number = int(len(self.ids)*(1.0-validation_split))
        self.ids_train = self.ids[0:training_number]
        self.ids_validation = self.ids[training_number:]


    def generate(self, type="train"):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            if type == "train":
                indexes = self.__get_exploration_order(self.ids_train)
            else:
                indexes = self.__get_exploration_order(self.ids_validation)
            # Generate batches
            imax = max(int(len(indexes)/self.batch_size), 1)
            for i in range(imax):
                # Find list of IDs
                if type == "train":
                    ids_temp = [self.ids_train[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                else:
                    ids_temp = [self.ids_validation[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

                # Generate data
                X, y = self.__data_generation(ids_temp)
                X = np.expand_dims(X, axis=1)
                y = np.expand_dims(y, axis=1)
                yield X, y

    def __get_exploration_order(self, ids):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(ids))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, ids):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        X[:, :] = self.df[(self.df['ARID'].isin(ids))][PhaseFeaturesLoader.x_indices].values
        # normalize the values:
        X[:, 0:2] /= 90.0
        X[:, 2:10] = np.log10(X[:, 2:10])
        phases = self.df[(self.df['ARID'].isin(ids))][PhaseFeaturesLoader.y_indices].values
        y = [PhaseFeaturesLoader.phase_index[p[0]] for p in phases]
        one_hot = self.sparsify(y, 4)
        return X, one_hot

    def get_len(self, type="train"):
        if type=="train":
            return len(self.ids_train)
        else:
            if type=="validation":
                return len(self.ids_validation)
            else:
                return 0

    def sparsify(self, y, n_classes=4):
        'Returns labels in binary NumPy array'
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                         for i in range(len(y))])

    def get_dataset(self):
        """
        :param phase_length:
            { "URZ":{'regP':10, 'regS':10, 'tele':10, 'N':30},
            "LPAZ":{'regP':10, 'regS':10, 'tele':10, 'N':30}}
        :param manual:
        :return:
        """

        dataset_x = self.df[(self.df['ARID'].isin(self.ids))][PhaseFeaturesLoader.x_indices].values
        # normalize the values:
        dataset_x[:, 0:2] /= 90.0
        dataset_x[:, 2:10] = np.log10(dataset_x[:, 2:10])
        dataset_y = self.df[(self.df['ARID'].isin(self.ids))][PhaseFeaturesLoader.y_indices].values.tolist()
        dataset_y = np.array([PhaseFeaturesLoader.phase_index[y[0]] for y in dataset_y])
        dataset_y = self.sparsify(dataset_y, len(PhaseFeaturesLoader.phases))

        # randomize the order of the datasets
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_x)
        np.random.seed(self.random_state)
        np.random.shuffle(dataset_y)

        dataset_x = np.expand_dims(dataset_x, axis=1)
        dataset_y = np.expand_dims(dataset_y, axis=1)

        return dataset_x, dataset_y


if __name__ == "__main__":
    phase_dataset = PhaseFeaturesLoader(filename="data/phase/ml_features.csv", batch_size=10)
    ds = []
    x = phase_dataset.generate
    counter = 0
    for i in x("train"):
        ds.append(i)
        if counter == 10:
            break
        counter += 1
    print(x)
    #assert len(dataset_x) == 41
    #assert len(dataset_y) == 41
    #print(len(dataset_x), len(dataset_y))