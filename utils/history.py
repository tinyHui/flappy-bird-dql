import pickle as pkl


def load(fname):
    with open(fname, 'rb') as data_file:
        return pkl.load(data_file)


def write(fname, data):
    with open(fname, 'wb') as data_file:
        pkl.dump(data, data_file, protocol=4)

