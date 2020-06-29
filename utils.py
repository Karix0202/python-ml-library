import numpy as np

def one_hot_encode(y):
    encoded = []
    if len(y.shape) == 1:
        for target in y:
            single = []
            for nmb in range(0, np.max(y) + 1):
                if nmb == target:
                    single.append(1)
                else:
                    single.append(0)
            encoded.append(single)
        return np.array(encoded)

    for line in y:
        single = []
        max_arg = np.argmax(line)
        for nmb in range(line.shape[0]):
            if nmb == max_arg:
                single.append(1)
            else:
                single.append(0)
        encoded.append(single)

    return np.array(encoded)


def assign_names(y, names):
    uniques = one_hot_encode(np.arange(0, y.shape[1]))
    encoded = one_hot_encode(y)
    encoded_ = list(encoded)
    for i in range(uniques.shape[0]):
        for j in range(encoded.shape[0]):
            if np.array_equal(uniques[i], encoded[j]):
                encoded_[j] = names[i]

    return encoded_
