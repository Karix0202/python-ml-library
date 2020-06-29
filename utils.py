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
