import numpy as np

def RandVector(v):
    """
    :param v:
    :return: an element at random using the (absolute) values of the input
    vector as selection weights.
    """
    # va = np.abs(v)
    #
    # r = np.random.rand()*np.sum(va)
    # i = 0
    # s = va[i]
    # while s < r:
    #     i += 1
    #     s += va[i]
    #
    # return i

    return np.where(np.random.multinomial(1, v) == 1)[0][0]
