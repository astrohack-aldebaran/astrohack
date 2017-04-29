# comb of very basic functions used by all

import numpy as np

def open_csv(name):
    return np.loadtxt(open(name, "r"), delimiter=",")