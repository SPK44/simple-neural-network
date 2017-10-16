import numpy as np

def parse_csv(file_name):
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    return data