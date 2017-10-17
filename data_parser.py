import numpy as np

def parse_csv(file_name, start_col, end_col, data=float):
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(start_col,end_col), dtype=data)
    return data

def enum_classes(class_data):
    dic = list(enumerate(frozenset(class_data)))
    for i in dic:
        class_data = np.core.defchararray.replace(class_data, i[1], str(i[0]))
    return class_data.astype(int)