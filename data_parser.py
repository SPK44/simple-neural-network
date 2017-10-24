import numpy as np

def parse_csv(file_name, start_col, end_col, data=float):
    dat = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(start_col,end_col), dtype=data)
    return dat

def enum_classes(class_data, ordered_set):
    
    dic = list(enumerate(ordered_set))
    
    new_data = np.zeros_like(class_data,dtype=int)
    
    for j in range(len(class_data)):
        for i in dic:
            if (i[1] == class_data[j]):
                new_data[j] = i[0]
            
    return new_data