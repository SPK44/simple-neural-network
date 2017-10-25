import numpy as np
import csv

std = np.empty(1)
mean = np.empty(1)

def parse_csv(file_name, start_col, end_col, data=float):
    dat = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(start_col,end_col), dtype=data)
    return dat

def reg_data(x):
    global std, mean
    std = np.std(x,axis=0)
    mean = np.mean(x,axis=0)
    return ((x - mean) / std)

def apply_reg(z):
    global std, mean
    return ((z - mean) / std)

def enum_classes(class_data, ordered_set):
    
    dic = list(enumerate(ordered_set))
    
    new_data = np.zeros_like(class_data,dtype=int)
    
    for j in range(len(class_data)):
        for i in dic:
            if (i[1] == class_data[j]):
                new_data[j] = i[0]
            
    return new_data

def prep_kaggle(class_data, ordered_set, filename):
    header_l = ["Id"] + ordered_set
    r_header = ",".join(header_l)
    
    format_l = ["%u"] + ["%.4f"]*class_data.shape[1]
    
    rowId = np.arange(1,class_data.shape[0]+1).reshape(class_data.shape[0],1)
    new_data = np.concatenate((rowId,class_data),axis=1)
    np.savetxt(filename,new_data,delimiter=",",header=r_header,fmt=format_l)