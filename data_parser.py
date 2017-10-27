import numpy as np
import csv

std = np.empty(1)
mean = np.empty(1)

def parse_csv(file_name, start_col, end_col, data=float):
    '''
    Load data using Numpy's built in Loader
    
    Args:
        file_name: file to load from
        start_col: First column to grab data from
        end_col: Last column to grab data from
        data: data type (float)
        
    Returns:
        A numpy array with your data loaded (hopefully)
    '''
    dat = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1, usecols=range(start_col,end_col), dtype=data)
    return dat

def reg_data(x):
    '''
    This will regularize and preprocess your data, effective zero centering the data
    Two globals will be stored, so you apply the same regularization to test data
    
    Args:
        x: Numpy array of your training data; Only your feature vector/array
        
    Returns:
        A numpy array that is regularized
    '''
    global std, mean
    std = np.std(x,axis=0)
    mean = np.mean(x,axis=0)
    return ((x - mean) / std)

def apply_reg(z):
    '''
    This will regularize test data uisng the same parameters as the training data
    
    Args:
        x: Numpy array for test data; Only use feature vector
        
    Returns:
        A numpy array that is regularized    
    '''
    global std, mean
    return ((z - mean) / std)

def enum_classes(class_data, ordered_set):
    '''
    This will enumerate the labels that you want to train against
    
    Args:
        class_data: Numpy array of training labels
        ordered_set: A list of the label, in the order that you want
        
    Returns:
        A numpy array that contains the labels, as ints        
    '''
    dic = list(enumerate(ordered_set))
    
    new_data = np.zeros_like(class_data,dtype=int)
    
    for j in range(len(class_data)):
        for i in dic:
            if (i[1] == class_data[j]):
                new_data[j] = i[0]
            
    return new_data

def prep_kaggle(class_data, ordered_set, filename):
    '''
    This will export predictions specified by my kaggle competition.
    
    Args:
        class_data: Numpy array of test predictions
        ordered_set: A list of the labels, in the order that you want
        filename: Filename to export to
    '''
    header_l = ["Id"] + ordered_set
    r_header = ",".join(header_l)
    
    format_l = ["%u"] + ["%.4f"]*class_data.shape[1]
    
    rowId = np.arange(1,class_data.shape[0]+1).reshape(class_data.shape[0],1)
    new_data = np.concatenate((rowId,class_data),axis=1)
    np.savetxt(filename,new_data,delimiter=",",header=r_header,fmt=format_l)