''' This file explains how to read and transform the output data '''

import numpy as np
import pandas as pd

def get_4D_output(y):
    ''' Transform the output's raw 2D format to a 4D matrix format.
    
    Input
        - y: DataFrame of shape (number of samples, 10 405), including the "id_sequence" column
        
    Output:
        - Y: array of shape (number of samples, 4, 51, 51)  
    '''
    
    # Removce the "id_sequence"
    y_noid = y.drop(columns="id_sequence")
    
    # Extract output images
    Y = np.transpose(np.reshape(np.array(y_noid),(-1,4,51,51)), (0, 1, 3, 2))

    return Y



def get_2D_output(y):
    ''' Transform the output's 4D matrix format to the raw 2D format.
    
    Input
        - y: array of shape (number of samples, 4, 51, 51)
        
    Output:
        - Y: array of shape (number of samples, 10 404)
    '''
    
    # Extract inputs images
    Y = np.transpose(y, (0,1,3,2)).reshape(-1, 10404)
    Y = pd.DataFrame(Y)
    Y.insert(0, "id_sequence", Y.index)
    
    return Y


