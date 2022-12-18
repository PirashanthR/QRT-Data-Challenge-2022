"""All the functions in utils.py are provided by challenge provider in baseline
those functions haven't been modified
"""

import numpy as np
import os
import pandas as pd
import torch

def parametersTransform(A, beta, D=250, F=10):
    if A.shape != (D, F):
        print('A has not the good shape')
        return
    
    if beta.shape[0] != F:
        print('beta has not the good shape')
        return        
    
    output = np.hstack( (np.hstack([A.T, beta.reshape((F, 1))])).T )
    
    return output

def createSubmission(A, beta, score, description="", path="."):
  output = parametersTransform(A, beta)
  df_output = pd.DataFrame(output)
  df_output.to_csv(os.path.join(path,'submission_%s_%f.csv'%(description, score)))
  return df_output

def extract_AB(model):
  A = model.A.weight.data.to(torch.device("cpu")).numpy()
  B = model.B.weight.data.to(torch.device("cpu")).numpy()
  return A, B.transpose()

def checkOrthonormality(A): 
    bool = True
    D, F = A.shape 
    Error = pd.DataFrame(A.T @ A - np.eye(F)).abs()
    
    if any(Error.unstack() > 1e-6):
        bool = False
     
    return bool

def metric(A, beta, X_reshape, Y_reshape): 
    
    if not checkOrthonormality(A):
        return -1.0    
    
    Ypred = (X_reshape @ A @ beta).unstack().T
    Ytrue = Y_reshape
    Ytrue = Ytrue.div(np.sqrt((Ytrue**2).sum()), 1)    
    Ypred = Ypred.div(np.sqrt((Ypred**2).sum()), 1)

    meanOverlap = (Ytrue * Ypred).sum().mean()

    return  meanOverlap  