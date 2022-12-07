import pandas as pd
import numpy as np
import os
from SeizurePredict.params import THRESHOLD
## -- FEATURE ENGINEERING --

def flatten(window_df):
    if len(np.unique(window_df.iloc[:,-1])) == 1:
        target = window_df.iloc[0,-1]
    elif np.unique(window_df.iloc[:,-1],return_counts=True)[1][1] >= THRESHOLD:
        target = 1
    else:
        target = 0
    t_df = window_df.drop(columns = "target").transpose()
    flatten = pd.DataFrame(np.array(t_df).reshape(1,t_df.shape[0]*t_df.shape[1]))
    flatten["Target"] = target
    return flatten
