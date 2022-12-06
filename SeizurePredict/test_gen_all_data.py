from pyedflib import highlevel
from scipy.fft import rfft
from scipy import signal
from imblearn.over_sampling import SMOTE

import os
import numpy as np
import pandas as pd

from SeizurePredict.filtering import filter_signals, CustomTranformer
from SeizurePredict.labeling import label_data
from SeizurePredict.featurengineer import flatten
from SeizurePredict.params import SAMPLE_RATE_DOWNSAMPLE, PATIENCES, LEN_WINDOW_DOWNSAMPLE, PATIENCES

## -- MAIN FUNCTIONS --

def preprocess_and_label(path_raw_data, scaler, patient_number, Fournier=False):

    # Load raw data
    signals, signal_headers, header = highlevel.read_edf(path_raw_data)

    # Preprocess data
    signals_preprocessed = filter_signals(signals, int(os.environ["SAMPLING_RATE"]), scaler, hp_frequency = 0.1, notch_frequency = 50, quality_factor = 30)

    if Fournier == True:
        signals_preprocessed = pd.DataFrame(np.array([abs(rfft(signals_preprocessed[i])) for i in range(len(signals_preprocessed))]))

    # Label data
    df = label_data(path_raw_data, signals_preprocessed, patient_number)

    return df

def downsampling(df):
    df_downsample = pd.DataFrame()
    all_df = pd.DataFrame()
    target_col = df.iloc[:,-1]
    num = int(0.1*(int(os.environ["SAMPLING_RATE"])))
    for i, column in enumerate(df.columns[:-1]):
        df_downsample = pd.DataFrame()
        for j in range(0,len(df)-int(os.environ["SAMPLING_RATE"])+1,int(os.environ["SAMPLING_RATE"])):
            x = np.array(df.iloc[j:j+int(os.environ["SAMPLING_RATE"]),i])
            x_resampled = pd.DataFrame(signal.resample(x, num), index=range(int(j/10),int(j/10)+num))
            df_downsample= pd.concat([df_downsample,x_resampled])
        all_df[column] = np.array(df_downsample).reshape(1,-1)[0]
    target = []
    for t in range(0,len(target_col)-int(os.environ["SAMPLING_RATE"])+1,int(os.environ["SAMPLING_RATE"])):
        for n in range(num):
            target.append(target_col[t])
    all_df["target"] = target
    return all_df

def flatten_dataframe(df):
    data = np.array([flatten(df.iloc[i:i+LEN_WINDOW_DOWNSAMPLE]) for i in range(0,len(df)-LEN_WINDOW_DOWNSAMPLE, int(os.environ["OVERLAP"])*SAMPLE_RATE_DOWNSAMPLE)])
    r=data.shape[0]
    c=data.shape[2]
    data = pd.DataFrame(data.reshape(r,c))
    return data

def concat_dataframes(dictionnary_of_dataframes):
    df = pd.concat([dictionnary_of_dataframes[i] for i in PATIENCES])
    return df

def create_x_and_y(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    return X,y

def oversampling(X, y):
    sm = SMOTE(sampling_strategy='minority', random_state=7)
    X, y = sm.fit_resample(X, y)
    return X, y

def preprocess(path_raw_data, scaler, patient_number, Fourier=False, type="test"):
    df = preprocess_and_label(path_raw_data, scaler, patient_number, Fourier)
    df = downsampling(df)
    df = flatten_dataframe(df)
    X,y = create_x_and_y(df)
    return X,y



if __name__ == "__main__":
    final_df_x = pd.DataFrame()
    final_df_y = pd.DataFrame()
    for pat in PATIENCES:
        print(pat)
        path = f"/home/lee/code/mariaaraujovitoria/SeizurePredict/raw_data/eeg{pat}.edf"
        X_test, y_test = preprocess(path, CustomTranformer(), int(pat), Fourier=False)
        final_df_x = pd.concat([final_df_x, X_test])
        final_df_y = pd.concat([final_df_y, y_test])
    final_df_x.to_csv("final_df_x.csv", index = False)
    final_df_y.to_csv("final_df_y.csv", index = False)
