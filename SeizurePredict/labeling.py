# Format the EEG
import pandas as pd
import os
import numpy as np

from pyedflib import highlevel
from itertools import groupby
from scipy import io


def eeg_formated(signals, names_ele):
    data_signals = signals.T # transpose the signals from datapoints
    data_signals = pd.DataFrame(data_signals) # create a pandas dataframe

    data_signals.columns = names_ele # rename columns

    return data_signals

# Format the annotations
def diagnosis(n):
    patient_A=io.loadmat(os.environ["ANNT_PATH"])["annotat_new"][0][n-1][0]
    patient_B=io.loadmat(os.environ["ANNT_PATH"])["annotat_new"][0][n-1][1]
    patient_C=io.loadmat(os.environ["ANNT_PATH"])["annotat_new"][0][n-1][2]

    #converting seconds to datapoints

    patient_A=patient_A.tolist()
    patient_B=patient_B.tolist()
    patient_C=patient_C.tolist()

    patient_A_dtp=[]
    patient_B_dtp=[]
    patient_C_dtp=[]
    for elem in patient_A:
        for i in range(int(os.environ["SAMPLING_RATE"])):
            patient_A_dtp.append(elem)
    for elem in patient_B:
        for i in range(int(os.environ["SAMPLING_RATE"])):
            patient_B_dtp.append(elem)

    for elem in patient_C:
        for i in range(int(os.environ["SAMPLING_RATE"])):
            patient_C_dtp.append(elem)

    target_=pd.DataFrame({"Diagnosis A":patient_A_dtp,"Diagnosis B":patient_B_dtp,"Diagnosis C":patient_C_dtp})

    return target_

# Create target variables when seizures lasts at least 10
def is_seizure(df):

    threshold = int(os.environ["SAMPLING_RATE"])*10

    df['is_seizure_A'] = df["Diagnosis A"].groupby((df["Diagnosis A"] != df["Diagnosis A"].shift()).cumsum()).transform('size') * df["Diagnosis A"]
    df['is_seizure_A'] = (df['is_seizure_A'] > threshold).astype(int)

    df['is_seizure_B'] = df["Diagnosis B"].groupby((df["Diagnosis B"] != df["Diagnosis B"].shift()).cumsum()).transform('size') * df["Diagnosis B"]
    df['is_seizure_B'] = (df['is_seizure_B'] > threshold).astype(int)

    df['is_seizure_C'] = df["Diagnosis C"].groupby((df["Diagnosis C"] != df["Diagnosis C"].shift()).cumsum()).transform('size') * df["Diagnosis C"]
    df['is_seizure_C'] = (df['is_seizure_C'] > threshold).astype(int)

    return df

# Create final target
def create_target(df):
    df['is_seizure_target'] = np.where(df['is_seizure_A'] + df['is_seizure_B'] + df['is_seizure_C'] >= 2, 1, 0)
    return df

# Remove useless
def remove_useless_columns(df):
    df.drop(columns=['Diagnosis A', 'Diagnosis B', 'Diagnosis C', 'is_seizure_A', 'is_seizure_B', 'is_seizure_C', 'ECG EKG', 'Resp Effort'], inplace=True)
    return df

# Final function to label
def label_data(path_raw_data, signals_preprocessed, n):

    signals, signal_headers, header = highlevel.read_edf(path_raw_data)

    names_ele = [signal_headers[iele]['label'] for iele in range(signals.shape[0])] # extract electrode names

    eeg_patient = eeg_formated(signals_preprocessed, names_ele) # format the ECG
    eeg_patient.rename(columns={'ECG EKG-REF':'ECG EKG', 'Resp Effort-REF':'Resp Effort'}, inplace=True)

    diagnosis_patient = diagnosis(n) # format the diagnosis

    data_patient = pd.merge(left=eeg_patient, right=diagnosis_patient, how='left', left_index=True, right_index=True) # merge ecg and diagnosis

    is_seizure(data_patient)
    create_target(data_patient)
    remove_useless_columns(data_patient)

    return data_patient
